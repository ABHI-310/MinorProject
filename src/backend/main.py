"""Verified PCB inference, persistent inspection history and labelled simulated RUL."""
from __future__ import annotations
from backend.maintenance import quality_trend, synthetic_review_demo

from contextlib import asynccontextmanager
from functools import lru_cache
import hashlib
import io
import json
import os
import re
from pathlib import Path
import sys
from uuid import UUID,uuid4

import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from backend.model_registry import ModelRegistry
from config import RUL_DIR
from database.db import Database,EventConflict
from rul_synthesis.export_rul_data import lifecycle_rows

EQUIPMENT_PATTERN=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,49}$'
CLASSES={'missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper'}
MAX_UPLOAD=10*1024*1024


class EdgeDetection(BaseModel):
    model_config=ConfigDict(extra='forbid')
    class_name: str
    confidence: float=Field(ge=0,le=1,allow_inf_nan=False)
    bbox: list[float]=Field(min_length=4,max_length=4)

    @field_validator('class_name')
    @classmethod
    def valid_class(cls,value):
        value='open_circuit' if value=='open' else value
        if value not in CLASSES:raise ValueError('Unknown PCB defect class')
        return value

    @field_validator('bbox')
    @classmethod
    def valid_box(cls,value):
        x1,y1,x2,y2=value
        if not np.isfinite(value).all() or min(value)<0 or x2<=x1 or y2<=y1:
            raise ValueError('bbox must be finite nonnegative xyxy coordinates with positive area')
        return value


class EdgeEvent(BaseModel):
    model_config=ConfigDict(extra='forbid')
    equipment_id: str=Field(default='EQP_001',pattern=EQUIPMENT_PATTERN)
    event_id: UUID | None=None
    boards_inspected: int | None=Field(default=None,ge=1,le=1000000)
    detections: list[EdgeDetection]=Field(max_length=10000)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def frontend_compatible(response,request):
    """The supplied frontend understands an available RUL object or null."""
    if not request.url.path.startswith('/api/') and response.get('rul') is not None:
        if response['rul'].get('health_score') is None:
            response['rul_status']=response['rul']
            response['rul']=None
    return response


@lru_cache(maxsize=1)
def synthetic_scenarios():
    path=RUL_DIR/'scenarios.json'
    if not path.is_file():raise HTTPException(503,'Generate the Phase 3 dataset first.')
    payload=json.loads(path.read_text())
    return {s['equipment_id']:s for s in payload['scenarios']}


def create_app(database_url=None,registry=None,image_store=None):
    database=Database(database_url);registry=registry or ModelRegistry()

    @asynccontextmanager
    async def lifespan(app):
        database.initialise();database.register_models(registry.list_models())
        yield
        database.engine.dispose()

    app=FastAPI(title='PCB Smart Maintenance API',version='2.0.0',lifespan=lifespan)
    api=APIRouter()
    app.state.db=database;app.state.registry=registry
    image_store=Path(image_store or os.getenv('PCB_IMAGE_STORE',str(ROOT/'data/application/inspection_images')))
    app.state.image_store=image_store
    app.add_middleware(CORSMiddleware,allow_origins=[s.strip() for s in os.getenv(
        'CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000').split(',') if s.strip()],
        allow_credentials=True,allow_methods=['GET','POST'],allow_headers=['*'])

    def persist(**kwargs):
        try:return database.record_inspection(**kwargs)
        except EventConflict as error:raise HTTPException(409,str(error)) from error
        except SQLAlchemyError as error:raise HTTPException(503,'Database write failed; inspection was not accepted.') from error

    @api.get('/health')
    def health():
        try:
            with database.session() as session:session.execute(text('SELECT 1'))
        except SQLAlchemyError as error:raise HTTPException(503,'Database unavailable') from error
        return {'status':'ok','database':database.engine.dialect.name,
                'available_models':[m['model_type'] for m in registry.list_models() if m['available']]}

    @api.get('/models')
    def models():return {'models':registry.list_models(),'comparison_note':'Models use different evaluation datasets; scores are not a controlled source-versus-target comparison.'}

    @api.post('/detect')
    async def detect(request: Request,file: UploadFile=File(...),equipment_id: str=Form('EQP_001',pattern=EQUIPMENT_PATTERN),
                     model_type: str=Form('source'),confidence: float=Form(.25,ge=.001,le=1,allow_inf_nan=False),
                     boards_inspected: int | None=Form(None,ge=1,le=1000000),
                     event_id: UUID | None=Form(None),annotate: bool=Form(True)):
        try:model_type=registry.canonical(model_type)
        except ValueError as error:raise HTTPException(422,str(error)) from error
        raw=await file.read(MAX_UPLOAD+1)
        if len(raw)>MAX_UPLOAD:raise HTTPException(413,'Image must be at most 10 MB.')
        if not raw:raise HTTPException(400,'Empty image upload.')
        request_hash=fingerprint({'image_sha256':hashlib.sha256(raw).hexdigest(),'equipment_id':equipment_id,
            'model_type':model_type,'confidence':confidence,'boards_inspected':boards_inspected})
        if event_id:
            try:prior=await run_in_threadpool(database.replay,event_id,request_hash)
            except EventConflict as error:raise HTTPException(409,str(error)) from error
            if prior:return frontend_compatible({**prior,'annotated_image':None},request)
        try:
            from PIL import Image,ImageOps,UnidentifiedImageError
            with Image.open(io.BytesIO(raw)) as decoded:
                if decoded.format not in ('PNG','JPEG'):raise ValueError('Only JPEG and PNG are supported.')
                if decoded.width*decoded.height>24000000:raise HTTPException(413,'Image must be at most 24 megapixels.')
                image=np.asarray(ImageOps.exif_transpose(decoded).convert('RGB'))[:,:,::-1].copy()
        except HTTPException:raise
        except (ValueError,OSError,Image.DecompressionBombError) as error:
            raise HTTPException(400,'Upload a valid JPEG or PNG image.') from error
        try:inferred=await run_in_threadpool(registry.infer,image,model_type,confidence,annotate)
        except (OSError,ValueError,RuntimeError) as error:
            raise HTTPException(503,'Verified model could not run; check server model files and PCB_DEVICE.') from error
        detections=inferred['detections']
        details={k:v for k,v in inferred.items() if k not in ('detections','annotated_image')}
        # Keep normalized images so persisted inspection cards survive a reload.
        from PIL import Image
        import base64
        image_store.mkdir(parents=True,exist_ok=True)
        output=io.BytesIO();Image.fromarray(image[:,:,::-1]).save(output,format='JPEG',quality=92)
        assets={'original':output.getvalue()}
        if inferred['annotated_image']:
            assets['annotated']=base64.b64decode(inferred['annotated_image'].split(',',1)[1])
        for kind,content in assets.items():
            destination=image_store/f'{request_hash}_{kind}.jpg'
            temporary=image_store/f'.{uuid4().hex}.tmp'
            try:
                temporary.write_bytes(content);os.replace(temporary,destination)
            except OSError as error:
                temporary.unlink(missing_ok=True)
                raise HTTPException(503,'Could not save inspection image; check storage permissions and free space.') from error
        details['image_url']=f'/api/inspection-images/{request_hash}/original'
        details['annotated_image_url']=f'/api/inspection-images/{request_hash}/annotated' if 'annotated' in assets else None
        result=await run_in_threadpool(persist,equipment_id=equipment_id,detections=detections,
            model_type=model_type,boards_inspected=boards_inspected,request_hash=request_hash,event_id=event_id,
            image_path=Path(file.filename or 'upload').name[:255],details=details)
        return frontend_compatible({**result,'annotated_image':inferred['annotated_image']},request)

    @api.post('/events/detections')
    def edge_event(event: EdgeEvent):
        data=event.model_dump(mode='json');data.pop('event_id')
        result=persist(equipment_id=event.equipment_id,detections=data['detections'],model_type='edge',
            boards_inspected=event.boards_inspected,event_id=event.event_id,request_hash=fingerprint(data))
        return {**result,'accepted':result['count']}

    @api.get('/metrics')
    def metrics(request: Request,equipment_id: str=Query('EQP_001',pattern=EQUIPMENT_PATTERN),hours: int=Query(24,ge=1,le=8760)):
        result=database.latest_metrics(equipment_id,hours)
        result['models']=registry.list_models()
        return frontend_compatible(result,request)

    @api.get('/dashboard')
    def dashboard(equipment_id: str=Query('EQP_001',pattern=EQUIPMENT_PATTERN),days: int=Query(30,ge=1,le=30)):
        result=database.dashboard_metrics(equipment_id,days)
        result['models']=registry.list_models()
        result['maintenance_trend']=quality_trend(result['history'])
        result['health']=health()
        result['health']['device']=os.getenv('PCB_DEVICE','cpu')
        return result

    @api.get('/inspection-images/{image_id}/{kind}',include_in_schema=False)
    def inspection_image(image_id: str,kind: str):
        if not re.fullmatch(r'[a-f0-9]{64}',image_id) or kind not in ('original','annotated'):
            raise HTTPException(404,'Inspection image not found.')
        path=image_store/f'{image_id}_{kind}.jpg'
        if not path.is_file():raise HTTPException(404,'Inspection image not found.')
        return FileResponse(path,media_type='image/jpeg')

    @api.get('/rul/synthetic/{equipment_id}')
    def synthetic_rul(equipment_id: str,limit: int=Query(500,ge=1,le=500)):
        item=synthetic_scenarios().get(equipment_id)
        if item is None:raise HTTPException(404,'Synthetic equipment ID not found.')
        return {'equipment_id':equipment_id,'synthetic':True,'failure_observed':item['failure_observed'],
                'failure_cycle':item['failure_cycle'],'cycle_unit':'100 simulated boards',
                'timeline':lifecycle_rows(item,'simulation')[:limit]}

    @api.get('/maintenance/demo')
    def maintenance_demo():
        return synthetic_review_demo()

    @api.post('/adapt',status_code=410)
    def retired_training():
        raise HTTPException(410,'Training is offline in the verified Kaggle notebooks; this API does not start training.')

    app.include_router(api,prefix='/api')
    app.include_router(api,include_in_schema=False)  # Existing edge/CLI clients keep working.
    @app.get('/demo/pcb.jpg',include_in_schema=False)
    def demo_image():
        sample=ROOT/'examples/pcb.jpg'
        if not sample.is_file():raise HTTPException(404,'Sample image is unavailable.')
        return FileResponse(sample,media_type='image/jpeg')

    frontend=Path(os.getenv('PCB_FRONTEND_BUILD',str(ROOT/'data/runtime/dashboard')))
    if (frontend/'index.html').is_file():
        app.mount('/',StaticFiles(directory=frontend,html=True),name='dashboard')
    return app


app=create_app()
