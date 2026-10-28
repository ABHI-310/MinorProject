from pathlib import Path
import json,base64,io
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from fastapi.testclient import TestClient
from backend.main import create_app
from backend.model_registry import ModelRegistry
root=Path.cwd();out=root/'artifacts/verification/app_false_positive_audit';out.mkdir(parents=True,exist_ok=True)
registry=ModelRegistry();app=create_app(f'sqlite:///{out}/isolated.db',registry=registry,image_store=out/'api_images')
records=[]
with TestClient(app) as client:
 for i in range(1,7):
  raw=(root/f'artifacts/verification/ui_failure_cases/input_{i}.jpeg').read_bytes()
  with Image.open(io.BytesIO(raw)) as im: bgr=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))[:,:,::-1].copy()
  for model in ['source','target']:
   response=client.post('/api/detect',files={'file':(f'input_{i}.jpeg',raw,'image/jpeg')},data={'model_type':model,'confidence':'.25','annotate':'true','equipment_id':'FP_AUDIT'})
   assert response.status_code==200,response.text
   api=response.json();direct=registry.infer(bgr,model,.25,True)
   assert api['detections']==direct['detections']
   (out/f'{model}_{i}.jpg').write_bytes(base64.b64decode(direct['annotated_image'].split(',',1)[1]))
   records.append({'image':i,'model':model,'api_matches_direct':True,'checkpoint_sha256':direct['checkpoint_sha256'],'detections':api['detections'],'above_050':[d for d in api['detections'] if d['confidence']>=.5]})
   print(i,model,[(d['class_name'],round(d['confidence'],3)) for d in api['detections']],flush=True)
(out/'results.json').write_text(json.dumps(records,indent=2))
sheet=Image.new('RGB',(1200,2880),'white');draw=ImageDraw.Draw(sheet)
for row in range(6):
 for col,name in enumerate(['source','target']):
  with Image.open(out/f'{name}_{row+1}.jpg') as im:
   im.thumbnail((590,445));sheet.paste(im,(col*600+(600-im.width)//2,row*480+30))
  draw.text((col*600+10,row*480+8),f'Photo {row+1}: app {name}, confidence >= 0.25',fill='black')
sheet.save(out/'comparison.jpg')
