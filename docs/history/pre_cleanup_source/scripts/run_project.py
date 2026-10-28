"""Build the Real-Minor dashboard with live API wiring and start the project."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--skip-build',action='store_true',help='Use the existing data/runtime/dashboard build')
    parser.add_argument('--check',action='store_true',help='Check prerequisites/build without starting the server')
    args=parser.parse_args()
    if not 1<=args.port<=65535:parser.error('port must be between 1 and 65535')
    os.chdir(ROOT)
    build=ROOT/'data/runtime/dashboard'
    missing=[name for name in ['fastapi','uvicorn','sqlalchemy','ultralytics','multipart','PIL'] if importlib.util.find_spec(name) is None]
    if missing:raise SystemExit('Missing packages: '+', '.join(missing)+'. Install backend/requirements.txt in the project environment.')
    for folder in [ROOT/'phase1_final',ROOT/'phase2/current_target']:
        info=json.loads((folder/'model_info.json').read_text())
        if hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest()!=info['checkpoint_sha256']:
            raise SystemExit('Model checksum mismatch: '+str(folder))
    if not (ROOT/'data/phase3_rul/scenarios.json').is_file():
        subprocess.run([sys.executable,'rul_synthesis/export_rul_data.py'],check=True)
    if not args.skip_build:
        npm=shutil.which('npm')
        if npm is None:raise SystemExit('Node.js/npm is required to build the dashboard. Use --skip-build if a build already exists.')
        # Verify and build a copy of the exact Real-Minor frontend snapshot.
        supplied=ROOT/'vendor/real_minor_frontend'
        manifest=json.loads((ROOT/'vendor/real_minor_frontend.manifest.json').read_text())
        for name,digest in manifest['sha256'].items():
            if hashlib.sha256((supplied/name).read_bytes()).hexdigest()!=digest:
                raise SystemExit('Supplied frontend checksum mismatch: '+name)
        source=ROOT/'data/runtime/frontend-source'
        if source.exists():shutil.rmtree(source)
        shutil.copytree(supplied,source)
        # Integration edits live separately; the supplied repository stays intact.
        integration=ROOT/'dashboard_integration'
        shutil.copytree(integration,source,dirs_exist_ok=True)
        dependencies=ROOT/'data/runtime/real-minor-dependencies'
        dependencies.mkdir(parents=True,exist_ok=True)
        lock_digest=hashlib.sha256((supplied/'package-lock.json').read_bytes()).hexdigest()
        marker=dependencies/'lock.sha256'
        if not (dependencies/'node_modules').is_dir() or not marker.is_file() or marker.read_text()!=lock_digest:
            for name in ['package.json','package-lock.json']:
                shutil.copy2(supplied/name,dependencies/name)
            subprocess.run([npm,'ci'],cwd=dependencies,check=True)
            marker.write_text(lock_digest)
        (source/'node_modules').symlink_to(dependencies/'node_modules',target_is_directory=True)
        environment=dict(os.environ)
        environment.setdefault('VITE_API_URL','/')
        subprocess.run([npm,'run','build','--','--outDir',str(build),'--emptyOutDir'],
                       cwd=source,env=environment,check=True)
        (build/'frontend_provenance.json').write_text(json.dumps({
            'repository':manifest['repository'],'branch':manifest['branch'],
            'commit':manifest['commit'],'source_unchanged':True,
            'integration_source':'dashboard_integration',
            'dashboard_data':'live API and persisted inspection records',
            'integration_sha256':{str(p.relative_to(integration)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(integration.rglob('*')) if p.is_file()}
        },indent=2)+'\n')
    if not (build/'index.html').is_file():raise SystemExit('Dashboard build missing. Run without --skip-build.')
    os.environ['PCB_FRONTEND_BUILD']=str(build)
    print('Verified models and dashboard ready. No training is started.',flush=True)
    if args.check:return
    print(f'Open http://127.0.0.1:{args.port}  |  API docs: /docs  |  Ctrl+C to stop',flush=True)
    os.execv(sys.executable,[sys.executable,'-m','uvicorn','backend.main:app','--host','127.0.0.1','--port',str(args.port)])


if __name__=='__main__':main()
