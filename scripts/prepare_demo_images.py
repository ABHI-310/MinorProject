"""Prepare ten fixed validation examples without selecting on predictions."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    package=ROOT/'artifacts/kaggle/phase2_v2/package'
    manifest=json.loads((package/'manifest.json').read_text())
    rows=sorted((r for r in manifest['views'] if r['dataset']=='target' and r['split']=='val' and r['kind']=='full'),key=lambda r:r['image'])
    rng=np.random.default_rng(42);used=set();selected=[]
    for cid in range(1,6):
        pool=[r for r in rows if r['image'] not in used and any(int(b[0])==cid for b in r['labels'])]
        for i in rng.choice(len(pool),size=2,replace=False):
            row=pool[int(i)];used.add(row['image']);selected.append((cid,row))
    out=ROOT/'examples/k500_demo'
    assert not out.exists(),'Demo folder already exists; preserve its fixed selection.'
    for _,r in selected:
        for key in ('image','label'):
            assert hashlib.sha256((package/r[key]).read_bytes()).hexdigest()==r[key+'_sha256']
    out.mkdir();records=[]
    for i,(cid,r) in enumerate(selected,1):
        prefix=f'{i:02d}_{manifest["names"][cid]}'
        image_name=prefix+Path(r['image']).suffix;label_name=prefix+'.txt'
        shutil.copyfile(package/r['image'],out/image_name);shutil.copyfile(package/r['label'],out/label_name)
        records.append(dict(image=image_name,label=label_name,original=r['image'],image_sha256=r['image_sha256'],
                            width=r['width'],height=r['height'],labels=r['labels'],
                            expected_counts={manifest['names'][c]:sum(int(b[0])==c for b in r['labels']) for c in range(1,6) if any(int(b[0])==c for b in r['labels'])}))
    (out/'manifest.json').write_text(json.dumps(dict(role='Existing validation examples, not independent test data',selection='Seed 42; two per class; exclude reused images; no prediction-based selection',names=manifest['names'],images=records),indent=2)+'\n')
    print('Prepared',len(records),'verified image/label pairs in',out)
    for r in records:print(r['image'],r['expected_counts'])

if __name__=='__main__':main()
