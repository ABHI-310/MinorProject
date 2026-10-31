"""Install the verified K500 candidate alongside existing models (no training)."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 'c05a9c35608076e0b0af158ac4ffcabd954f3adfa6928552387e35e875615f6d'


def install(archive, destination):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip() is not None:
            raise ValueError('Invalid candidate archive')
        hashes = json.loads(z.read('checksums.json'))
        if set(names) != set(hashes) | {'checksums.json'}:
            raise ValueError('Archive inventory mismatch')
        payloads = {name: z.read(name) for name in hashes}
        for name, payload in payloads.items():
            if hashlib.sha256(payload).hexdigest() != hashes[name]:
                raise ValueError('Checksum mismatch: ' + name)
    settings = json.loads(payloads['inference_settings.json'])
    if hashlib.sha256(payloads['model.pt']).hexdigest() != EXPECTED or settings['model_sha256'] != EXPECTED:
        raise ValueError('Unexpected candidate model')
    report = json.loads(payloads['validation_nms040_report.json'])
    if report['checkpoint_sha256'] != EXPECTED:
        raise ValueError('Report/model mismatch')
    for key, value in {'imgsz': 640, 'conf': .25, 'iou': .4, 'rect': False,
                       'classes': [1, 2, 3, 4, 5], 'max_det': 300,
                       'agnostic_nms': False, 'augment': False}.items():
        if settings[key] != value:
            raise ValueError('Unexpected inference setting: ' + key)
    info = dict(checkpoint_sha256=EXPECTED, classes=settings['names'], image_size=640,
                evaluated_class_ids=settings['classes'], inference=settings,
                validation_mAP50=report['mAP50'], validation_mAP50_95=report['mAP50_95'],
                operating_point=settings['validation_operating_point'], limitations=settings['limitations'])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'model.pt').write_bytes(payloads['model.pt'])
    (destination / 'model_info.json').write_text(json.dumps(info, indent=2) + '\n')
    # Copy evidence by fixed names, never extract archive paths.
    for name in ['inference_settings.json', 'training_signature.json', 'training_results.csv',
                 'selection.json', 'validation_report.json', 'validation_nms040_report.json',
                 'confidence_threshold_report.json']:
        (destination / name).write_bytes(payloads[name])
    print('Installed verified candidate:', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    install(args.archive, ROOT / 'artifacts/models/k500')
