#!/usr/bin/env python3
"""
crop_vistas_lmdb.py — Read full-image LMDB and keys, produce cropped-object and context crops in a new LMDB
"""
import argparse
import io
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
import lmdb
import json
import re
from tqdm import tqdm

def img2png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def open_env(path: Path, size: int = 50 * 1024**3) -> lmdb.Environment:
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass
    env = lmdb.open(str(path), map_size=size, subdir=True,
                    readonly=False, lock=False, readahead=False,
                    meminit=False)
    env.set_mapsize(size)
    return env


def load_keys(keys_path: Path):
    with open(keys_path, 'rb') as f:
        return pickle.load(f)


def load_targets(tgt_path: Path):
    with open(tgt_path, 'rb') as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--full_lmdb', type=Path, required=True,
                   help='Path to vistasfullIMG.lmdb')
    p.add_argument('--keys_pkl', type=Path, required=True,
                   help='Path to vistafullIMG.keys.pkl')
    p.add_argument('--target_pkl', type=Path,
                   help='Optional path to vistafullIMG.target.pkl')
    p.add_argument('--panoptic_json', type=Path, required=True,
                   help='Path to panoptic segments JSON')
    p.add_argument('--crop_lmdb', type=Path, required=True,
                   help='Output path for cropped LMDB')
    p.add_argument('--workers', type=int, default=1,
                   help='Number of parallel workers')
    args = p.parse_args()

    # load label and segment info
    pan_data = json.load(open(args.panoptic_json))
    img_ann = {ann['image_id']: ann for ann in pan_data.get('annotations', [])}
    cat_map = {c['id']: c.get('name') for c in pan_data.get('categories', [])}

    # load existing lmdb
    env_full = lmdb.open(str(args.full_lmdb), readonly=True, lock=False)
    keys = load_keys(args.keys_pkl)
    targets = load_targets(args.target_pkl) if args.target_pkl else None

    crop_env = open_env(args.crop_lmdb)

    with env_full.begin() as txn_full:
        for key in tqdm(keys, desc='Images'):
            data = txn_full.get(key)
            if data is None:
                continue
            record = pickle.loads(data)
            img_bytes = record.get('image') or record.get('ctx') or record.get('full')
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            lat, lon = record['lat'], record['lon']
            img_id = key.decode().split('_')[-1]

            ann = img_ann.get(int(img_id))
            if not ann:
                continue
            # panoptic mask
            mask_img = Image.open(Path(ann['file_name']))
            pan_arr = np.array(mask_img).astype(np.uint32)
            pan_id = pan_arr[:,:,0] + (pan_arr[:,:,1] << 8) + (pan_arr[:,:,2] << 16)

            for seg in ann.get('segments_info', []):
                seg_id = seg.get('id')
                cat_id = seg.get('category_id')
                bbox = seg.get('bbox')
                if not bbox or seg_id is None or cat_id is None:
                    continue
                x,y,w,h = map(int, bbox)
                x2, y2 = x+w, y+h
                submask = pan_id[y:y2, x:x2] == seg_id
                if not submask.any():
                    continue
                # crops
                ctx = img.crop((x,y,x2,y2))
                mf = Image.new('RGB', img.size)
                mask_img_single = Image.fromarray((pan_id==seg_id).astype('uint8')*255)
                mf.paste(img, mask=mask_img_single)
                masked = mf.crop((x,y,x2,y2))
                # prepare key and data
                label_raw = cat_map.get(cat_id, 'unk')
                label = re.sub(r"[^\w]+", '_', label_raw.lower()).strip('_')
                crop_key = f"{lat:.5f}_{lon:.5f}_{label}_{img_id}_{seg_id}".encode()
                payload = pickle.dumps({'lat':lat,'lon':lon,
                                        'ctx':img2png(ctx),'mask':img2png(masked)})
                with crop_env.begin(write=True) as txn:
                    txn.put(crop_key, payload)

    crop_env.close()
    env_full.close()
    print('Cropped LMDB created at', args.crop_lmdb)

if __name__ == '__main__':
    main()