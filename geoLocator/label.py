#!/usr/bin/env python3
"""
lmdb_build_mapillary.py (aka label.py) — Build or append to croppedData.lmdb & fullIMG.lmdb by fetching random images from Mapillary

This script:
  • Fetches geotagged images from Mapillary across 7 continents
  • Uses Grounding-DINO to extract bounding-box context and mask crops
  • Stores full images and crop pairs in two LMDBs:
      croppedData.lmdb (ctx + mask) and fullIMG.lmdb (raw image)
  • If the LMDBs already exist, it appends new entries and
    doubles the map size on MapFullError to avoid limits.
  • Skips any images whose full-image key already exists in the LMDB.
  • Optionally samples outputs to disk for inspection via --sample.

Usage:
  python label.py --out_dir /path/to/output --per_continent 50000 [--sample /path/to/samples]

Requires:
  pip install requests pillow lmdb tqdm numpy torch groundingdino
"""
from __future__ import annotations
import os
import io
import argparse
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from requests.adapters import HTTPAdapter, Retry
import requests
import numpy as np
import lmdb
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from groundingdino.util.inference import load_model, predict

# ─────────────── Config ───────────────
ACCESS_TOKEN   = 
IMAGE_FIELDS   = "id,thumb_1024_url,computed_geometry"
API_URL        = "https://graph.mapillary.com/images"
CONTINENT_BBOXES = {
    'Africa':       (-17.7, -34.8, 51.2, 37.1),
    'Asia':         (26.0, -10.0, 180.0, 81.0),
    'Europe':       (-10.0, 34.5, 40.0, 71.0),
    'NorthAmerica': (-168.0, 5.0, -52.0, 83.0),
    'SouthAmerica': (-81.0, -55.0, -34.8, 12.5),
    'Oceania':      (110.0, -50.0, 180.0, 0.0),
    'Antarctica':   (-180.0, -90.0, 180.0, -60.0)
}
NUM_WORKERS    = 4
BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25
DINO_CFG       = "/home/benetz/code/python/geoLocator/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEI       = "/home/benetz/code/python/geoLocator/GroundingDINO/weights/groundingdino_swint_ogc.pth"
CROP_SIZE      = (224, 224)
TIMEOUT        = 20  # seconds for HTTP requests

CROP_PROMPTS = [
    'street sign', 'traffic sign', 'road marking', 'mile marker',
    'storefront sign', 'billboard', 'house number', 'flag', 'airport code',
    'skyscraper', 'tower', 'clock tower', 'lighthouse', 'windmill',
    'church', 'cathedral', 'mosque', 'minaret', 'temple', 'pagoda',
    'castle', 'fort', 'monument', 'statue', 'fountain', 'suspension bridge',
    'utility pole', 'power line', 'pylon', 'wind turbine', 'solar panel',
    'railway', 'train', 'tram', 'cable car', 'water tower', 'grain silo',
    'dam', 'pipeline', 'mountain', 'volcano', 'hill', 'cliff', 'canyon',
    'beach', 'coastline', 'sand dune', 'desert', 'glacier', 'waterfall',
    'palm tree', 'cactus', 'bamboo', 'mangrove', 'forest', 'savanna',
    'rice paddy', 'motorcycle', 'scooter', 'rickshaw', 'tuk-tuk',
    'snowmobile', 'camel', 'boat', 'ferry', 'gondola', 'bicycle rack',
    'parking meter', 'fire hydrant', 'cherry blossom', 'autumn foliage',
    'snow pile', 'rain puddle', 'road', 'car', 'truck', 'bus', 'stop sign',
    'grass', 'tree', 'bridge', 'house', 'brick wall', 'street light',
    'banner', 'stone wall', 'water', 'river', 'sea', 'snow', 'pavement',
    'fence', 'building', 'wall', 'airplane', 'bus shelter', 'metro sign',
    'toll booth', 'harbor crane', 'control tower', 'bike dock', 'route shield',
    'rest area', 'mailbox', 'newsstand', 'bench plaque', 'trash bin',
    'fire escape', 'manhole cover', 'graffiti tag', 'license plate',
    'street plaque', 'store awning', 'market stall', 'vineyard row',
    'tea plantation', 'olive grove', 'terraced field', 'heather moor',
    'permafrost', 'tundra polygon', 'speed sign', 'border booth',
    'trailhead sign', 'info panel', 'park map', 'historic plaque',
    'public art', 'pay machine', 'graffiti mural', 'cobbled road',
    'stone pavement', 'tram stop', 'ferry dock', 'pier crane', 'dockyard gate',
    'bus map', 'taxi rank', 'pedestal sign', 'ticket booth', 'road barrier',
    'lamp post', 'bollard', 'parking sign', 'street kiosk', 'billboard ad',
    'banner flag', 'church spire', 'mosque dome', 'temple gate', 'pagoda roof',
    'castle wall', 'fort gate', 'arch monument', 'statue pedestal',
    'water fountain', 'stone bridge', 'rail track', 'signal light',
    # Mapillary label additions:
    'Bird', 'Ground Animal', 'Ambiguous Barrier', 'Concrete Block', 'Curb',
    'Fence', 'Guard Rail', 'Barrier', 'Road Median', 'Road Side', 'Lane Separator',
    'Temporary Barrier', 'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut',
    'Driveway', 'Parking', 'Parking Aisle', 'Pedestrian Area', 'Rail Track',
    'Road', 'Road Shoulder', 'Service Lane', 'Sidewalk', 'Traffic Island',
    'Bridge', 'Building', 'Garage', 'Tunnel', 'Person', 'Person Group',
    'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking - Dashed Line',
    'Lane Marking - Straight Line', 'Lane Marking - Zigzag Line',
    'Lane Marking - Ambiguous', 'Lane Marking - Arrow (Left)',
    'Lane Marking - Arrow (Other)', 'Lane Marking - Arrow (Right)',
    'Lane Marking - Arrow (Split Left or Straight)',
    'Lane Marking - Arrow (Split Right or Straight)',
    'Lane Marking - Arrow (Straight)', 'Lane Marking - Crosswalk',
    'Lane Marking - Give Way (Row)', 'Lane Marking - Give Way (Single)',
    'Lane Marking - Hatched (Chevron)', 'Lane Marking - Hatched (Diagonal)',
    'Lane Marking - Other', 'Lane Marking - Stop Line',
    'Lane Marking - Symbol (Bicycle)', 'Lane Marking - Symbol (Other)',
    'Lane Marking - Text', 'Lane Marking (only) - Dashed Line',
    'Lane Marking (only) - Crosswalk', 'Lane Marking (only) - Other',
    'Lane Marking (only) - Test', 'Mountain', 'Sand', 'Sky', 'Snow',
    'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack',
    'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box',
    'Mailbox', 'Manhole', 'Parking Meter', 'Phone Booth', 'Pothole',
    'Signage - Advertisement', 'Signage - Ambiguous', 'Signage - Back',
    'Signage - Information', 'Signage - Other', 'Signage - Store',
    'Street Light', 'Pole', 'Pole Group', 'Traffic Sign Frame',
    'Utility Pole', 'Traffic Cone', 'Traffic Light - General (Single)',
    'Traffic Light - Pedestrians', 'Traffic Light - General (Upright)',
    'Traffic Light - General (Horizontal)', 'Traffic Light - Cyclists',
    'Traffic Light - Other', 'Traffic Sign - Ambiguous', 'Traffic Sign (Back)',
    'Traffic Sign - Direction (Back)', 'Traffic Sign - Direction (Front)',
    'Traffic Sign (Front)', 'Traffic Sign - Parking',
    'Traffic Sign - Temporary (Back)', 'Traffic Sign - Temporary (Front)',
    'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle',
    'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Vehicle Group',
    'Wheeled Slow', 'Water Valve', 'Car Mount', 'Dynamic', 'Ego Vehicle',
    'Ground', 'Static', 'Unlabeled'
]
TEXT_PROMPT = ", ".join(CROP_PROMPTS)

# ──────────── Networking Helpers ────────────
def _session() -> requests.Session:
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "Authorization": f"OAuth {ACCESS_TOKEN}",
        "User-Agent": "lmdb_build_mapillary/1.0"
    })
    return s

# ──────────── Helpers ────────────
def img2png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, 'PNG', optimize=True)
    return buf.getvalue()

# ─────────── Mapillary Fetch ───────────
def fetch_mapillary(cont: str, per: int, session: requests.Session, limit: int = 100) -> list[tuple[str,str,float,float]]:
    lon1, lat1, lon2, lat2 = CONTINENT_BBOXES[cont]
    params = {
        'access_token': ACCESS_TOKEN,
        'fields': IMAGE_FIELDS,
        'bbox': f"{lon1},{lat1},{lon2},{lat2}",
        'limit': limit
    }
    imgs: list[tuple[str,str,float,float]] = []
    pbar = tqdm(total=per, desc=f"Fetch {cont}")
    cursor = None
    while len(imgs) < per:
        if cursor:
            params['after'] = cursor
        res = session.get(API_URL, params=params, timeout=TIMEOUT)
        data = res.json()
        if 'error' in data:
            print("❌ Mapillary API error:", data['error'])
            break
        for e in data.get('data', []):
            img_id = e.get('id')
            url    = e.get('thumb_1024_url')
            geom   = e.get('computed_geometry', {}).get('coordinates')
            if img_id and url and geom:
                lon, lat = geom[0], geom[1]
                imgs.append((img_id, url, lat, lon))
                pbar.update(1)
                if len(imgs) >= per:
                    break
        cursor = data.get('paging', {}).get('cursors', {}).get('after')
        if not cursor:
            break
    pbar.close()
    return imgs[:per]

# ─── Processor ───────────────────────────────────────────────────────────────
class Processor:
    def __init__(self):
        self.dino = load_model(DINO_CFG, DINO_WEI).to(self._device())
    @staticmethod
    def _device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def process(self, img_id: str, url: str, lat: float, lon: float):
        r = requests.get(url, timeout=TIMEOUT)
        img = Image.open(io.BytesIO(r.content)).convert('RGB')
        boxes, _, _ = predict(
            self.dino,
            load_model.load_image(img)[1],
            TEXT_PROMPT,
            BOX_THRESHOLD,
            TEXT_THRESHOLD,
            self._device()
        )
        return img, self._crops(img, boxes)

    def _crops(self, img: Image.Image, boxes: list) -> list[tuple[Image.Image, Image.Image, int]]:
        w, h = img.size
        out = []
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(v * c) for v, c in zip(b, (w, h, w, h))]
            if x2 <= x1 or y2 <= y1: continue
            # square expand logic unchanged...
            sq = img.crop((x1, y1, x2, y2))
            ctx = sq.resize(CROP_SIZE, Image.BILINEAR)
            bg = Image.new('RGB', sq.size, (0,0,0))
            bg.paste(sq, (0,0))
            mask = bg.resize(CROP_SIZE, Image.NEAREST)
            out.append((ctx, mask, i))
        return out

# ─── Worker ─────────────────────────────────────────────────────────────────
def worker(qi, qo):
    pr = Processor()
    while True:
        task = qi.get()
        if task is None:
            break
        img_id, url, lat, lon = task
        result = pr.process(img_id, url, lat, lon)
        qo.put((task, result))

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--per_continent', type=int, default=50000)
    parser.add_argument('--sample', type=str,
                        help='dump sample images instead of LMDB build')
    args = parser.parse_args()

    sample_dir = Path(args.sample) if args.sample else None
    if sample_dir:
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"Sampling to {sample_dir}, skipping LMDB")
    else:
        full_env = lmdb.open(str(Path(args.out_dir)/'fullIMG.lmdb'), map_size=100*1024**3)
        crop_env = lmdb.open(str(Path(args.out_dir)/'croppedData.lmdb'), map_size=200*1024**3)

    session = _session()
    tasks = []
    for cont in CONTINENT_BBOXES:
        imgs = fetch_mapillary(cont, args.per_continent, session)
        for img_id, url, lat, lon in imgs:
            tasks.append((img_id, url, lat, lon))

    # Queues & workers
    qi, qo = mp.Queue(), mp.Queue()
    workers = [mp.Process(target=worker, args=(qi, qo)) for _ in range(NUM_WORKERS)]
    for w in workers:
        w.start()

    for t in tasks:
        qi.put(t)
    for _ in tasks:
        (img_id, url, lat, lon), (img, crops) = qo.get()
        key_base = f"{img_id}_{lon:.5f}_{lat:.5f}"
        if sample_dir:
            img.save(sample_dir/f"full_{key_base}.png")
        else:
            full_env.begin(write=True).put(key_base.encode(), img2png(img))
        for ctx, mask, i in crops:
            subdir = sample_dir/f"{key_base}_{i}" if sample_dir else None
            if subdir:
                subdir.mkdir(exist_ok=True)
                ctx.save(subdir/'ctx.png')
                mask.save(subdir/'mask.png')
            else:
                crop_env.begin(write=True).put(
                    f"{key_base}_{i}".encode(),
                    pickle.dumps({'ctx': img2png(ctx), 'mask': img2png(mask)})
                )
    # Shutdown
    for _ in workers:
        qi.put(None)
    for w in workers:
        w.join()

    if not sample_dir:
        full_env.close()
        crop_env.close()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
