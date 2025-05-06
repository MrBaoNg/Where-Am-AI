#!/usr/bin/env python3
"""
geo_infer.py — Production inference: Grounding DINO + crop/full ensemble.
"""
import argparse, math, json
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F
import timm
from torchvision import transforms
from torchvision.transforms import functional as TF
from groundingdino.util.inference import load_model, predict, load_image
import warnings
warnings.filterwarnings("ignore")
# ── Constants ─────────────────────────────────────────────────────────────
RAD = math.pi / 180.0
CROP_SIZE = (224, 224)
TEXT_PROMPT   = "object"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD= 0.25

IM_MEAN = (0.485,0.456,0.406); IM_STD = (0.229,0.224,0.225)
CROP_MEAN = IM_MEAN+IM_MEAN; CROP_STD = IM_STD+IM_STD

# ── Helpers ───────────────────────────────────────────────────────────────
def xyz_to_latlon(xyz):
    x,y,z = xyz[...,0], xyz[...,1], xyz[...,2]
    lat = torch.asin(torch.clamp(z, -1, 1)) / RAD
    lon = torch.atan2(y, x) / RAD
    return torch.stack([lat, lon], dim=-1)

def center_crop(img, size=224):
    w,h = img.size
    if w < size or h < size:
        scale = size/min(w,h)
        img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
        w,h = img.size
    left, top = (w-size)//2, (h-size)//2
    return img.crop((left, top, left+size, top+size))

# ── Transforms ───────────────────────────────────────────────────────────
context_tf = transforms.Compose([
    transforms.Lambda(lambda img: center_crop(img)),
    transforms.RandomPerspective(0.2, p=0.5),
    transforms.GaussianBlur(5, sigma=(0.1,2.0)),
    transforms.ToTensor()
])
mask_tf = transforms.Compose([
    transforms.Lambda(lambda img: TF.pad(
        img,
        ((max(img.size)-img.size[0])//2,
         (max(img.size)-img.size[1])//2,
         0, 0),
        fill=0
    )),
    transforms.Lambda(lambda img: center_crop(img)),
    transforms.ToTensor()
])
full_tf = transforms.Compose([
    transforms.Lambda(lambda img: center_crop(img)),
    transforms.RandomPerspective(0.2, p=0.5),
    transforms.GaussianBlur(5, sigma=(0.1,2.0)),
    transforms.ToTensor()
])
normalize_crop = transforms.Normalize(CROP_MEAN, CROP_STD)
normalize_full = transforms.Normalize(IM_MEAN, IM_STD)

# ── Inference Class ──────────────────────────────────────────────────────
class GeoInfer:
    def __init__(self, dino_cfg, dino_w, crop_ckpt, full_ckpt,
                 crop_arch, full_arch, device):
        self.device = device
        # Grounding DINO
        self.dino = load_model(dino_cfg, dino_w).to(device).eval()
        # Crop branch
        self.crop_net = timm.create_model(crop_arch, pretrained=False,
                                          num_classes=3, in_chans=6).to(device).eval()
        ck = torch.load(crop_ckpt, map_location=device)['model_state_dict']
        self.crop_net.load_state_dict(ck, strict=False)
        # Full branch
        self.full_net = timm.create_model(full_arch, pretrained=False,
                                          num_classes=3, in_chans=3).to(device).eval()
        ck2 = torch.load(full_ckpt, map_location=device)['model_state_dict']
        self.full_net.load_state_dict(ck2, strict=False)

    def _get_boxes(self, img, img_path: Path):
        # DINO expects a path or file-like; we pass the path
        tensor = load_image(str(img_path))[1].to(self.device)
        boxes, _, _ = predict(
            self.dino,
            tensor,
            TEXT_PROMPT, BOX_THRESHOLD,
            TEXT_THRESHOLD, self.device
        )
        return boxes

    def predict(self, img_path: Path):
        img = Image.open(img_path).convert('RGB')
        boxes = self._get_boxes(img, img_path)

        # Crop branch preds
        crop_preds = []
        for b in boxes:
            x1,y1,x2,y2 = [int(v*c) for v,c in zip(b,(img.width,img.height,img.width,img.height))]
            if x2 <= x1 or y2 <= y1:
                continue
            sq = img.crop((x1, y1, x2, y2))
            ctx = context_tf(sq)
            m   = mask_tf(sq)
            inp = torch.cat([ctx, m], dim=0).unsqueeze(0).to(self.device)
            inp = normalize_crop(inp.squeeze(0)).unsqueeze(0)
            with torch.no_grad():
                out = self.crop_net(inp)
            crop_preds.append(F.normalize(out, dim=-1))
        cp = torch.cat(crop_preds).mean(0, keepdim=True) if crop_preds else None

        # Full branch pred
        tfull = full_tf(img).unsqueeze(0).to(self.device)
        tfull = normalize_full(tfull.squeeze(0)).unsqueeze(0)
        with torch.no_grad():
            fpred = F.normalize(self.full_net(tfull), dim=-1)

        # Ensemble & output
        final = 0.5 * fpred + (0.5 * cp if cp is not None else fpred)
        latlon = xyz_to_latlon(final).cpu().squeeze(0).tolist()
        print(latlon[0], latlon[1])

# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dino_cfg',  type=Path, default='/home/benetz/code/python/Where-Am-AI/geoLocator/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py')
    p.add_argument('--dino_w',    type=Path, default='/home/benetz/code/python/Where-Am-AI/geoLocator/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth')
    p.add_argument('--crop_ckpt', type=Path, default='/home/benetz/code/python/Where-Am-AI/geoLocator/checkpoints/smaller_crop_model.pth')
    p.add_argument('--full_ckpt', type=Path, default='/home/benetz/code/python/Where-Am-AI/geoLocator/checkpoints/full_model.pth')
    p.add_argument('--crop_arch', type=str, default='resnest50d')
    p.add_argument('--full_arch', type=str, default='convnextv2_tiny')
    p.add_argument('--image',     type=Path, required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gi = GeoInfer(
        args.dino_cfg, args.dino_w,
        args.crop_ckpt, args.full_ckpt,
        args.crop_arch, args.full_arch,
        device
    )
    gi.predict(args.image)

if __name__=='__main__':
    main()
