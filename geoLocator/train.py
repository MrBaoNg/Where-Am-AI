#!/usr/bin/env python3
"""
train_pathC_amp_unit_sphere_final.py —
Full updated training script with integrated augmentations, normalization,
AdamW + warm‑up + cosine scheduler, and simplified backbone training.
"""
import io, argparse, math, pickle, random
import lmdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
import timm
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# ──────────────────────────────── Constants & Utilities ──────────────────────
RAD = math.pi / 180.0
EARTH_RADIUS_KM = 6371.0
patience = 2

def latlon_to_xyz(latlon):
    lat = latlon[:,0]*RAD; lon = latlon[:,1]*RAD
    x = torch.cos(lat)*torch.cos(lon)
    y = torch.cos(lat)*torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x,y,z],dim=-1)

def xyz_to_latlon(xyz):
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    lat = torch.asin(torch.clamp(z,-1,1))/RAD
    lon = torch.atan2(y,x)/RAD
    return torch.stack([lat,lon],dim=-1)

def geodesic_dist(l1,l2):
    lat1,lon1 = l1[:,0]*RAD,l1[:,1]*RAD
    lat2,lon2 = l2[:,0]*RAD,l2[:,1]*RAD
    dlat = lat1-lat2; dlon = lon1-lon2
    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    a = torch.clamp(a,0,1)
    return 2*EARTH_RADIUS_KM*torch.atan2(torch.sqrt(a),torch.sqrt(1-a))

# ───────────────────────────────── Datasets ─────────────────────────────────
class CropLMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform):
        self.lmdb_path = Path(lmdb_path)
        kc, tc = self.lmdb_path.with_suffix('.keys.pkl'), self.lmdb_path.with_suffix('.targets.pkl')
        if kc.exists() and tc.exists():
            self.keys = pickle.load(open(kc,'rb'))
            self.targets_latlon = pickle.load(open(tc,'rb'))
        else:
            env = lmdb.open(str(self.lmdb_path), subdir=True, readonly=True, lock=False, readahead=True, meminit=False)
            with env.begin() as txn:
                self.keys = [k for k,_ in txn.cursor()]
            self.targets_latlon=[]
            with env.begin() as txn:
                for k in self.keys:
                    rec = pickle.loads(txn.get(k))
                    self.targets_latlon.append(torch.tensor([rec['lat'],rec['lon']],dtype=torch.float32))
            pickle.dump(self.keys, open(kc,'wb')); pickle.dump(self.targets_latlon,open(tc,'wb'))
            env.close()
        self.transform=transform
        self.env=None; self.txn=None

    def __len__(self): return len(self.keys)
    def __getitem__(self,idx):
        if self.env is None:
            self.env = lmdb.open(str(self.lmdb_path), subdir=True, readonly=True,
                                 lock=False, readahead=True, meminit=False, map_async=True)
            self.txn = self.env.begin(write=False)
        rec = pickle.loads(self.txn.get(self.keys[idx]))
        ctx  = Image.open(io.BytesIO(rec['ctx'])).convert('RGB')
        mask = Image.open(io.BytesIO(rec['mask'])).convert('RGB')
        return self.transform(ctx,mask), self.targets_latlon[idx]

class FullLMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform):
        self.lmdb_path = Path(lmdb_path)
        kc, tc = self.lmdb_path.with_suffix('.keys.pkl'), self.lmdb_path.with_suffix('.targets.pkl')
        if kc.exists() and tc.exists():
            self.keys = pickle.load(open(kc,'rb'))
            self.targets_latlon = pickle.load(open(tc,'rb'))
        else:
            env = lmdb.open(str(self.lmdb_path), subdir=True, readonly=True, lock=False, readahead=True, meminit=False)
            with env.begin() as txn:
                self.keys = [k for k,_ in txn.cursor()]
            self.targets_latlon=[]
            with env.begin() as txn:
                for k in self.keys:
                    lat,lon = k.decode().split('_',1)
                    self.targets_latlon.append(torch.tensor([float(lat),float(lon)],dtype=torch.float32))
            pickle.dump(self.keys, open(kc,'wb')); pickle.dump(self.targets_latlon,open(tc,'wb'))
            env.close()
        self.transform=transform
        self.env=None; self.txn=None

    def __len__(self): return len(self.keys)
    def __getitem__(self,idx):
        if self.env is None:
            self.env = lmdb.open(str(self.lmdb_path), subdir=True, readonly=True,
                                 lock=False, readahead=True, meminit=False, map_async=True)
            self.txn = self.env.begin(write=False)
        img = Image.open(io.BytesIO(self.txn.get(self.keys[idx]))).convert('RGB')
        return self.transform(img), self.targets_latlon[idx]

# ──────────────────────────────── Transforms & Augments ───────────────────────────────
IMAGENET_MEAN=(0.485,0.456,0.406); IMAGENET_STD=(0.229,0.224,0.225)
CROP_MEAN=IMAGENET_MEAN+IMAGENET_MEAN; CROP_STD=IMAGENET_STD+IMAGENET_STD

class ContextTransform:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1) ensure smaller side >= 224
        w, h = img.size
        if w < 224 or h < 224:
            scale = 224 / min(w, h)
            img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            w, h = img.size

        # 2) center-crop 224×224
        left = (w - 224) // 2
        top  = (h - 224) // 2
        img = TF.crop(img, top, left, 224, 224)

        # 3) now that it’s 224×224, apply augment
        img = self.augment(img)

        # 4) to tensor
        return TF.to_tensor(img)

class FullImageTransform:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1) ensure both dims >=224
        w, h = img.size
        if w < 224 or h < 224:
            scale = max(224/w, 224/h)
            img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            w, h = img.size

        # 2) jittered center‑crop 224×224
        max_dx = int(0.1 * (w - 224))
        max_dy = int(0.1 * (h - 224))
        cx = (w - 224)//2 + random.randint(-max_dx, max_dx)
        cy = (h - 224)//2 + random.randint(-max_dy, max_dy)
        cx = max(0, min(cx, w - 224))
        cy = max(0, min(cy, h - 224))
        img = TF.crop(img, cy, cx, 224, 224)

        # 3) now apply augment on 224×224
        img = self.augment(img)

        # 4) to tensor
        return TF.to_tensor(img)

class MaskTransform:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        # pad to square
        w, h = img.size
        max_wh = max(w, h)
        pad_l = (max_wh - w)//2
        pad_t = (max_wh - h)//2
        pad_r = max_wh - w - pad_l
        pad_b = max_wh - h - pad_t
        img = TF.pad(img, (pad_l, pad_t, pad_r, pad_b), fill=0)

        # pad further if <224
        if max_wh < 224:
            extra = 224 - max_wh
            ep = extra // 2
            img = TF.pad(img, (ep, ep, extra-ep, extra-ep), fill=0)

        # center-crop if larger
        w2, h2 = img.size
        if w2 > 224 or h2 > 224:
            left = (w2 - 224)//2
            top  = (h2 - 224)//2
            img = TF.crop(img, top, left, 224, 224)

        # **no augment on the mask**
        return TF.to_tensor(img)

context_tf=ContextTransform(); full_tf=FullImageTransform(); mask_tf=MaskTransform()
def crop_transform(ctx,mask): return torch.cat([context_tf(ctx),mask_tf(mask)],dim=0)
normalize_crop=transforms.Normalize(CROP_MEAN,CROP_STD)
normalize_full=transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)

# ─────────────────────────────────── Training ──────────────────────────────────
def train_branch(name,model_name,train_loader,val_loader,in_chans,
                 epochs,out_path,device,resume_ckpt=None,e2e_every=5,ensemble_loader=None):
    model=timm.create_model(model_name,pretrained=True,num_classes=3,in_chans=in_chans).to(device)
    for p in model.parameters(): p.requires_grad=True
    head_params=[p for p in model.parameters() if p.requires_grad]
    back_params=[p for p in model.parameters() if not p.requires_grad]
    optimizer=optim.AdamW([{'params':head_params,'lr':1e-4,'weight_decay':1e-4}])
    def lr_lambda(e): return (e+1)/2 if e<2 else 0.02+0.98*0.5*(1+math.cos(math.pi*(e-2)/(epochs-2)))
    head_lr, back_lr = 1e-3, 1e-4
    total_steps = len(train_loader) * epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=head_lr,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1e4,
    )

    criterion=nn.MSELoss(); scaler=GradScaler(); best_km=1e9; no_imp=0

    if resume_ckpt and resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device)
        sd   = ckpt.get('model_state_dict', ckpt)

        model_dict = model.state_dict()
        matched, discarded = [], []
        for k, v in sd.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
                matched.append(k)
            else:
                discarded.append(k)

        model.load_state_dict(model_dict)
        print(f"✅ Loaded {len(matched)}/{len(sd)} parameters from {resume_ckpt.name}")
        if discarded:
            print(f"   ⚠️ Discarded {len(discarded)} parameters (shape/name mismatch)")

    for ep in range(1,epochs+1):
        # train
        model.train(); total_loss=0.0
        with tqdm(train_loader,desc=f"{name} Train E{ep}") as pbar:
            for imgs,tgt in pbar:
                batch=(normalize_crop(imgs) if in_chans==6 else normalize_full(imgs)).to(device)
                #print(f"[Train Debug] batch min/max={batch.min():.4f}/{batch.max():.4f}")
                tgt=tgt.to(device); tgt_xyz=latlon_to_xyz(tgt)
                optimizer.zero_grad()
                with autocast():
                    out=model(batch); pred=F.normalize(out,dim=-1); loss=criterion(pred,tgt_xyz)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                nn_utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer); scaler.update()
                scheduler.step()
                total_loss+=loss.item()*batch.size(0)
                with torch.no_grad():
                    km=geodesic_dist(xyz_to_latlon(pred),tgt).mean().item()
                pbar.set_postfix(mse=loss.item(),km=f"{km:.1f}km")

        # val
        model.eval(); all_d=[]; mse_sum=0.0
        with torch.no_grad():
            for imgs,tgt in tqdm(val_loader,desc=f"{name} Val E{ep}",leave=False):
                batch=(normalize_crop(imgs) if in_chans==6 else normalize_full(imgs)).to(device)
                #print(f"[Val Debug]   batch min/max={batch.min():.4f}/{batch.max():.4f}")
                tgt=tgt.to(device); tgt_xyz=latlon_to_xyz(tgt)
                with autocast(): out=model(batch)
                pred=F.normalize(out,dim=-1); mse=criterion(pred,tgt_xyz)
                d=geodesic_dist(xyz_to_latlon(pred),tgt)
                mse_sum+=mse.item()*batch.size(0); all_d.append(d.cpu().numpy())
        dists=np.concatenate(all_d); mean_km,med_km=dists.mean(),np.median(dists)
        plt.figure(); plt.plot(np.sort(dists),np.arange(len(dists))/len(dists)); plt.grid(True)
        plt.title(f"{name} CDF E{ep}"); plt.savefig(out_path.with_name(f"{name}_cdf_ep{ep}.png")); plt.close()
        print(f"{name} E{ep}: Train MSE={total_loss/len(train_loader.dataset):.4f} MeanKM={mean_km:.1f} MedKM={med_km:.1f}")

        if mean_km<best_km: best_km=mean_km; no_imp=0; torch.save({'model_state_dict':model.state_dict()},out_path)
        else: no_imp+=1
        if no_imp>=patience:
            print(f"No improvement in {patience} epochs—stopping early at epoch {ep}."); break

        if name=='full' and ensemble_loader and ep% e2e_every==0:
            validate_ensemble(name,out_path,model,ensemble_loader,val_loader,device,ep,out_path.parent)

def validate_ensemble(crop_model_name,crop_ckpt,full_model,crop_loader,full_loader,device,epoch,save_dir):
    crop_model=timm.create_model(crop_model_name,pretrained=True,num_classes=3,in_chans=6).to(device)
    sd=torch.load(crop_ckpt,map_location=device)['model_state_dict']
    crop_model.load_state_dict(sd,strict=False)
    crop_model.eval(); full_model.eval()
    cpreds=[]
    with torch.no_grad():
        for imgs,_ in tqdm(crop_loader,desc="Crop Preds",leave=False):
            batch=normalize_crop(imgs).to(device)
            print(f"[Ensemble Crop Debug] batch min/max={batch.min():.4f}/{batch.max():.4f}")
            cpreds.append(F.normalize(crop_model(batch),dim=-1).cpu())
    cpreds=torch.cat(cpreds); idx=0; errs=[]
    with torch.no_grad():
        for imgs,tgt in tqdm(full_loader,desc="E2E Val",leave=False):
            batch=normalize_full(imgs).to(device)
            print(f"[Ensemble Full Debug] batch min/max={batch.min():.4f}/{batch.max():.4f}")
            pf=F.normalize(full_model(batch),dim=-1)
            pc=cpreds[idx:idx+batch.size(0)].to(device); idx+=batch.size(0)
            final=0.5*pf+0.5*pc
            d=geodesic_dist(xyz_to_latlon(final),tgt.to(device))
            errs.extend(d.cpu().tolist())
    errs=np.array(errs)
    print(f"[E2E] Mean={errs.mean():.1f}km Med={np.median(errs):.1f}km")
    sd=np.sort(errs); cdf=np.arange(len(errs))/len(errs)
    plt.figure(); plt.plot(sd,cdf); plt.grid(True)
    plt.title(f"E2E CDF Epoch {epoch}"); plt.savefig(save_dir/f'e2e_cdf_ep{epoch}.png'); plt.close()


def sanity_check(crop_ds, full_ds,
                 crop_model, full_model,
                 device, num_samples=5,
                 output_dir: Path = None):
    """
    Perform a visual + tensor debug on both crop and full branches.
    - Opens LMDB envs if needed.
    - Samples num_samples from each dataset.
    - Prints tensor min/max to catch blank arrays.
    - Saves or shows:
        * original PILs,
        * raw tensors (split for crop),
        * transformed inputs,
        * optional model prediction vs target + km error.
    """
    # Optionally prepare output directory
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)

    # Ensure LMDB txns are open
    for ds in (crop_ds, full_ds):
        if ds is not None and ds.env is None:
            ds.env = lmdb.open(
                str(ds.lmdb_path), subdir=True, readonly=True,
                lock=False, readahead=True, meminit=False, map_async=True
            )
            ds.txn = ds.env.begin(write=False)

        def _debug_branch(ds, model, transform, in_ch, name):
            model.eval() if model else None
            print(f"\n--- {name} Sanity Check ({num_samples} samples) ---")
            for i in tqdm(range(num_samples), desc=f"{name} samples"):
                idx = random.randrange(len(ds))
                print(f"[{name}] idx={idx}")

                # LOAD RAW PIL(S) AND BUILD TENSORS
                if in_ch == 6:
                    rec = pickle.loads(ds.txn.get(ds.keys[idx]))
                    ctx  = Image.open(io.BytesIO(rec['ctx'])).convert('RGB')
                    mask = Image.open(io.BytesIO(rec['mask'])).convert('RGB')
                    raw = torch.cat([transforms.ToTensor()(ctx),
                                    transforms.ToTensor()(mask)], dim=2)  # H×W×6
                    inp = transform(ctx, mask).unsqueeze(0)               # 1×6×224×224

                    # APPLY 6‑CHANNEL NORMALIZER
                    inp = normalize_crop(inp.squeeze(0)).unsqueeze(0).to(device)

                else:
                    raw_img = Image.open(io.BytesIO(ds.txn.get(ds.keys[idx]))).convert('RGB')
                    raw = transforms.ToTensor()(raw_img)                   # 3×H×W
                    inp = transform(raw_img).unsqueeze(0)                  # 1×3×224×224

                    # APPLY 3‑CHANNEL NORMALIZER
                    inp = normalize_full(inp.squeeze(0)).unsqueeze(0).to(device)

                # Now raw.min/max and inp.min/max reflect normalization
                print(f" raw tensor min/max = {raw.min():.4f}/{raw.max():.4f}")
                print(f" inp   tensor min/max = {inp.min():.4f}/{inp.max():.4f}")

                # RUN THE MODEL
                if model is not None:
                    with torch.no_grad():
                        pred = model(inp)
                        latlon = xyz_to_latlon(F.normalize(pred, dim=-1)).cpu().squeeze(0)
                        targ   = ds.targets_latlon[idx]
                        err_km = geodesic_dist(latlon.unsqueeze(0),
                                            targ.unsqueeze(0)).item()
                    print(f" Predicted = ({latlon[0]:.2f},{latlon[1]:.2f}), "
                        f"Target = ({targ[0]:.2f},{targ[1]:.2f}), Err = {err_km:.1f} km")


    # Run crop branch
    if crop_ds and crop_model:
        _debug_branch(crop_ds, crop_model, crop_transform, in_ch=6, name="Crop")
    # Run full branch
    if full_ds and full_model:
        _debug_branch(full_ds, full_model, full_tf, in_ch=3, name="Full")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--crop_lmdb', required=True, type=Path)
    p.add_argument('--full_lmdb', required=True, type=Path)
    p.add_argument('--crop_model_name', type=str, default='resnest50d')
    p.add_argument('--full_model_name', type=str, default='convnextv2_tiny')
    p.add_argument('--epochs_crop', type=int, default=10)
    p.add_argument('--epochs_full', type=int, default=8)
    p.add_argument('--batch_crop', type=int, default=64)
    p.add_argument('--batch_full', type=int, default=64)
    p.add_argument('--e2e_every', type=int, default=5)
    p.add_argument('--out_dir', type=Path, default=Path('./checkpoints/'))
    p.add_argument('--verify', type=bool, default=False)
    args = p.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    crop_ds = CropLMDBDataset(args.crop_lmdb, transform=crop_transform)
    full_ds = FullLMDBDataset(args.full_lmdb, transform=full_tf)

    
    idxs = list(range(len(full_ds)))
    random.shuffle(idxs)
    v_n = int(0.1 * len(idxs))
    full_val, full_train = idxs[:v_n], idxs[v_n:]
    full_train_loader = DataLoader(Subset(full_ds, full_train),
                                   batch_size=args.batch_full, shuffle=True,
                                   num_workers=4, pin_memory=True)
    full_val_loader   = DataLoader(Subset(full_ds, full_val),
                                   batch_size=args.batch_full, shuffle=False,
                                   num_workers=2, pin_memory=True)

    val_set = set(tuple(t.tolist()) for i,t in enumerate(full_ds.targets_latlon) if i in full_val)
    crop_train = [i for i,t in enumerate(crop_ds.targets_latlon) if tuple(t.tolist()) not in val_set]
    crop_val   = [i for i,t in enumerate(crop_ds.targets_latlon) if tuple(t.tolist()) in val_set]
    crop_train_loader = DataLoader(Subset(crop_ds, crop_train),
                                   batch_size=args.batch_crop, shuffle=True,
                                   num_workers=4, pin_memory=True)
    crop_val_loader   = DataLoader(Subset(crop_ds, crop_val),
                                   batch_size=args.batch_crop, shuffle=False,
                                   num_workers=2, pin_memory=True)
    if not args.verify:
        train_branch('crop', args.crop_model_name,
                    crop_train_loader, crop_val_loader,
                    in_chans=6, epochs=args.epochs_crop,
                    out_path=args.out_dir/'smaller_crop_model.pth',
                    device=DEVICE, resume_ckpt=args.out_dir/'smaller_crop_model.pth')

        train_branch('full', args.full_model_name,
                    full_train_loader, full_val_loader,
                    in_chans=3, epochs=args.epochs_full,
                    out_path=args.out_dir/'full_model.pth',
                    device=DEVICE,
                    resume_ckpt=None, e2e_every=args.e2e_every,
                    ensemble_loader=crop_val_loader)
    else:
        crop_model = timm.create_model(args.crop_model_name, pretrained=False, num_classes=3, in_chans=6).to(DEVICE)
        crop_model.load_state_dict(torch.load(args.out_dir/"crop_model.pth")['model_state_dict'])

        full_model = timm.create_model(args.full_model_name, pretrained=False, num_classes=3, in_chans=3).to(DEVICE)

    # Then:
        sanity_check(crop_ds, full_ds, crop_model, full_model, DEVICE, num_samples=8, output_dir=Path("/home/benetz/code/python/geoLocator/sanity_outputs"))

if __name__ == '__main__':
    main()
