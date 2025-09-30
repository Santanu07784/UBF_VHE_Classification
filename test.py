# -*- coding: utf-8 -*-
# test.py — ensemble inference for cancer classification (UBF+VHE dual input)
#           - Saves predicted tiles by class (normal/cancer)
#           - Saves colored tiles for both UBF & VHE
#           - Saves Grad-CAM overlays with same filenames as input
#           - Prints/saves Accuracy, Precision, Recall, F1, Confusion Matrix

import os, time, csv, shutil, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image as PILImage

import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Optional: CAM imports
try:
    from torchcam.methods import GradCAM
    from torchcam.utils import overlay_mask
    from torchvision.transforms.functional import to_pil_image
    TORCHCAM_OK = True
except Exception:
    TORCHCAM_OK = False


# ------------------------
# Dataset for paired UBF+VHE
# ------------------------
class UBVVHETestDataset(Dataset):
    def __init__(self, root, transform, class_outline=12):
        self.ubf_paths = sorted(list(Path(root, "UBF", "test").glob("*.png")))
        self.vhe_paths = sorted(list(Path(root, "VHE", "test").glob("*.png")))
        assert len(self.ubf_paths) == len(self.vhe_paths), "UBF/VHE count mismatch!"
        self.transform = transform
        self.class_outline = int(class_outline)

    def __len__(self): return len(self.ubf_paths)

    def _label_from_name(self, name):
        try:
            cid = int(name.split('_')[-2])
            return 0 if cid < self.class_outline else 1
        except Exception:
            return None

    def __getitem__(self, idx):
        ubf = PILImage.open(self.ubf_paths[idx]).convert("RGB")
        vhe = PILImage.open(self.vhe_paths[idx]).convert("RGB")
        x1, x2 = self.transform(ubf), self.transform(vhe)
        x = torch.cat([x1, x2], dim=0)  # 6-channels
        label = self._label_from_name(self.ubf_paths[idx].name)
        return x, label, str(self.ubf_paths[idx]), str(self.vhe_paths[idx])


# ------------------------
# Model (dual branch ResNet18)
# ------------------------
class MyModelUBFVHE(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_ubf = timm.create_model('resnet18', pretrained=False, in_chans=3, num_classes=2)
        self.res_ubf.fc = nn.Identity()
        self.res_vhe = timm.create_model('resnet18', pretrained=False, in_chans=3, num_classes=2)
        self.res_vhe.fc = nn.Identity()
        self.fc1 = nn.Linear(512, 16)
        self.fc2 = nn.Linear(512, 16)
        self.fc_final = nn.Linear(32, 2)

    def forward(self, x):
        x1 = self.res_ubf(x[:,0:3,:,:])
        x1 = self.fc1(x1)
        x2 = self.res_vhe(x[:,3:6,:,:])
        x2 = self.fc2(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.fc_final(x)


def load_checkpoint_into_model(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    new_state = {}
    for k, v in state.items():
        k2 = k[7:] if k.startswith("module.") else k
        new_state[k2] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


# ------------------------
# Visualization Helpers
# ------------------------
def create_colored_tile(tile_size, pred_label):
    color = (0,0,255) if pred_label==0 else (128,0,128)  # blue=normal, purple=cancer
    return PILImage.new("RGB", (tile_size, tile_size), color)

def _unnormalize_to_pil(t):
    t = t.detach().cpu()
    t = t * 0.5 + 0.5
    t = torch.clamp(t, 0, 1)
    return to_pil_image(t)

def save_gradcam_overlay(models, img, pred_class, fname_ubf, fname_vhe, out_dir, device):
    if not TORCHCAM_OK: return
    out_dir = Path(out_dir)
    ubf_dir   = out_dir / "UBF"
    vhe_dir   = out_dir / "VHE"
    panel_dir = out_dir / "Panel"
    for d in [ubf_dir, vhe_dir, panel_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Split UBF/VHE
    ubf_img = img[0:3]; vhe_img = img[3:6]
    ubf_base = _unnormalize_to_pil(ubf_img)
    vhe_base = _unnormalize_to_pil(vhe_img)

    def _cam_for_branch(model, x, branch):
        with torch.enable_grad():
            inp = x.unsqueeze(0).clone().to(device).requires_grad_(True)
            layer = "res_ubf.layer4" if branch=="UBF" else "res_vhe.layer4"
            cam = GradCAM(model, target_layer=layer)
            scores = model(inp)
            act_maps = cam(pred_class, scores)
            cam_map = act_maps[0].detach().cpu()
            if cam_map.ndim==3: cam_map=cam_map.squeeze(0)
            for h in cam.hook_handles: h.remove()
        return cam_map

    # Average CAMs
    ubf_cams, vhe_cams = [], []
    for m in models:
        ubf_cams.append(_cam_for_branch(m, img, "UBF"))
        vhe_cams.append(_cam_for_branch(m, img, "VHE"))
    ubf_cam = torch.stack(ubf_cams).mean(0)
    vhe_cam = torch.stack(vhe_cams).mean(0)
    for cam in [ubf_cam, vhe_cam]:
        cam -= cam.min(); cam /= (cam.max()+1e-8)

    ubf_overlay = overlay_mask(ubf_base, to_pil_image(ubf_cam, mode="F"), alpha=0.5)
    vhe_overlay = overlay_mask(vhe_base, to_pil_image(vhe_cam, mode="F"), alpha=0.5)

    # Save with same input filenames
    ubf_overlay.save(ubf_dir / Path(fname_ubf).name)
    vhe_overlay.save(vhe_dir / Path(fname_vhe).name)

    # 2×2 panel
    w,h = ubf_base.size
    panel = PILImage.new("RGB", (2*w, 2*h), (255,255,255))
    panel.paste(ubf_base, (0,0))
    panel.paste(ubf_overlay, (w,0))
    panel.paste(vhe_base, (0,h))
    panel.paste(vhe_overlay, (w,h))
    panel.save(panel_dir / f"panel_{Path(fname_ubf).name}")


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=r"\datasets")
    ap.add_argument("--checkpoints_dir", default=r"\UBF_VHE")
    ap.add_argument("--results_dir", default=r"\results\cancer_test")
    ap.add_argument("--im_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--class_outline", type=int, default=12)
    ap.add_argument("--gpus", default="0")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 20
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    out_root = Path(args.results_dir)
    out_norm, out_cancer = out_root/"normal", out_root/"cancer"
    out_colored_ubf = out_root/"colored_tiles"/"UBF"
    out_colored_vhe = out_root/"colored_tiles"/"VHE"
    out_hm = out_root/"heatmaps"
    for d in [out_norm, out_cancer, out_colored_ubf, out_colored_vhe, out_hm]:
        d.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize((args.im_size, args.im_size)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    ds = UBVVHETestDataset(args.dataroot, transform, class_outline=args.class_outline)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=max(0, os.cpu_count()//4), pin_memory=True)
    print(f"Loaded {len(ds)} paired images (UBF+VHE)")

    # Load ensemble models
    models = []
    for fold in range(5):
        ckpt = Path(args.checkpoints_dir) / f"basic_fold{fold}_best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        m = MyModelUBFVHE().to(device)
        load_checkpoint_into_model(m, str(ckpt), device)
        models.append(m)
        print(f"Loaded {ckpt}")

    all_rows, y_true, y_pred = [], [], []
    n0 = n1 = 0
    t0 = time.time()

    for imgs, labels, fnames_ubf, fnames_vhe in dl:
        imgs = imgs.to(device)
        with torch.no_grad():
            probs_stack = [torch.softmax(m(imgs), dim=1) for m in models]
            avg_probs = torch.stack(probs_stack, dim=0).mean(0)
            preds = avg_probs.argmax(1).cpu().numpy()

        for i in range(len(fnames_ubf)):
            pred = int(preds[i])
            dst_dir = out_norm if pred==0 else out_cancer
            shutil.copy2(fnames_ubf[i], dst_dir/Path(fnames_ubf[i]).name)

            # Save colored tiles for both UBF and VHE
            tile_ubf = create_colored_tile(args.im_size, pred)
            tile_vhe = create_colored_tile(args.im_size, pred)
            tile_ubf.save(out_colored_ubf / Path(fnames_ubf[i]).name)
            tile_vhe.save(out_colored_vhe / Path(fnames_vhe[i]).name)

            prob0, prob1 = float(avg_probs[i,0].cpu()), float(avg_probs[i,1].cpu())
            label = "" if labels[i] is None else int(labels[i])
            all_rows.append([fnames_ubf[i], pred, prob0, prob1, label])

            if labels[i] is not None:
                y_true.append(int(labels[i])); y_pred.append(pred)

            n0 += (pred==0); n1 += (pred==1)

            if TORCHCAM_OK:
                save_gradcam_overlay(models, imgs[i].cpu(), pred, fnames_ubf[i], fnames_vhe[i], out_hm, device)

    # Save CSV
    csv_path = out_root/"predictions.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["path","pred","prob_normal","prob_cancer","label"])
        w.writerows(all_rows)

    print(f"\nSaved {n0} normal → {out_norm}")
    print(f"Saved {n1} cancer → {out_cancer}")
    print(f"Colored tiles → {out_root/'colored_tiles'}")
    print(f"Predictions CSV → {csv_path}")
    if TORCHCAM_OK: print(f"Grad-CAMs → {out_hm}")

    if len(y_true)>0:
        acc=accuracy_score(y_true,y_pred); prec=precision_score(y_true,y_pred,zero_division=0)
        rec=recall_score(y_true,y_pred,zero_division=0); f1=f1_score(y_true,y_pred,zero_division=0)
        cm=confusion_matrix(y_true,y_pred,labels=[0,1])
        rep=classification_report(y_true,y_pred,target_names=["normal(0)","cancer(1)"],digits=3,zero_division=0)

        print("\n=== Metrics ===")
        print(f"Accuracy : {acc*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall   : {rec*100:.2f}%")
        print(f"F1 Score : {f1*100:.2f}%")
        print("Confusion matrix:\n",cm)

        np.savetxt(out_root/"confusion_matrix.csv",cm,fmt="%d",delimiter=",")
        with open(out_root/"metrics.txt","w") as f:
            f.write(f"Accuracy : {acc*100:.2f}%\nPrecision: {prec*100:.2f}%\nRecall   : {rec*100:.2f}%\nF1 Score : {f1*100:.2f}%\n\n")
            f.write(rep)

    print(f"Done in {time.time()-t0:.2f}s")


if __name__=="__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
