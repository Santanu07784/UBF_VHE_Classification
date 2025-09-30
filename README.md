# Dual-Branch ResNet Classification for UBF + VHE Digital Pathology

This repository provides the PyTorch implementation of a **dual-branch ResNet18 framework** for cancer classification using **unstained bright-field (UBF)** and **virtually H&E-stained (VHE)** histology images.  


---

---

## 🔹 Requirements
- Python 3.8+  
- PyTorch ≥ 1.11  
- timm  
- torchvision  
- scikit-learn  
- torchcam (optional, for Grad-CAM)

Install dependencies with:
```bash
pip install torch torchvision timm scikit-learn torchcam
```

---

## 🔹 Dataset Structure

Tiles must be prepared as 512×512 pixel PNG images.

```text
datasets/
│── UBF/
│    ├── train/*.png
│    └── test/*.png
│── VHE/
     ├── train/*.png
     └── test/*.png
```

Each UBF tile must have a matching VHE tile with the same filename.

Filenames should contain a numeric class ID:  
- Example: `tile_004_65.png` → class 0 (normal)  
- Example: `tile_014_47.png` → class 1 (cancer)

---

## 🔹 Training

Run training for each fold (example: fold 0):

```bash
python train.py   --dataroot ./datasets   --checkpoints_dir ./checkpoints/UBF_VHE   --fold 0   --epochs 100   --batch_size 32   --lr 1e-4   --gpus 0
```

Models are saved under `checkpoints/UBF_VHE/` as:

```text
basic_fold0_best.pth
basic_fold1_best.pth
...
basic_fold4_best.pth
```

---

## 🔹 Testing

After training all folds, run:

```bash
python test.py   --dataroot ./datasets   --checkpoints_dir ./checkpoints/UBF_VHE   --results_dir ./results/cancer_test   --im_size 512   --batch_size 32   --gpus 0
```

---

## 🔹 Outputs

The script saves predictions, colored tiles, heatmaps, and metrics into `--results_dir` (e.g., `./results/cancer_test/`).

```text
cancer_test/
│── normal/                 # UBF tiles predicted as normal (class 0)
│── cancer/                 # UBF tiles predicted as cancer (class 1)
│── colored_tiles/
│    ├── UBF/               # Colored UBF tiles (blue=normal, purple=cancer)
│    └── VHE/               # Colored VHE tiles (blue=normal, purple=cancer)
│── heatmaps/
│    ├── UBF/               # Grad-CAM overlays on UBF tiles
│    ├── VHE/               # Grad-CAM overlays on VHE tiles
│    └── Panel/             # 2×2 panels: [UBF | UBF+CAM ; VHE | VHE+CAM]
│── predictions.csv         # Path, predicted label, prob_normal, prob_cancer, true label
│── confusion_matrix.csv    # 2×2 confusion matrix
│── metrics.txt             # Accuracy, Precision, Recall, F1 + classification report
```

---



## 🔹 Notes & Tips
- Ensure UBF and VHE tiles are correctly paired and aligned (same filename), otherwise the dual-branch input will be mismatched.
- If your dataset is imbalanced, consider using class weights or focal loss during training.
- For Grad-CAM, `torchcam` is optional but convenient. If unavailable, simple saliency maps or guided backprop can be used.
- Save model checkpoints frequently and keep logs of training/validation loss to detect overfitting early.

---

## 🔹 Citation

If you use this code or adapt it for your research, please cite:

**Deep Learning based Label-Free Virtual Staining and Classification of Human Tissues using Digital Slide Scanner.**  


---



