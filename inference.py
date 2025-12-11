#!/usr/bin/env python3
"""
inference_folder.py
Usage example:
python inference_folder.py --weights checkpoints/best_fold_1.pth --input_dir test_images --out_dir results --size 1024 --threshold 0.5 --save_prob
"""
import os
import cv2
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import torch

# try to import albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAVE_ALB = True
except Exception:
    HAVE_ALB = False

# import your UNet - adjust path if necessary
from unet import UNet

# -------------------------
# Helpers
# -------------------------
def build_transform(size):
    H, W = size, size
    if HAVE_ALB:
        return A.Compose([
            A.Resize(H, W),
            A.Normalize(),   # mean/std default (0,1) in albumentations Normalize -> but ToTensorV2 expects float
            ToTensorV2()
        ])
    else:
        # fallback: we'll do resize + scale to [0,1] and normalize manually later
        return None

def preprocess_image(img_bgr, transform, size):
    """
    img_bgr: original loaded by cv2 (BGR)
    Returns tensor cuda/ cpu later in batch: np.ndarray or torch.Tensor
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb.shape[:2]

    if transform is not None:
        aug = transform(image=img_rgb)
        img_tensor = aug["image"]  # if ToTensorV2 => torch.Tensor C,H,W dtype=float32
        return img_tensor, (H0, W0)
    else:
        # fallback: resize + to tensor + normalize by ImageNet stats
        target = (size, size)
        img_resized = cv2.resize(img_rgb, (target[1], target[0]), interpolation=cv2.INTER_LINEAR)
        img_resized = img_resized.astype("float32") / 255.0
        # normalize with ImageNet stats
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        img_resized = (img_resized - mean) / std
        # HWC -> CHW
        img_chw = np.transpose(img_resized, (2,0,1)).astype("float32")
        img_tensor = torch.from_numpy(img_chw)
        return img_tensor, (H0, W0)

def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    return model

def save_mask(mask_binary, out_path):
    # mask_binary: uint8 0/255 HxW
    cv2.imwrite(out_path, mask_binary)

def save_probmap(prob_map, out_path):
    # prob_map float HxW [0,1] -> save as 16-bit or PNG gray scaled
    norm = (prob_map * 255.0).clip(0,255).astype("uint8")
    cv2.imwrite(out_path, norm)

def overlay_mask_on_image(orig_bgr, mask_bin, alpha=0.4, color=(0,0,255)):
    """
    orig_bgr: original BGR image (HxWx3)
    mask_bin: 0/255 uint8 HxW
    color: BGR tuple for mask color
    """
    overlay = orig_bgr.copy()
    mask_bool = (mask_bin > 0)
    overlay[mask_bool] = (overlay[mask_bool] * (1-alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay

# -------------------------
# Main inference over folder
# -------------------------
def run_inference(weights, input_dir, out_dir, size=1024, threshold=0.5, device="cpu",
                  save_prob=False, save_overlay=True, overlay_alpha=0.4, exts=("png","jpg","jpeg")):

    os.makedirs(out_dir, exist_ok=True)
    mask_dir = os.path.join(out_dir, "masks")
    prob_dir = os.path.join(out_dir, "probs")
    overlay_dir = os.path.join(out_dir, "overlay")
    os.makedirs(mask_dir, exist_ok=True)
    if save_prob: os.makedirs(prob_dir, exist_ok=True)
    if save_overlay: os.makedirs(overlay_dir, exist_ok=True)

    # collect images
    files = []
    for e in exts:
        files.extend(sorted(glob(os.path.join(input_dir, f"*.{e}"))))
    if len(files) == 0:
        raise ValueError(f"No images found in {input_dir} with extensions {exts}")

    # build transform
    transform = build_transform(size)

    # load model
    model = UNet(in_channels=3, out_channels=1)
    model = load_checkpoint(model, weights, device)
    model.to(device)
    model.eval()

    # iterate
    for img_path in tqdm(files, desc="Inference"):
        fname = os.path.basename(img_path)
        name, _ = os.path.splitext(fname)
        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            print("Warning: failed to read", img_path); continue

        img_tensor, (H0, W0) = preprocess_image(orig_bgr, transform, size)
        # ensure batch and device
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device).float()

        with torch.no_grad():
            logits = model(img_tensor)               # [1,1,Hs,Ws]
            probs = torch.sigmoid(logits)[0,0].cpu().numpy()

        # resize back to orig size (W,H) note cv2 expects (width,height)
        prob_resized = cv2.resize(probs, (W0, H0), interpolation=cv2.INTER_LINEAR)

        mask_bin = (prob_resized > threshold).astype("uint8") * 255

        # save mask
        mask_out = os.path.join(mask_dir, f"{name}_mask.jpeg")
        save_mask(mask_bin, mask_out)

        # save prob map if requested
        if save_prob:
            prob_out = os.path.join(prob_dir, f"{name}_prob.jpeg")
            save_probmap(prob_resized, prob_out)

        # overlay
        if save_overlay:
            overlay = overlay_mask_on_image(orig_bgr, mask_bin, alpha=overlay_alpha, color=(0,0,255))
            cv2.imwrite(os.path.join(overlay_dir, f"{name}_overlay.jpeg"), overlay)

    print("Done. Results saved to:", out_dir)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to model .pth")
    parser.add_argument("--input_dir", required=True, help="folder with input images")
    parser.add_argument("--out_dir", default="inference_results", help="folder to save masks/probs/overlay")
    parser.add_argument("--size", type=int, default=1024, help="resize size used during preprocessing (square)")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for binary mask")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_prob", action="store_true", help="save probability maps")
    parser.add_argument("--no_overlay", action="store_true", help="do not save overlay images")
    parser.add_argument("--overlay_alpha", type=float, default=0.4, help="alpha for overlay")
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        size=args.size,
        threshold=args.threshold,
        device=args.device,
        save_prob=args.save_prob,
        save_overlay=not args.no_overlay,
        overlay_alpha=args.overlay_alpha
    )
