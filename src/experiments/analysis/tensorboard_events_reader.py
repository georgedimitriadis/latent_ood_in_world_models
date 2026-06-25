"""
Generation-based evaluation for the OCL on the OOD test splits.

Two metrics:
  - pixel MSE between the generated image and the ground-truth target
    (continuous, comparable to the training mse term)
  - EXACT-MATCH accuracy: fraction of generated images that are FULLY correct,
    i.e. every pixel matches the target after both are snapped to the nearest
    palette colour. This is the metric that says "did the model apply the rule
    correctly", which raw MSE cannot.
  - (bonus) per-pixel accuracy: mean fraction of correct pixels, a softer signal
    that degrades gracefully and is useful for watching partial progress.

Why palette snapping: model.generate returns continuous RGB in [0,1] from the
dVAE decoder, but ground truth is a small fixed set of palette colours. We map
every pixel of both images to the nearest palette entry, then compare indices.
"""

import torch


def build_palette_tensor(palette_rgb_0_255, device):
    """
    palette_rgb_0_255: list/array of (R,G,B) on 0..255 for the colours your data
    uses (the SAME values baked into convert_to_ocl.py's PALETTE, in any order).
    Returns a (K, 3) float tensor on 0..1, used as snapping targets.
    """
    pal = torch.tensor(palette_rgb_0_255, dtype=torch.float32, device=device) / 255.0
    return pal  # (K, 3)


def snap_to_palette(img, palette):
    """
    img:     (B, C, H, W) in [0,1], C=3
    palette: (K, 3) in [0,1]
    Returns: (B, H, W) long tensor of palette indices (nearest colour per pixel).
    """
    B, C, H, W = img.shape
    # (B, H, W, C)
    px = img.permute(0, 2, 3, 1).reshape(-1, C)            # (B*H*W, 3)
    # squared distance to each palette colour: (B*H*W, K)
    d = torch.cdist(px.unsqueeze(0), palette.unsqueeze(0)).squeeze(0)
    idx = d.argmin(dim=-1)                                  # (B*H*W,)
    return idx.reshape(B, H, W)


@torch.no_grad()
def evaluate_generation(model, test_loader, palette, device):
    """
    Runs model.generate over a test split and returns a dict of metrics.
    Assumes each batch is (B, N+1, 2, C, H, W) like the train/val tensors.
    """
    model.eval()
    total_imgs = 0
    sum_mse = 0.0
    sum_exact = 0           # count of fully-correct images
    sum_pixacc = 0.0        # summed per-image pixel accuracy

    for images in test_loader:
        images = images.to(device, non_blocking=True)
        # support = all pairs except the last; query input = A of the last pair
        support = images[:, :-1]                 # (B, N, 2, C, H, W)
        query   = images[:, -1, 0]               # (B, C, H, W)
        target  = images[:, -1, 1]               # (B, C, H, W) ground-truth D

        gen = model.generate(support, query)     # (B, C, H, W) in [0,1]

        B = gen.shape[0]
        total_imgs += B

        # continuous pixel MSE (sum over pixels, mean over batch — matches the
        # training mse convention of /B). Adjust if you prefer mean over pixels.
        sum_mse += ((gen - target) ** 2).sum().item() / 1.0

        # palette-snapped exact match
        gen_idx    = snap_to_palette(gen, palette)      # (B, H, W)
        target_idx = snap_to_palette(target, palette)   # (B, H, W)

        correct_pixels = (gen_idx == target_idx)        # (B, H, W) bool
        # an image is fully correct iff ALL its pixels match
        fully_correct = correct_pixels.reshape(B, -1).all(dim=1)  # (B,)
        sum_exact += fully_correct.sum().item()

        # per-image pixel accuracy, then summed
        pixacc = correct_pixels.reshape(B, -1).float().mean(dim=1)  # (B,)
        sum_pixacc += pixacc.sum().item()

    return {
        'mse': sum_mse / max(total_imgs, 1),
        'exact_match': sum_exact / max(total_imgs, 1),   # in [0,1]
        'pixel_acc': sum_pixacc / max(total_imgs, 1),    # in [0,1]
        'n': total_imgs,
    }