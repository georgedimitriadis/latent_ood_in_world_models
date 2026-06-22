"""
Convert ARC-style task datasets (.npz produced by generate_datasets_main.py)
into the .h5 format consumed by the OCL dataloaders (create_dataset.py).

Source .npz layout
-------------------
samples   : (num_tasks, 32, 32, 20) int8
            The 20 panes (last axis) are interleaved input/output:
            [in_0, out_0, in_1, out_1, ..., in_9, out_9].
            Tasks with fewer than 10 pairs are zero-padded in the middle,
            with the final real pair always sitting at panes 18/19.
languages : (num_tasks,) object   (task descriptions; carried through optionally)

Target .h5 layout (per phase key: 'train', 'val')
--------------------------------------------------
dataset[phase] : (num_tasks, N_pairs, 2, 32, 32, 3) float/uint
                 axis 1 = analogy example index
                 axis 2 = (input A, output B) pair
                 axis 5 = 3 channels (single channel replicated, colours preserved)

The OCL loader permutes the trailing (H, W, C) to (C, H, W) and divides by 255,
so we store colours on a 0..255 scale across 3 identical channels.
"""

import argparse
import numpy as np
import h5py


def extract_pairs(task_panes):
    """
    task_panes: (32, 32, 20), interleaved [in_0, out_0, in_1, out_1, ...].
    Returns a list of (input, output) arrays, dropping zero-padded pairs.
    A pair is considered padding only if BOTH panes are all-zero.
    """
    pairs = []
    for k in range(0, 20, 2):
        inp = task_panes[:, :, k]
        out = task_panes[:, :, k + 1]
        if inp.sum() == 0 and out.sum() == 0:
            continue
        pairs.append((inp, out))
    return pairs


# Code -> RGB (0..255), taken from the project's Colour.map_int_to_colour
# (alpha dropped; the OCL works in RGB), with ONE deliberate override:
#   code 0 (Transparent in the enum) is the WHITE SURROUND around the canvas
#   and must render white (255,255,255). Code 1 (Black) is the canvas interior.
# The canvas may change size but the overall image (surround included) does not.
# The converter prints every code it actually finds so you can verify the mapping.
_COLOUR_MAP_RGBA = {
    0:   [1, 1, 1, 1],                      # Transparent -> WHITE SURROUND (override)
    1:   [0, 0, 0, 1],                      # Black canvas
    2:   [0, 116 / 255, 217 / 255, 1],      # Blue   #0074D9
    3:   [1, 65 / 255, 54 / 255, 1],        # Red    #FF4136
    4:   [46 / 255, 204 / 255, 64 / 255, 1],# Green  #2ECC40
    5:   [1, 220 / 255, 0, 1],              # Yellow #FFDC00
    6:   [170 / 255, 170 / 255, 170 / 255, 1],  # Gray   #AAAAAA
    7:   [240 / 255, 18 / 255, 190 / 255, 1],   # Purple #F012BE
    8:   [1, 133 / 255, 27 / 255, 1],       # Orange #FF851B
    9:   [127 / 255, 219 / 255, 1, 1],      # Azure  #7FDBFF
    10:  [135 / 255, 12 / 255, 37 / 255, 1],# Burgundy #870C25
    255: [1, 1, 1, 1],                      # White (holes)
}
PALETTE = {code: tuple(round(c * 255) for c in rgba[:3])
           for code, rgba in _COLOUR_MAP_RGBA.items()}
FALLBACK_RGB = (128, 128, 128)  # any code outside the map shows up as grey, not lost


def to_three_channels_255(arr2d, palette):
    """
    arr2d: (32, 32) integer colour codes.
    Map each code to its RGB triple, producing (32, 32, 3) on a 0..255 scale.
    The OCL loader divides by 255, so these land in [0, 1] with distinct hues.
    """
    h, w = arr2d.shape
    out = np.empty((h, w, 3), dtype=np.float32)
    codes = np.rint(arr2d).astype(int)  # canvases may carry tiny colour noise
    for code in np.unique(codes):
        rgb = palette.get(int(code), FALLBACK_RGB)
        out[codes == code] = rgb
    return out


def build_phase_array(samples, n_pairs, palette, codes_seen):
    """
    samples: (num_tasks, 32, 32, 20)
    Returns (num_tasks_kept, n_pairs, 2, 32, 32, 3) float32, and the kept indices.
    Tasks with fewer than n_pairs real pairs are skipped; tasks with more are
    truncated to the first n_pairs. `codes_seen` is a set updated in place with
    every integer colour code encountered, for reporting.
    """
    out_tasks = []
    kept = []
    for t in range(samples.shape[0]):
        pairs = extract_pairs(samples[t])
        if len(pairs) < n_pairs:
            continue
        pairs = pairs[:n_pairs]
        ex = np.zeros((n_pairs, 2, 32, 32, 3), dtype=np.float32)
        for i, (inp, out) in enumerate(pairs):
            codes_seen.update(np.unique(np.rint(inp).astype(int)).tolist())
            codes_seen.update(np.unique(np.rint(out).astype(int)).tolist())
            ex[i, 0] = to_three_channels_255(inp, palette)
            ex[i, 1] = to_three_channels_255(out, palette)
        out_tasks.append(ex)
        kept.append(t)
    if not out_tasks:
        raise ValueError(
            f"No task had at least n_pairs={n_pairs} real pairs. "
            f"Lower --n_pairs."
        )
    return np.stack(out_tasks, axis=0), kept


def convert(npz_path, h5_path, n_pairs, val_frac, palette, seed):
    data = np.load(npz_path, allow_pickle=True)
    samples = data["samples"]
    assert samples.ndim == 4 and samples.shape[1:3] == (32, 32) and samples.shape[3] == 20, \
        f"Unexpected samples shape {samples.shape}; expected (num_tasks, 32, 32, 20)."

    codes_seen = set()
    arr, kept = build_phase_array(samples, n_pairs, palette, codes_seen)

    mapped = sorted(c for c in codes_seen if c in palette)
    unmapped = sorted(c for c in codes_seen if c not in palette)
    print(f"[{npz_path}] colour codes present: {sorted(codes_seen)}")
    print(f"            mapped -> {mapped}; "
          f"unmapped (shown grey) -> {unmapped if unmapped else 'none'}")
    if unmapped:
        print(f"            WARNING: codes {unmapped} are not in PALETTE; "
              f"edit PALETTE if these are real colours.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(arr.shape[0])
    arr = arr[perm]
    n_val = max(1, int(round(arr.shape[0] * val_frac)))
    val = arr[:n_val]
    train = arr[n_val:]

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("train", data=train, compression="gzip")
        f.create_dataset("val", data=val, compression="gzip")
        f.attrs["n_pairs"] = n_pairs
        f.attrs["channels"] = 3
        f.attrs["image_size"] = 32
        f.attrs["source_npz"] = npz_path

    print(f"            kept {arr.shape[0]} tasks "
          f"(train={train.shape[0]}, val={val.shape[0]}), "
          f"each (N_pairs={n_pairs}, 2, 32, 32, 3) -> {h5_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--translate_npz", required=True, help="Path to the translate .npz")
    p.add_argument("--rotate_npz", required=True, help="Path to the rotate .npz")
    p.add_argument("--translate_h5", default="translate_ocl.h5")
    p.add_argument("--rotate_h5", default="rotate_ocl.h5")
    p.add_argument("--n_pairs", type=int, default=3,
                   help="Fixed number of analogy pairs per task (>=2; query is the last).")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    assert args.n_pairs >= 2, "Need at least 2 pairs: support + query."

    convert(args.translate_npz, args.translate_h5, args.n_pairs,
            args.val_frac, PALETTE, args.seed)
    convert(args.rotate_npz, args.rotate_h5, args.n_pairs,
            args.val_frac, PALETTE, args.seed)


if __name__ == "__main__":
    main()
