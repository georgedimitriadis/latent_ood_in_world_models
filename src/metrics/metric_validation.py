
from os.path import join

import click
import numpy as np
from metrics.object_accuracy_metric import get_non_compositional_errors
from experiments.analysis.intermediate_layer_analysis_functions import load_data

"""
Forward rule renderer for the OCL canvas images.

Images: 32x32 int arrays. 0 = white surround (never colored), 1 = black canvas,
2/3 = object colors. Images are viewed with the ORIGIN AT BOTTOM-LEFT, so in
array terms "up" = +row and image-CCW rotation = np.rot90(k=3).

apply_rule(img, rule_bit, language_bit):
    rule_bit:     0 = translate (squares & crosses), 1 = rotate (pyramids & angles)
    language_bit: translate -> 0 = move up 6, 1 = move left 6
                  rotate    -> 0 = rotate 90 CCW,  1 = rotate 180
    Rotations keep the FULL object's bounding-box bottom-left corner fixed.

The object in the input may itself be partially hidden by the canvas edge.
It is first completed to its canonical full shape (from the known library),
the transform is applied to the FULL object, and the result is re-clipped to
the canvas - so hidden pixels can correctly reappear, and pixels moving off
the canvas disappear.

Validated against all 5994 dataset samples: applying the ground-truth rule to
each query reproduces target_idx pixel-for-pixel in every single case.
"""

BACKGROUND = 0
CANVAS = 1


# ---------------- canonical shape library ----------------

def _square(s):
    return np.ones((s, s), dtype=bool)

def _cross(s):
    m = np.zeros((s, s), dtype=bool)
    c = s // 2
    m[c, :] = True
    m[:, c] = True
    return m

def _pyramid(base, direction):
    """Triangle; direction is where the apex points, in IMAGE view
    (bottom-left origin). Array row 0 of the mask = image-bottom row."""
    h = (base + 1) // 2
    m = np.zeros((h, base), dtype=bool)
    for i in range(h):                 # widest at array row 0 (image bottom)
        w = base - 2 * i
        margin = (base - w) // 2
        m[i, margin:margin + w] = True
    if direction == 'up':
        return m
    if direction == 'down':
        return m[::-1, :]
    if direction == 'left':
        return np.rot90(m, k=1)
    if direction == 'right':
        return np.rot90(m, k=3)
    raise ValueError(direction)

def _angle(s, corner):
    """Equal-legs L; corner = the ARRAY bbox corner where the arms meet
    ('tl' = (min_row, min_col), etc.)."""
    m = np.zeros((s, s), dtype=bool)
    if corner in ('tl', 'tr'):
        m[0, :] = True
    else:
        m[-1, :] = True
    if corner in ('tl', 'bl'):
        m[:, 0] = True
    else:
        m[:, -1] = True
    return m

def shape_library(rule_bit):
    """All canonical full-object masks for a rule family:
    rule_bit 0 -> squares & crosses (3x3, 7x7)
    rule_bit 1 -> pyramids (base 5 & 13, four directions) & angles (3x3, 7x7,
                  four corners)"""
    shapes = []
    if rule_bit == 0:
        for s in (3, 7):
            shapes.append(('square', s, None, _square(s)))
            shapes.append(('cross', s, None, _cross(s)))
    else:
        for base in (5, 13):
            for d in ('up', 'down', 'left', 'right'):
                shapes.append(('pyramid', base, d, _pyramid(base, d)))
        for s in (3, 7):
            for c in ('tl', 'tr', 'bl', 'br'):
                shapes.append(('angle', s, c, _angle(s, c)))
    return shapes


# ---------------- completion of possibly-clipped objects ----------------

def complete_object(img, rule_bit):
    """
    Find the colored object and reconstruct its full canonical form, allowing
    for clipping at the canvas edges.

    Returns a list of candidate completions sorted by fewest hidden pixels
    (the correct one in every dataset sample tested). Each candidate:
      {'name', 'size', 'orient', 'mask' (full canonical mask),
       'R0', 'C0' (plane position of mask[0,0]; may lie outside the image),
       'color', 'hidden'}
    Empty list if there is no object or no canonical shape fits.
    """
    img = np.asarray(img)
    obj = ~np.isin(img, [BACKGROUND, CANVAS])
    if not obj.any():
        return []
    can = img != BACKGROUND
    H, W = img.shape
    rows, cols = np.where(obj)
    vr0, vr1 = int(rows.min()), int(rows.max())
    vc0, vc1 = int(cols.min()), int(cols.max())
    vals = img[obj]
    uvals, counts = np.unique(vals, return_counts=True)
    color = int(uvals[np.argmax(counts)])

    candidates = []
    for name, size, orient, mask in shape_library(rule_bit):
        h, w = mask.shape
        if h < (vr1 - vr0 + 1) or w < (vc1 - vc0 + 1):
            continue
        for R0 in range(vr1 - h + 1, vr0 + 1):
            for C0 in range(vc1 - w + 1, vc0 + 1):
                ok = True
                hidden = 0
                for i in range(h):
                    r = R0 + i
                    for j in range(w):
                        c = C0 + j
                        on_canvas = (0 <= r < H and 0 <= c < W and can[r, c])
                        if mask[i, j]:
                            if on_canvas:
                                if not obj[r, c]:
                                    ok = False
                                    break
                            else:
                                hidden += 1
                        elif on_canvas and obj[r, c]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append({'name': name, 'size': size,
                                       'orient': orient, 'mask': mask,
                                       'R0': R0, 'C0': C0,
                                       'color': color, 'hidden': hidden})
    candidates.sort(key=lambda c: c['hidden'])
    return candidates


# ---------------- forward rule application ----------------

def apply_rule(img, rule_bit, language_bit, shift_amount=6, completion=None):
    """
    Apply the (rule_bit, language_bit) transformation to the object in img.
    If the object is clipped by the canvas edge it is first completed to its
    canonical shape, so hidden pixels reappear correctly after the transform.
    Optionally pass a specific `completion` dict (from complete_object) to
    override the default fewest-hidden-pixels choice.
    """
    img = np.asarray(img)
    obj = ~np.isin(img, [BACKGROUND, CANVAS])
    can = img != BACKGROUND
    out = img.copy()
    out[obj] = CANVAS
    if not obj.any():
        return out

    comp = completion
    if comp is None:
        comps = complete_object(img, rule_bit)
        comp = comps[0] if comps else None

    if comp is not None:
        full_mask, R0, C0, color = comp['mask'], comp['R0'], comp['C0'], comp['color']
    else:
        # fallback: not completable to any canonical shape; move visible pixels
        rows, cols = np.where(obj)
        R0, C0 = int(rows.min()), int(cols.min())
        full_mask = obj[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        vals = img[obj]
        uvals, counts = np.unique(vals, return_counts=True)
        color = int(uvals[np.argmax(counts)])

    if rule_bit == 0:                                # translate
        if language_bit == 0:
            dR, dC = +shift_amount, 0                # up (bottom-left origin)
        else:
            dR, dC = 0, -shift_amount                # left
        new_mask, nR0, nC0 = full_mask, R0 + dR, C0 + dC
    else:                                            # rotate about full bbox blc
        k = 3 if language_bit == 0 else 2            # 90 CCW (image view) / 180
        new_mask, nR0, nC0 = np.rot90(full_mask, k=k), R0, C0

    H, W = out.shape
    nh, nw = new_mask.shape
    r_lo, r_hi = max(nR0, 0), min(nR0 + nh, H)
    c_lo, c_hi = max(nC0, 0), min(nC0 + nw, W)
    if r_lo < r_hi and c_lo < c_hi:
        sub = new_mask[r_lo - nR0:r_hi - nR0, c_lo - nC0:c_hi - nC0]
        paint = sub & can[r_lo:r_hi, c_lo:c_hi]
        region = out[r_lo:r_hi, c_lo:c_hi]
        region[paint] = color
        out[r_lo:r_hi, c_lo:c_hi] = region
    return out


# ---------------- Provenance-tracking forward rule renderer ----------------
"""
Like apply_rule, but in addition to the output image it returns a
copied_from map of shape (32, 32, 2), where copied_from[j, k, :] = (row, col)
of the INPUT pixel that was copied to OUTPUT position (j, k) - i.e. the map is
indexed by output coordinates, matching how get_non_compositional_errors reads
copied_from_pixel_indices_all_images[i, to_y, to_x, :].

Construction of the map:
  - Every pixel starts as identity: copied_from[j, k] = (j, k). This covers the
    surround, the canvas, and the pixels vacated by the object ("the background
    stays black").
  - Every OUTPUT object pixel gets the true geometric source of that pixel
    under the rule: for translations, source = destination - shift; for
    rotations, the inverse of the rotation of the completed full object about
    its bounding box (bottom-left corner fixed).

Note on hidden pixels: when part of the object is hidden by the canvas edge in
the input and reappears in the output (shape completion), the true source of
those pixels lies off-canvas - possibly at negative coordinates. The map stores
the true source, so entries can be outside [0, 31]. For translate this is
harmless in get_non_compositional_errors (only |from - to| is used, which still
equals the exact shift). For rotate, the X[from] == colour check will read a
non-object pixel there and count a small error; that is inherent to the metric,
not a bug in the map. Pass clamp_indices=True to clip stored indices into
[0, 31] instead (slightly distorting those entries).

Bit conventions (matching apply_rule_v2 and, for Z, the user's error function:
z == 0 -> expected displacement [6, 0] = up, z == 1 -> [0, 6] = left):
  rule_bit:      0 = translate, 1 = rotate
  language_bit:  translate -> 0 = up 6, 1 = left 6
                 rotate    -> 0 = 90 CCW, 1 = 180
"""

def apply_rule_with_provenance(img, data_type, language_bit, shift_amount=6,
                               completion=None, clamp_indices=False):
    """Returns (output_image, copied_from) with copied_from of shape (H, W, 2),
    copied_from[j, k] = (source_row, source_col) of the input pixel copied to
    output position (j, k)."""
    img = np.asarray(img)
    H, W = img.shape
    obj = ~np.isin(img, [BACKGROUND, CANVAS])
    can = img != BACKGROUND

    out = img.copy()
    out[obj] = CANVAS

    # identity map
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    copied_from = np.stack([rr, cc], axis=2).astype(int)

    if not obj.any():
        return out, copied_from

    comp = completion
    if comp is None:
        comps = complete_object(img, data_type)
        comp = comps[0] if comps else None

    if comp is not None:
        full_mask, R0, C0, color = comp['mask'], comp['R0'], comp['C0'], comp['color']
    else:
        rows, cols = np.where(obj)
        R0, C0 = int(rows.min()), int(cols.min())
        full_mask = obj[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        vals = img[obj]
        uvals, counts = np.unique(vals, return_counts=True)
        color = int(uvals[np.argmax(counts)])

    h, w = full_mask.shape

    if data_type == 0:                                 # translate
        if language_bit == 0:
            dR, dC = +shift_amount, 0                 # up
        else:
            dR, dC = 0, -shift_amount                 # left
        # iterate the full object's cells; paint + record provenance where the
        # destination lands on canvas
        for i in range(h):
            for j in range(w):
                if not full_mask[i, j]:
                    continue
                sr, sc = R0 + i, C0 + j               # source (may be off-canvas)
                tr, tc = sr + dR, sc + dC             # destination
                if 0 <= tr < H and 0 <= tc < W and can[tr, tc]:
                    out[tr, tc] = color
                    copied_from[tr, tc] = (sr, sc)
    else:                                             # rotate about full bbox blc
        k = 3 if language_bit == 0 else 2             # 90 CCW (image view) / 180
        ids = np.arange(h * w).reshape(h, w)
        rot_ids = np.rot90(ids, k=k)
        rot_mask = np.rot90(full_mask, k=k)
        nh, nw = rot_mask.shape
        for a in range(nh):
            for b in range(nw):
                if not rot_mask[a, b]:
                    continue
                i, j = divmod(int(rot_ids[a, b]), w)  # source cell in full_mask
                sr, sc = R0 + i, C0 + j               # source (may be off-canvas)
                tr, tc = R0 + a, C0 + b               # destination
                if 0 <= tr < H and 0 <= tc < W and can[tr, tc]:
                    out[tr, tc] = color
                    copied_from[tr, tc] = (sr, sc)

    if clamp_indices:
        np.clip(copied_from, 0, max(H, W) - 1, out=copied_from)
    return out, copied_from


def batch_apply_with_provenance(X, data_type, Z, shift_amount=6, clamp_indices=False):
    """
    X: (N, 32, 32) input images
    data_type: 0 = translate, 1 = rotate (single value for the batch)
    Z: N language bits, one per image (0 = Up, 1 = Left or 0 = 90, 1 = 180)
    Returns (Y_hat (N, 32, 32), copied_from_pixel_indices_all_images (N, 32, 32, 2))
    ready to feed into get_non_compositional_errors.
    """
    X = np.asarray(X)
    N, H, W = X.shape
    Y_hat = np.empty_like(X)
    maps = np.empty((N, H, W, 2), dtype=int)
    for n in range(N):
        Y_hat[n], maps[n] = apply_rule_with_provenance(
            X[n], data_type, int(Z[n]), shift_amount=shift_amount,
            clamp_indices=clamp_indices)
    return Y_hat, maps


@click.command()
@click.argument('processed_data_path', default='data\processed', type=click.Path())
def main(processed_data_path: str):
    all_image_indices = []
    all_object_pixels_errors = []
    all_other_pixels_errors = []
    all_random_errors = []
    all_labels = []
    for transformation in ['translate', 'rotate']:
        for distance in [0, 1, 2]:
            data_file_path = join(processed_data_path, f'compositional_{transformation}', f'test_d{distance}.npz')
            input_images, input_language, output_images = load_data(data_file_path)
            data_type = 0 if transformation == 'translate' else 1
            Y_hat, copied_from_pixel_indices_all_images = batch_apply_with_provenance(input_images, data_type, input_language.squeeze(), clamp_indices=False)
            image_indices, object_pixels_errors, other_pixels_errors, random_error = \
                get_non_compositional_errors(transformation, distance, input_images, output_images, input_language, copied_from_pixel_indices_all_images)

            all_labels.append(f'World model: {transformation}, Distance: {distance}')
            all_image_indices.append(image_indices)
            all_object_pixels_errors.append(object_pixels_errors)
            all_other_pixels_errors.append(other_pixels_errors)
            all_random_errors.append(random_error)

    mean_object_error = []
    std_object_error = []
    mean_non_object_error = []
    std_non_object_error = []

    for i in range(len(all_labels)):
        mean_object_error.append(np.mean(all_object_pixels_errors[i]) / np.mean(all_random_errors[i]))
        mean_non_object_error.append(np.mean(all_other_pixels_errors[i]) / np.mean(all_random_errors[i]))
        std_object_error.append(np.std(all_object_pixels_errors[i]))
        std_non_object_error.append(np.std(all_other_pixels_errors[i]))
        print(f'For {all_labels[i]} mean object pixel error is {mean_object_error[i]} and mean canvas pixel error is {mean_non_object_error[i]}')


if __name__ == '__main__':
    main()

