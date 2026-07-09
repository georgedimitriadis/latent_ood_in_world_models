
from experiments.analysis.ocl_related.get_ocl_test_data import get_ocl_results
import pickle
import numpy as np
from os.path import join
from pathlib import Path
from collections import Counter
import click
import pandas as pd

BACKGROUND = 0
CANVAS = 1
CATEGORY_DESCRIPTIONS = ['Fully correct', 'Correct shape and rule - Wrong colour', 'Correct rule - Wrong shape', 'Correct shape - Wrong rule', 'All (shape and rule) Wrong']

def create_results(saved_models_path, processed_data_path, pickle_file_dir):
    pickle_file = join(pickle_file_dir, 'ocl_test_results.pcl')

    all_results = get_ocl_results(saved_models_path, processed_data_path, data_type='test')

    with open(pickle_file, 'wb') as f:
        pickle.dump(all_results, f)

def object_mask(img):
    return ~np.isin(img, [BACKGROUND, CANVAS])

def canvas_mask(img):
    return img != BACKGROUND

def bbox_of(mask):
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())

def object_info(img):
    m = object_mask(img)
    bb = bbox_of(m)
    if bb is None:
        return None
    r0, r1, c0, c1 = bb
    rel = m[r0:r1+1, c0:c1+1]
    vals = img[m]
    uvals, counts = np.unique(vals, return_counts=True)
    color = int(uvals[np.argmax(counts)])
    return {'mask': m, 'bbox': bb, 'rel_mask': rel, 'color': color, 'anchor': (r0, c0)}


def shift_matches(q_img, t_img, dr, dc):
    q_obj = object_mask(q_img); t_obj = object_mask(t_img)
    q_can = canvas_mask(q_img); t_can = canvas_mask(t_img)
    H, W = q_img.shape
    q_obj_s = np.zeros_like(q_obj); q_can_s = np.zeros_like(q_can)
    src_r0 = max(0, -dr); src_r1 = min(H, H - dr)
    src_c0 = max(0, -dc); src_c1 = min(W, W - dc)
    if src_r0 < src_r1 and src_c0 < src_c1:
        q_obj_s[src_r0+dr:src_r1+dr, src_c0+dc:src_c1+dc] = q_obj[src_r0:src_r1, src_c0:src_c1]
        q_can_s[src_r0+dr:src_r1+dr, src_c0+dc:src_c1+dc] = q_can[src_r0:src_r1, src_c0:src_c1]
    both = q_can_s & t_can
    return np.array_equal(q_obj_s & both, t_obj & both)


def _rotation_try(q_obj, t_obj, q_can, t_can, R0, C0, h, w, k):
    """Vectorized consistency test for one candidate full bbox + rotation k."""
    H, W = q_obj.shape
    ids = np.arange(h * w).reshape(h, w)
    rot = np.rot90(ids, k=k)
    nh, nw = rot.shape
    # dest position (within rotated box) of each source cell id
    dest = np.empty((h * w, 2), dtype=int)
    rr, cc = np.divmod(np.arange(nh * nw), nw)
    dest[rot.ravel()] = np.stack([rr, cc], axis=1)

    i_grid, j_grid = np.divmod(np.arange(h * w), w)
    src_r = R0 + i_grid; src_c = C0 + j_grid
    dst_r = R0 + dest[:, 0]; dst_c = C0 + dest[:, 1]

    def sample(mask, r, c):
        inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        out = np.zeros(len(r), dtype=bool)
        out[inb] = mask[r[inb], c[inb]]
        return out, inb

    q_can_v, q_inb = sample(q_can, src_r, src_c)
    t_can_v, t_inb = sample(t_can, dst_r, dst_c)
    q_obj_v, _ = sample(q_obj, src_r, src_c)
    t_obj_v, _ = sample(t_obj, dst_r, dst_c)

    src_known = q_can_v
    dst_known = t_can_v
    both = src_known & dst_known
    if np.any(both & (q_obj_v != t_obj_v)):
        return False

    # every visible target object pixel must lie in the rotated box
    tb = bbox_of(t_obj)
    if tb is not None:
        tr0, tr1, tc0, tc1 = tb
        if tr0 < R0 or tr1 > R0 + nh - 1 or tc0 < C0 or tc1 > C0 + nw - 1:
            return False
    # every visible query object pixel must lie in the (source) box
    qb = bbox_of(q_obj)
    if qb is not None:
        qr0, qr1, qc0, qc1 = qb
        if qr0 < R0 or qr1 > R0 + h - 1 or qc0 < C0 or qc1 > C0 + w - 1:
            return False
    return True


def rotation_matches(q_img, t_img, k_array, max_ext=12):
    q = object_info(q_img)
    if q is None:
        return False
    q_obj = q['mask']; t_obj = object_mask(t_img)
    q_can = canvas_mask(q_img); t_can = canvas_mask(t_img)
    cb = bbox_of(q_can)
    cr0, cr1, cc0, cc1 = cb
    vr0, vr1, vc0, vc1 = q['bbox']

    touches = {
        'b': vr0 == cr0, 't': vr1 == cr1,
        'l': vc0 == cc0, 'r': vc1 == cc1,
    }
    any_touch = any(touches.values())
    vert_clip = touches['b'] or touches['t']
    horz_clip = touches['l'] or touches['r']

    # extension allowed on a side if that side is clipped, or if perpendicular
    # clipping exists (hidden rows/cols can extend the bbox in the other axis)
    def rng(allowed):
        return range(0, max_ext + 1) if allowed else [0]

    eb_r = rng(touches['b'] or horz_clip)
    et_r = rng(touches['t'] or horz_clip)
    el_r = rng(touches['l'] or vert_clip)
    er_r = rng(touches['r'] or vert_clip)

    if not any_touch:
        return _rotation_try(q_obj, t_obj, q_can, t_can, vr0, vc0,
                             vr1 - vr0 + 1, vc1 - vc0 + 1, k_array)

    for eb in eb_r:
        R0 = vr0 - eb
        for et in et_r:
            R1 = vr1 + et
            h = R1 - R0 + 1
            if h > 13:
                continue
            for el in el_r:
                C0 = vc0 - el
                for er in er_r:
                    C1 = vc1 + er
                    w = C1 - C0 + 1
                    if w > 13:
                        continue
                    if _rotation_try(q_obj, t_obj, q_can, t_can, R0, C0, h, w, k_array):
                        return True
    return False


RULES = ['up6', 'left6', 'rot90ccw', 'rot180']
FAMILY_RULES = {'translate': ['up6', 'left6'], 'rotate': ['rot90ccw', 'rot180']}

def rule_matches(q_img, t_img, rule):
    if rule == 'up6':
        return shift_matches(q_img, t_img, +6, 0)
    if rule == 'left6':
        return shift_matches(q_img, t_img, 0, -6)
    if rule == 'rot90ccw':
        return rotation_matches(q_img, t_img, 3)
    if rule == 'rot180':
        return rotation_matches(q_img, t_img, 2)
    raise ValueError(rule)

def detect_rule(q_img, t_img, family=None):
    candidates = FAMILY_RULES[family] if family in FAMILY_RULES else RULES
    matches = [r for r in candidates if rule_matches(q_img, t_img, r)]
    if len(matches) == 1:
        return matches[0], []
    if len(matches) > 1:
        return matches[0], matches
    return None, []


def classify_sample(q_img, gen_img, t_img):
    """
    1 = gen identical to target
    2 = same shape as target, followed rule (correct position), wrong color
    3 = different shape, followed rule (correct position), color ignored
    4 = same shape, did not follow rule (wrong position), color ignored
    5 = neither
    Shape = the object's visible pixel silhouette relative to its bounding box.
    Position/"followed the rule" = the bbox's image bottom-left corner
    (array (min_row, min_col)) matches the target's.
    """
    if np.array_equal(gen_img, t_img):
        return 1
    t = object_info(t_img)
    g = object_info(gen_img)
    if g is None or t is None:
        return 5
    same_shape = (g['rel_mask'].shape == t['rel_mask'].shape and
                  np.array_equal(g['rel_mask'], t['rel_mask']))
    followed_rule = (g['anchor'] == t['anchor'])
    same_color = (g['color'] == t['color'])
    if same_shape and followed_rule and not same_color:
        return 2
    if not same_shape and followed_rule:
        return 3
    if same_shape and not followed_rule:
        return 4
    return 5


def analyze_all_results(all_results):
    """
    Returns (categories, rules):
      categories = {1: [(transform, split, i), ...], ..., 5: [...]}
      rules      = {(transform, split, i): 'up6'/'left6'/'rot90ccw'/'rot180' or None}
    """
    from collections import defaultdict
    categories = defaultdict(list)
    rules = {}
    for transform in ['translate', 'rotate']:
        for split in [0, 1, 2]:
            d = all_results[transform][split]
            for i in range(d['gen_idx'].shape[0]):
                cat = classify_sample(d['query'][i], d['gen_idx'][i], d['target_idx'][i])
                categories[cat].append((transform, split, i))
                r, _amb = detect_rule(d['query'][i], d['target_idx'][i], transform)
                rules[(transform, split, i)] = r
    return dict(categories), rules

def categories_to_df(categories):
    num_of_entries = 0
    for i in categories:
        num_of_entries += len(categories[i])
    df = pd.DataFrame(columns=['category', 'category_description', 'rule_type', 'distance', 'batch_number'], index=range(num_of_entries))
    j = 0
    for i in categories:
        for k in categories[i]:
            df.loc[j, 'category'] = i
            df.loc[j, 'category_description'] = CATEGORY_DESCRIPTIONS[i - 1]
            df.loc[j, 'rule_type'] = k[0]
            df.loc[j, 'distance'] = k[1]
            df.loc[j, 'batch_number'] = k[2]
            j += 1

    return df

@click.command()
@click.argument('saved_models_path', type=click.Path())
@click.argument('processed_data_path', type=click.Path())
def main(saved_models_path, processed_data_path):
    pickle_file_dir = join(Path(processed_data_path).parent.absolute(), 'results')
    if not Path(pickle_file_dir).exists():
        Path(pickle_file_dir).mkdir(exist_ok=True)

    pickle_file = join(pickle_file_dir, 'ocl_test_results.pcl')
    if not Path(pickle_file).exists():
        create_results(saved_models_path, processed_data_path, pickle_file_dir)

    with open(pickle_file, 'rb') as f:
        all_results = pickle.load(f)

    categories, rules = analyze_all_results(all_results)

    for c in [1, 2, 3, 4, 5]:
        lst = categories.get(c, [])
        tcount = sum(1 for x in lst if x[0] == 'translate')
        print(f"Category {c} - {CATEGORY_DESCRIPTIONS[c-1]}: {len(lst)}  (translate {tcount}, rotate {len(lst) - tcount})")

    print("Rules:", Counter(rules.values()))

    with open(join(pickle_file_dir, 'categories.pcl'), 'wb') as f:
        pickle.dump(categories, f)

if __name__ == "__main__":

    main()