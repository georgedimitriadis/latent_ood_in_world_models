

"""
Classify OCL test results into categories 1-5 and detect the ground-truth rule
for every sample by exact forward rendering.

Requires apply_rule_v2.py (the forward renderer with canonical shape
completion) in the same directory.

Categories (gen_idx vs target_idx):
  1 = identical
  2 = same shape, followed rule (correct position), wrong color
  3 = different shape, followed rule (correct position), color ignored
  4 = same shape, did not follow rule (wrong position), color ignored
  5 = neither

Rule detection: for each sample, the query object is completed to its canonical
shape and each of the family's two rules is rendered forward; the rule whose
output equals target_idx pixel-for-pixel is the ground truth. This is exact and
unambiguous on all 5994 samples.

Usage:  python classify_ocl_results_v2.py ocl_test_results.pcl
   or:  from classify_ocl_results_v2 import analyze_all_results
"""

from experiments.analysis.ocl_related.get_ocl_test_data import get_ocl_results
import pickle
import numpy as np
from os.path import join
from pathlib import Path
from collections import Counter, defaultdict
import click
import pandas as pd
from metrics.metric_validation import apply_rule, complete_object

BACKGROUND = 0
CANVAS = 1
BITS_TO_NAME = {(0, 0): 'up6', (0, 1): 'left6', (1, 0): 'rot90ccw', (1, 1): 'rot180'}
FAMILY_BITS = {'translate': [(0, 0), (0, 1)], 'rotate': [(1, 0), (1, 1)]}
CATEGORY_DESCRIPTIONS = ['Fully correct', 'Correct shape and rule - Wrong colour', 'Correct rule - Wrong shape', 'Correct shape - Wrong rule', 'All (shape and rule) Wrong']

def create_results(saved_models_path, processed_data_path, pickle_file_dir):
    pickle_file = join(pickle_file_dir, 'ocl_test_results.pcl')

    all_results = get_ocl_results(saved_models_path, processed_data_path, data_type='test')

    with open(pickle_file, 'wb') as f:
        pickle.dump(all_results, f)


def object_info(img):
    m = ~np.isin(img, [BACKGROUND, CANVAS])
    if not m.any():
        return None
    rows, cols = np.where(m)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    rel = m[r0:r1+1, c0:c1+1]
    vals = img[m]
    uvals, counts = np.unique(vals, return_counts=True)
    color = int(uvals[np.argmax(counts)])
    return {'rel_mask': rel, 'color': color, 'anchor': (r0, c0)}


def detect_rule(query_img, target_img, family):
    """Render-based exact rule detection. Returns 'left6'/'up6'/'rot90ccw'/
    'rot180', or None if no (completion, rule) combination reproduces the
    target (should not happen on well-formed data)."""
    comps = complete_object(query_img, 0 if family == 'translate' else 1)
    candidates = comps if comps else [None]
    for rb, lb in FAMILY_BITS[family]:
        for comp in candidates:
            out = apply_rule(query_img, rb, lb, completion=comp)
            if np.array_equal(out, target_img):
                return BITS_TO_NAME[(rb, lb)]
    return None


def classify_sample(query_img, gen_img, target_img):
    if np.array_equal(gen_img, target_img):
        return 1
    t = object_info(target_img)
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
      categories = {1: [(transform, distance, i), ...], ..., 5: [...]}
      rules      = {(transform, distance, i): 'left6'/'up6'/'rot90ccw'/'rot180'/None}
    """
    categories = defaultdict(list)
    rules = {}
    for transform in ['translate', 'rotate']:
        for distance in [0, 1, 2]:
            d = all_results[transform][distance]
            for i in range(d['gen_idx'].shape[0]):
                cat = classify_sample(d['query'][i], d['gen_idx'][i], d['target_idx'][i])
                categories[cat].append((transform, distance, i))
                rules[(transform, distance, i)] = detect_rule(
                    d['query'][i], d['target_idx'][i], transform)
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
    """
    This will count the different types of errors done on the test data by the OCL.
    The results should be
    Category 1 - Fully correct: 1515  (translate 779, rotate 736)
    Category 2 - Correct shape and rule - Wrong colour: 28  (translate 17, rotate 11)
    Category 3 - Correct rule - Wrong shape: 1032  (translate 244, rotate 788)
    Category 4 - Correct shape - Wrong rule: 248  (translate 228, rotate 20)
    Category 5 - All (shape and rule) Wrong: 3171  (translate 1729, rotate 1442)
    Rules: Counter({'left6': 1922, 'rot180': 1750, 'rot90ccw': 1247, 'up6': 1075})
    :param saved_models_path: The folder where the OCL saved model is
    :param processed_data_path: The folder where the processed data are (data/processed)
    :return:
    """
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
    undetected = [k for k, v in rules.items() if v is None]
    print("Samples with undetected rule:", len(undetected))

    # rule x category cross-table
    print("\nCategory breakdown per rule:")
    cross = Counter()
    cat_of = {s: c for c, lst in categories.items() for s in lst}
    for k, r in rules.items():
        cross[(r, cat_of[k])] += 1
    for r in ['left', 'up', 'rot90', 'rot180']:
        row = "  ".join(f"cat{c}: {cross.get((r, c), 0):>4}" for c in [1, 2, 3, 4, 5])
        print(f"  {r:<9} {row}")

    with open(join(pickle_file_dir, 'categories.pcl'), 'wb') as f:
        pickle.dump(categories, f)

if __name__ == "__main__":

    main()

