
import itertools
import math
import random
from typing import List

from src.structure.geometry.basic_geometry import Dimension2D, Surround
from src.structure.object.primitives import Primitive, Object, Point, Parallelogram


def do_two_objects_overlap(object_a: Primitive | Object, object_b: Primitive | Object) -> bool:

    left = max(object_a.required_dist_to_others.Left, object_b.required_dist_to_others.Left)
    right = max(object_a.required_dist_to_others.Right, object_b.required_dist_to_others.Right)
    up = max(object_a.required_dist_to_others.Up, object_b.required_dist_to_others.Up)
    down = max(object_a.required_dist_to_others.Down, object_b.required_dist_to_others.Down)
    surround = Surround(up, down, left, right)

    top_left_a = object_a.bbox.top_left
    top_left_a.x -= surround.Left
    top_left_a.y += surround.Up
    bottom_right_a = object_a.bbox.bottom_right
    bottom_right_a.x += surround.Right
    bottom_right_a.y -= surround.Down

    #top_left_a.x -= object_a.required_dist_to_others.Left
    #top_left_a.y += object_a.required_dist_to_others.Up
    #bottom_right_a = object_a.bbox.bottom_right
    #bottom_right_a.x += object_a.required_dist_to_others.Right
    #bottom_right_a.y -= object_a.required_dist_to_others.Down

    top_left_b = object_b.bbox.top_left
    #top_left_b.x -= object_b.required_dist_to_others.Left
    #top_left_b.y += object_b.required_dist_to_others.Up
    bottom_right_b = object_b.bbox.bottom_right
    #bottom_right_b.x += object_b.required_dist_to_others.Right
    #bottom_right_b.y -= object_b.required_dist_to_others.Down

    # If either of the new bounding boxes has both dimensions smaller than 1 then there is no overlap
    if (object_a.bbox.top_left.x > object_a.bbox.bottom_right.x and
        object_a.bbox.top_left.y < object_a.bbox.bottom_right.y) or \
            (object_b.bbox.top_left.x > object_b.bbox.bottom_right_b.x and
             object_b.bbox.top_left_b.y < object_b.bbox.bottom_right_b.y):
        return False

    # If one rectangle is on left side of other, then no overlap
    if top_left_a.x > bottom_right_b.x or top_left_b.x > bottom_right_a.x:
        return False

    # If one rectangle is above other, then no overlap
    if bottom_right_a.y > top_left_b.y or bottom_right_b.y > top_left_a.y:
        return False

    return True


def random_permutation(possible_values, num_unique_nums):
    return random.sample(possible_values, num_unique_nums)


def colours_permutations(colours: List[int], max_samples: int = 1000, with_black: bool = True) -> List[dict[int, int]]:
    colours = set(colours)
    all_colours = {2, 3, 4, 5, 6, 7, 8, 9, 10}
    if with_black:
        colours.update({1})
        all_colours.update({1})

    n_mappings = math.perm(len(all_colours), len(colours))
    if (n_mappings > max_samples):
        sampled_mappings = set()
        while len(sampled_mappings) < max_samples:
            sampled_mappings.add(tuple(random_permutation(list(all_colours), len(colours))))
        all_mappings = list(sampled_mappings)
    else:
        all_mappings = list(itertools.permutations(list(all_colours), len(colours)))

    result = []
    for mapping in all_mappings:
        result.append(dict(zip(colours, mapping)))

    return result
