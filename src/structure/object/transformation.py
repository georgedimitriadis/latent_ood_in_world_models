from __future__ import annotations

import itertools
from enum import Enum
from typing import List

import numpy as np

from structure.geometry.basic_geometry import Point, Dimension2D, Vector, RelativePoint, Orientation


class ObjectTransformations(int, Enum):
    translate_to_coordinates: int = 0
    translate_by: int = 1
    translate_along: int = 2
    translate_relative_point_to_point: int = 3
    translate_until_touch: int = 4
    translate_until_fit: int = 5
    rotate: int = 6
    scale: int = 7
    shear: int = 8
    mirror: int = 9
    flip: int = 10
    grow: int = 11
    randomise_colour: int = 12
    randomise_shape: int = 13
    replace_colour: int = 14
    replace_all_colours: int = 15
    delete: int = 16
    fill_holes: int = 17
    fill: int = 18

    def get_random_parameters(self, random_obj_or_not: str = 'Random'):
        args = {}
        if self.name == 'translate_to_coordinates':
            args['target_point'] = Point.random(min_x=0, min_y=0, min_z=0)
            args['object_point'] = Point.random(min_x=0, min_y=0, min_z=0)
        if self.name == 'translate_by':
            args['distance'] = Dimension2D.random(min_dx=-20, max_dx=20, min_dy=-20, max_dy=20)
        if self.name == 'translate_along':
            args['direction'] = Vector.random()
        if self.name == 'translate_relative_point_to_point':
            args['relative_point'] = RelativePoint.random()
            args['other_point'] = Point.random(min_x=0, min_y=0, min_z=0)
        if self.name == 'scale':
            args['factor'] = np.random.choice([-4, -3, -2, 2, 3, 4], p=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05])
        if self.name == 'rotate':
            args['times'] = np.random.randint(1, 4)
        if self.name == 'shear':
            if random_obj_or_not == 'Random':
                args['_shear'] = int(np.random.gamma(shape=1, scale=15) + 10)  # Mainly between 1 and 75
            else:
                args['_shear'] = int(np.random.gamma(shape=1, scale=10) + 5)  # Mainly between 0.05 and 0.4
                args['_shear'] = 40 if args['_shear'] > 40 else args['_shear']
        if self.name == 'mirror' or self.name == 'flip':
            args['axis'] = np.random.choice([Orientation.Up, Orientation.Down, Orientation.Left, Orientation.Right])
        if self.name == 'mirror':
            args['on_axis'] = False if np.random.rand() < 0.5 else True
        if self.name == 'randomise_colour':
            if random_obj_or_not == 'Random':
                args['ratio'] = int(np.random.gamma(shape=2, scale=10) + 1)  # Mainly between 10 and 40
                args['ratio'] = int(60) if args['ratio'] > 60 else args['ratio']
            else:
                args['ratio'] = int(np.random.gamma(shape=3, scale=5) + 2)  # Mainly between 10 and 40
                args['ratio'] = int(60) if args['ratio'] > 60 else args['ratio']
        if self.name == 'randomise_shape':
            args['add_or_subtract'] = 'sum' if np.random.random() > 0.5 else 'subtract'
            if random_obj_or_not == 'Random':
                args['ratio'] = int(np.random.gamma(shape=3, scale=7) + 1)  # Mainly between 0.1 and 0.3
                args['ratio'] = 50 if args['ratio'] > 50 else args['ratio']
            else:
                args['ratio'] = int(np.random.gamma(shape=3, scale=5) + 2)  # Mainly between 0.1 and 0.3
                args['ratio'] = 50 if args['ratio'] > 50 else args['ratio']
        if self.name == 'replace_colour':
            args['initial_colour'] = np.random.choice(np.arange(2, 11))
            args['final_colour'] = np.random.choice(np.arange(2, 11))
        if self.name == 'replace_all_colours':
            new_colours = np.arange(2, 11)
            np.random.shuffle(new_colours)
            args['colour_swap_hash'] = {2: new_colours[0], 3: new_colours[1], 4: new_colours[2], 5: new_colours[3],
                                        6: new_colours[4], 7: new_colours[5], 8: new_colours[6], 9: new_colours[7],
                                        10: new_colours[8]}
        if self.name == 'fill_holes' or self.name == 'fill':
            args['colour'] = np.random.randint(2, 11)

        return args

    # TODO: Complete the returns for all the other Transformations
    def get_all_possible_parameters(self) -> List[int | Orientation | None]:
        if self.name == 'rotate':
            return [None, 1, 2, 3]
        if self.name == 'scale':
            return [-3, -2, 1, 2, 3]
        if self.name == 'flip':
            return [None, Orientation.Up, Orientation.Up_Right, Orientation.Right,
                    Orientation.Down_Right, Orientation.Down,
                    Orientation.Down_Left, Orientation.Left, Orientation.Up_Left]
        if self.name == 'mirror':
            params = list(itertools.product(*[[Orientation.Up, Orientation.Up_Right, Orientation.Right,
                                               Orientation.Down_Right, Orientation.Down,
                                               Orientation.Down_Left, Orientation.Left, Orientation.Up_Left],
                                               [False, True]]))
            params.append(None)
            return params

    @staticmethod
    def list_to_transformation_args(transformation_name, values):
        args = {}
        if transformation_name == 'translate_to_coordinates':
            args['target_point'] = Point(values[0][0], values[0][1])
            args['object_point'] = Point(values[1][0], values[1][1])
        if transformation_name == 'translate_by':
            args['distance'] = Dimension2D(values[0], values[1])
        if transformation_name == 'translate_along':
            args['direction'] = Vector(Orientation.get_orientation_from_name(values[0]), values[1],
                                       Point(values[2][0], values[2][0]))
        if transformation_name == 'translate_until_touch':
            args['other'] = values
        if transformation_name == 'translate_relative_point_to_point':
            args['relative_point'] = values[0]
            args['other_point'] = values[1]
        if transformation_name == 'translate_until_fit':
            args['other'] = values
        if transformation_name == 'rotate':
            args['times'] = values
        elif transformation_name == 'scale':
            args['factor'] = values
        elif transformation_name == 'shear':
            args['_shear'] = values
        elif transformation_name == 'mirror':
            args['axis'] = Orientation.get_orientation_from_name(values[0])
            args['on_axis'] = values[1]
        elif transformation_name == 'flip':
            args['axis'] = Orientation.get_orientation_from_name(values)
        elif transformation_name == 'randomise_colour':
            args['ratio'] = values
        elif transformation_name == 'randomise_shape':
            args['ratio'] = values
        elif transformation_name == 'replace_colour':
            args['initial_colour'] = values[0]
            args['final_colour'] = values[1]
        elif transformation_name == 'replace_all_colours':
            args['colours_hash'] = values
        elif transformation_name == 'fill_holes' or transformation_name == 'fill':
            args['colour'] = values

        return args

    @staticmethod
    def get_transformation_from_name(name: str) -> ObjectTransformations:
        for i in range(len(ObjectTransformations)):
            if ObjectTransformations(i).name == name:
                return ObjectTransformations(i)


# TODO: Not used yet. Let's see if it will become useful
class CanvasTransformations(Enum):
    add_new_object: int = 0
    remove_object: int = 1
    group_objects: int = 2
    translate_group_to_coordinates: int = 3
    translate_group_by: int = 4
    translate_group_along: int = 5
    rotate_group: int = 6
    scale_group: int = 7
    mirror_group: int = 8
    flip_group: int = 9
    replace_group_colour: int = 10
    replace_group_all_colours: int = 11
    delete_group: int = 12
    or_objects_or: int = 13
    xor_objects_xor: int = 14
    and_objects: int = 15

    @staticmethod
    def get_transformation_from_name(name: str) -> ObjectTransformations:
        for i in range(len(ObjectTransformations)):
            if ObjectTransformations(i).name == name:
                return ObjectTransformations(i)