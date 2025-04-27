
import inspect
from typing import List, Dict, Any, Union

import numpy as np

from structure.canvas.canvas import Canvas
from structure.geometry.basic_geometry import Point, Dimension2D, Colour, Surround, Surround_Percentage, Orientation
from structure.geometry.probabilities import DistributionOver_ObjectTypes, DistributionOver_ObjectTransformations, \
    UniformDistribution
from structure.object.primitives import Primitive, ObjectType
from dsl.functions import dsl_functions as dsl


class RandomPrimitiveInCanvas:

    def __init__(self,
                 main_colour: Colour | List[Colour] | int | List[int] | None = None,
                 secondary_colour: Colour | int | None = None,
                 debug: bool = False):
        """
                Create a new Primitive chosen randomly.
                :param main_colour: The Colour of the Primitive
                :param secondary_colour: A Colour for Primitives that require a second colour
                :param debug: If True print some stuff
                """

        self.main_colour = main_colour
        self.secondary_colour = secondary_colour
        self.debug = debug


    @staticmethod
    def generate_random_type(obj_probs: DistributionOver_ObjectTypes) -> ObjectType:
        """
        Generate a random type given the type distribution
        :param obj_probs: The probabilities of each type to be generated
        :return: The ObjectType
        """
        obj_type = ObjectType.random(_probabilities=obj_probs)
        return obj_type

    @staticmethod
    def generate_random_colour(except_colours: List[Colour]) -> Colour:
        """
        Generate a random Colour that is not one of the except_colours ones
        :param except_colours: Colours not to select from
        :return: The Colour
        """
        return Colour.random(except_colours)

    def generate_random_canvas_pos(self, obj: Primitive, canvas: Canvas,
                                   allowed_canvas_limits: Surround_Percentage | Surround =
                                   Surround_Percentage(Up=0.25, Down=0.25, Left=0.25, Right=0.25),
                                   excluded_points: List[Point] = ()) -> Primitive | None:
        """
        Generates a random canvas_pos that is allowed given the size of the Canvas and its Primitives' required_dist_to_others values
        :param obj: The Primitive to give a canvas_pos
        :param allowed_canvas_limits: The limits of the canvas that the object is allowed to be in. If it is Surround_Percentage then the numbers are floats denoting the percentage of the object's size allowed to be out of canvas (so Left=0.2 means allow the object to be out of canvas on the left by 20% of its length). If it is Surround then the numbers are the canvas pixel coordinates (so Up=2, Down= 13 means within the 2nd and the 13th pixel on the y axis).
        :param canvas: The canvas into which the Primitive will go
        :return: A new Object with the newly calculated canvas_pos or None if no good position exists
        """
        def a_try():
            possible_pos = canvas.where_object_fits_on_canvas(obj=obj, allowed_canvas_limits=allowed_canvas_limits,
                                                              excluded_points=excluded_points)
            if self.debug: print(len(possible_pos))
            if len(possible_pos) > 0:
                return np.random.choice(possible_pos)
            return None
        if self.debug: print('BT')
        canvas_pos = a_try()
        if self.debug: print(f'{canvas_pos} AT')
        tries = 0

        if canvas_pos is None:
            if self.debug: print('returning unchanged object')
            return obj

        if self.debug: print(f'translating to new coordinates {canvas_pos}')
        obj.translate_to_coordinates(target_point=canvas_pos)

        if self.debug: print(tries)

        return obj

    def generate_random_canvas_pos_in_relation_to_object(self, main: Primitive, other: Primitive, canvas: Canvas,
                                                         allowed_orientations: List[Orientation],
                                                         allowed_canvas_limits: Surround_Percentage | Surround =
                                                         Surround_Percentage(Up=0.25, Down=0.25, Left=0.25, Right=0.25)
                                                         ) -> Primitive | None:
        """
        Generates a random canvas pos for the main object depending which orientations in respect to the other object
        the main object is allowed to have (Up, Up_Right, etc).
        :param main: The object to get a canvas pos for
        :param other: The other object around which the main object will be oriented
        :param canvas: The canvas
        :param allowed_orientations: A list of the allowed Orientations (e.g. [Orientation.Up])
        :param allowed_canvas_limits: The limits within the canvas (see self.generate_random_canvas_pos)
        :return: The main object with the chosen canvas pos
        """

        all_points = set([Point(x, y) for x in range(0, canvas.size.dx + 1) for y in range(0, canvas.size.dy + 1)])

        available_points = []
        if main.visible_bbox is None:
            main_vis_dims = Dimension2D(main.bbox.bottom_right.x - main.bbox.top_left.x,
                                        main.bbox.top_left.y - main.bbox.bottom_right.y)
        else:
            main_vis_dims = Dimension2D(main.visible_bbox.bottom_right.x - main.visible_bbox.top_left.x,
                                        main.visible_bbox.top_left.y - main.visible_bbox.bottom_right.y)

        if Orientation.Up in allowed_orientations:
            up = [Point(x, y) for x in range(other.visible_bbox.top_left.x - main_vis_dims.dx + 1,
                                             other.visible_bbox.bottom_right.x + main_vis_dims.dx - 1) for y in
                  range(other.visible_bbox.top_left.y + 1, canvas.size.dy)]
            available_points = available_points + up
        if Orientation.Up_Right in allowed_orientations:
            up_right = [Point(x, y) for x in range(other.visible_bbox.bottom_right.x + 1, canvas.size.dx) for y in
                        range(other.visible_bbox.top_left.y + 1, canvas.size.dy)]
            available_points = available_points + up_right
        if Orientation.Right in allowed_orientations:
            right = [Point(x, y) for x in range(other.visible_bbox.bottom_right.x + 1, canvas.size.dx) for y in
                     range(other.visible_bbox.bottom_right.y - main_vis_dims.dy + 1,
                           other.visible_bbox.top_left.y + main_vis_dims.dy - 2)]
            available_points = available_points + right
        if Orientation.Down_Right in allowed_orientations:
            down_right = [Point(x, y) for x in range(other.visible_bbox.bottom_right.x + 1, canvas.size.dx) for y in
                          range(0, other.visible_bbox.bottom_right.y)]
            available_points = available_points + down_right
        if Orientation.Down in allowed_orientations:
            down = [Point(x, y) for x in range(other.visible_bbox.top_left.x - main_vis_dims.dx + 1,
                                               other.visible_bbox.bottom_right.x + main_vis_dims.dx -1 ) for y in
                    range(0, other.visible_bbox.top_left.y)]
            available_points = available_points + down
        if Orientation.Down_Left in allowed_orientations:
            down_left = [Point(x, y) for x in range(0, other.visible_bbox.bottom_right.x) for y in
                         range(0, other.visible_bbox.bottom_right.y)]
            available_points = available_points + down_left
        if Orientation.Left in allowed_orientations:
            left = [Point(x, y) for x in range(0, other.visible_bbox.bottom_right.x) for y in
                    range(other.visible_bbox.bottom_right.y - main_vis_dims.dy + 1,
                          other.visible_bbox.top_left.y + main_vis_dims.dy - 2)]
            available_points = available_points + left
        if Orientation.Up_Left in allowed_orientations:
            up_left = [Point(x, y) for x in range(0, other.visible_bbox.bottom_right.x) for y in
                       range(other.visible_bbox.top_left.y + 1, canvas.size.dy)]
            available_points = available_points + up_left

        #if self.debug: print(available_points)
        excluded_points = all_points - set(available_points)

        return self.generate_random_canvas_pos(obj=main, canvas=canvas, allowed_canvas_limits=allowed_canvas_limits,
                                               excluded_points=list(excluded_points))

    @staticmethod
    def generate_random_dimensions(arg_ranges: Dict[str, UniformDistribution], obj_type: ObjectType) -> Dict:
        """
        Generates random dimensions (size, height, length) according to the type of the Primitive and the user defined
        ranges
        :param arg_ranges: A Dict with different ranges for the different Primitive arguments
        :param obj_type: The type of the Primitive
        :return: The dict that holds the correct dimensions of the Primitive
        """
        args = {}
        if obj_type.name != 'Dot':
            if isinstance(arg_ranges['size'], list):
                size_dist_x = arg_ranges['size'][0]
                size_dist_y = arg_ranges['size'][1]
            else:
                size_dist_x = size_dist_y = arg_ranges['size']

        if obj_type.name in ['InverseCross', 'Steps', 'Pyramid', 'Diagonal']:  # These object have height not size
            if 'height' not in arg_ranges:
                args['height'] = size_dist_y.sample()
            else:
                args['height'] = arg_ranges['height'].sample()
            if obj_type.name == 'InverseCross' and args['height'] % 2 == 0:  # Inverted Crosses need odd height
                args['height'] += 1

        if obj_type.name == 'InverseCross':
            if 'fill_height' not in arg_ranges:
                dist = UniformDistribution((1, args['height'] - 2)) if args['height'] > 3 else UniformDistribution(0)
                args['fill_height'] = dist.sample()
            else:
                args['fill_height'] = arg_ranges['fill_height'].sample()
            if args['fill_height'] % 2 == 0:  # Inverted Crosses need odd fill_height
                args['fill_height'] += 1

        if obj_type.name == 'Steps':  # Steps also has depth
            if 'depth' not in arg_ranges:
                args['depth'] = UniformDistribution((1, args['height'])).sample()
            else:
                args['depth'] = arg_ranges['depth'].sample()

        if not np.any(np.array(['InverseCross', 'Steps', 'Pyramid', 'Dot', 'Diagonal', 'Fish', 'Bolt', 'Tie'])
                      == obj_type.name):
            size = Dimension2D(size_dist_x.sample(), size_dist_y.sample())
            if np.any(np.array(['Cross', 'InvertedCross']) == obj_type.name):  # Crosses need odd size
                if size.dx % 2 == 0:
                    size.dx += 1
                if size.dy % 2 == 0:
                    size.dy += 1
            args['size'] = size

        return args

    @staticmethod
    def generate_random_thickness_given_size(size: Dimension2D) -> Surround:
        """
        Generate a random Surround for the thickness of a Hole given its dimensions
        :param size: The dimensions of the Primitive
        :return: The Surround thickness
        """
        if size.dx < 4:
            size.dx = 4
        if size.dy < 4:
            size.dy = 4
        up = np.random.randint(1, size.dy - 2)
        down = np.random.randint(1, size.dy - up)
        left = np.random.randint(1, size.dx - 2)
        right = np.random.randint(1, size.dx - left)

        return Surround(up, down, left, right)

    @staticmethod
    def generate_random_allowed_distance_to_others(obj: Primitive,
                                                   arg_ranges: Dict[str, List[UniformDistribution] | UniformDistribution]) -> Surround:
        """
        Creates a default min_distance_to_others for the Primitive. This will be a Surround and negative numbers mean an
        allowed overlap by that many pixels while positive numbers mean a minimum distance away from the object.
        :param obj: The Primitive
        :param arg_ranges: The dict of the ranges for the different arguments.If there is a 'required_dist_to_others' entry then it can be a list of 4 floats or UniformDistributions or one float or UniformDistribution. If there is no such entry then the default is fully random.
        :return: The min_distance_to_others Surround
        """
        if 'required_dist_to_others' in arg_ranges:
            if isinstance(arg_ranges['required_dist_to_others'], list): # If the required_dist_to_others has separate distributions for each Surround value
                values = []
                for i in range(4):
                    if isinstance(arg_ranges['required_dist_to_others'][i].range, float): # If this is a Distribution with just a float for a range then it is a percentage of the Primitive's dimension
                        values.append(int(obj.dimensions.dx * arg_ranges['required_dist_to_others'][i].sample()) if (i == 0 or i == 1) else \
                                      int(obj.dimensions.dy * arg_ranges['required_dist_to_others'][i].sample()))
                    elif isinstance(arg_ranges['required_dist_to_others'][i].range, tuple) or \
                            isinstance(arg_ranges['required_dist_to_others'][i].range, int): # If it is a proper ranged Distribution or a Distribution with just an int then sample from it
                        values.append(arg_ranges['required_dist_to_others'][i].sample())

                min_distance_to_others = Surround(values[0], values[1], values[2], values[3])

            elif isinstance(arg_ranges['required_dist_to_others'].range, tuple) or \
                isinstance(arg_ranges['required_dist_to_others'].range, int): # If the required_dist_to_others has one distribution with a proper range (or just an int) use it to generate the Surround values
                min_distance_to_others = Surround(arg_ranges['required_dist_to_others'].sample(),
                                                  arg_ranges['required_dist_to_others'].sample(),
                                                  arg_ranges['required_dist_to_others'].sample(),
                                                  arg_ranges['required_dist_to_others'].sample())
            elif isinstance(arg_ranges['required_dist_to_others'].range, float): # If the required_dist_to_others has one distribution with a single float value for a range then this is the percentage of the Primitive's dimentions
                min_distance_to_others = Surround(int(obj.dimensions.dy * arg_ranges['required_dist_to_others'].sample()),
                                                  int(obj.dimensions.dy * arg_ranges['required_dist_to_others'].sample()),
                                                  int(obj.dimensions.dx * arg_ranges['required_dist_to_others'].sample()),
                                                  int(obj.dimensions.dx * arg_ranges['required_dist_to_others'].sample()))
        else: # If there is no required_dist_to_others then use the Primitive's dimensions to create a distribution to sample from
            min_distance_to_others = Surround(UniformDistribution((int(-obj.dimensions.dy),
                                                                   int(obj.dimensions.dy))).sample(),
                                              UniformDistribution((int(-obj.dimensions.dy),
                                                                   int(obj.dimensions.dy))).sample(),
                                              UniformDistribution((int(-obj.dimensions.dy),
                                                                   int(obj.dimensions.dy))).sample(),
                                              UniformDistribution((int(-obj.dimensions.dy),
                                                                   int(obj.dimensions.dy))).sample())

        min_distance_to_others += obj.border_size

        return min_distance_to_others

    def _create_random_primitive(self, obj_probs: DistributionOver_ObjectTypes,
                                 arg_ranges: Dict[str, Union[List[UniformDistribution], UniformDistribution]],
                                 id: int | None = None) \
            -> Primitive | None:
        """
        Creates a randomly generated Primitive
        :param obj_probs: The Distribution of the different types
        :param arg_ranges: A Dict with different Uniform Distributions for the different Primitive arguments
        :return: The Primitive
        """
        args = {'_id': id,
                'actual_pixels_id': id}

        # Type
        obj_type = self.generate_random_type(obj_probs)

        # Border size
        if 'border_size' not in arg_ranges:
            args['border_size'] = Surround(0, 0, 0, 0)
        else:
            if isinstance(arg_ranges['border_size'], list):
                up = arg_ranges['border_size'][0].sample()
                down = arg_ranges['border_size'][1].sample()
                left = arg_ranges['border_size'][2].sample()
                right = arg_ranges['border_size'][3].sample()
            elif isinstance(arg_ranges['border_size'], UniformDistribution):
                up = arg_ranges['border_size'].sample()
                down = arg_ranges['border_size'].sample()
                left = arg_ranges['border_size'].sample()
                right = arg_ranges['border_size'].sample()
            args['border_size'] = Surround(up, down, left, right)

        # Colour
        if self.main_colour is None:
            main_colour = [self.generate_random_colour(except_colours=[Colour.Black]).index]
        else:
            if isinstance(self.main_colour, Colour):
                main_colour = [self.main_colour.index]
            elif isinstance(self.main_colour, list):
                if isinstance(self.main_colour[0], Colour):
                    main_colour = [c.index for c in self.main_colour]
                elif isinstance(self.main_colour[0], int):
                    main_colour = self.main_colour
            elif isinstance(self.main_colour, int):
                main_colour = [self.main_colour]

        if obj_type != ObjectType.Random:
            main_colour = main_colour[0]

        args['colour'] = main_colour

        if obj_type.name == 'InverseCross':
            if self.secondary_colour is None:
                secondary_colour = self.generate_random_colour(except_colours=[Colour.Black, main_colour])
            else:
                secondary_colour = self.secondary_colour
            args['fill_colour'] = secondary_colour.index

        # Size / Height / Depth
        dim_args = self.generate_random_dimensions(arg_ranges=arg_ranges, obj_type=obj_type)
        for k in dim_args:
            args[k] = dim_args[k]

        # Thickness
        if obj_type.name == 'Hole':  # Hole has also thickness
            if 'thickness' in arg_ranges:
                if isinstance(arg_ranges['thickness'], list):
                    args['thickness'] = Surround(arg_ranges['thickness'][0].sample(), arg_ranges['thickness'][1].sample(),
                                                 arg_ranges['thickness'][2].sample(), arg_ranges['thickness'][3].sample())
                else:
                    args['thickness'] = Surround(arg_ranges['thickness'].sample(),
                                                 arg_ranges['thickness'].sample(),
                                                 arg_ranges['thickness'].sample(),
                                                 arg_ranges['thickness'].sample())
            else:
                args['thickness'] = self.generate_random_thickness_given_size(args['size'])

        # Occupancy_prob for Random
        if obj_type.name == 'Random':
            if 'occupancy_prob' in arg_ranges:
                args['occupancy_prob'] = arg_ranges['occupancy_prob'].sample()
            else:
                args['occupancy_prob'] = UniformDistribution(range=(0.1, 0.9)).sample()

        # _center_on for Bolt
        if obj_type.name == 'Bolt':
            if '_center_on' in arg_ranges:
                args['_center_on'] = arg_ranges['_center_on'].sample()
            else:
                args['_center_on'] = UniformDistribution(range=(0, 1)).sample()

        # gap for Bolt
        if obj_type.name == 'Spiral':
            if 'gap' in arg_ranges:
                args['gap'] = arg_ranges['gap'].sample()
            else:
                args['gap'] = UniformDistribution(range=(0, 1)).sample()

        if self.debug: print(args)

        # Object
        obj = obj_type.generate_primitive(args)

        # required_dist_to_others
        obj.required_dist_to_others = self.generate_random_allowed_distance_to_others(obj=obj, arg_ranges=arg_ranges)

        return obj

    def create_random_primitive(self, obj_probs: DistributionOver_ObjectTypes,
                                arg_ranges: Dict[str, Union[List[UniformDistribution], UniformDistribution]]) -> Primitive | None:
        tries = 0
        obj = self._create_random_primitive(obj_probs=obj_probs, arg_ranges=arg_ranges)
        while obj is None and tries < 1000:
            obj = self._create_random_primitive()

        return obj

    @staticmethod
    def do_a_series_of_random_transformations(obj: Primitive, num_of_transformations: int,
                                              trans_probs: DistributionOver_ObjectTransformations,
                                              arg_ranges: Dict[str, Dict[str, Any]]) -> Primitive:
        """
        Do a number of transformations on the obj Primitive. Each transformation is sampled from the trans_probs
        DistributionOver_ObjectTransformations while its arguments are sampled from the arg_ranges[transformation_name]
        dictionary that holds different distributions for each of the arguments of the transformation function.
        :param obj: The Primitive to transform
        :param num_of_transformations: The number of transformations
        :param trans_probs: DistributionOver_ObjectTransformations distribution to sample each transformation from
        :param arg_ranges: A Dict of dicts formatted as follows. arg_ranges['transformation_function_name'] = {'arg1': Distribution, 'arg2': Distribution, etc}
        :return: The transformed Primitive
        """

        for _ in range(num_of_transformations):
            trans_type = trans_probs.sample()
            name_of_args = inspect.getfullargspec(getattr(obj, trans_type)).args
            args = {}
            for name in name_of_args:
                if name in arg_ranges[trans_type]:
                    args[name] = arg_ranges[trans_type][name].sample()
            # do the transformation
            getattr(obj, trans_type)(**args)

        return obj

    @staticmethod
    def empty_canvas_if_objects_overlap(canvas: Canvas):
        overlap = False
        for o in canvas.objects:
            for k in dsl.select_rest_of_the_objects(canvas, o):
                if o.is_object_superimposed(k):
                    overlap = True

        if overlap:
            canvas = Canvas(canvas.size)

        return canvas
