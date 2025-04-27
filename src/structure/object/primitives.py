
from __future__ import annotations

from copy import copy
import json
from json import JSONEncoder
from random import randrange

import numpy as np
from numpy.random import shuffle

from experiments.data.generation import constants as const
from typing import List, Dict, Tuple
from enum import Enum
from scipy import ndimage as ndi

from structure.geometry.probabilities import DistributionOver_ObjectTypes
from structure.object.object import Object
from structure.geometry.basic_geometry import Point, Bbox, Dimension2D, Orientation, Surround, Vector, Colour

MAX_PAD_SIZE = const.MAX_PAD_SIZE


# TODO: JSON Encoder is broken. Needs fixing

class PrimitivesJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return 'actual_pixels_id_placeholder' #obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, Orientation):
            return obj.__repr__()
        return obj.__dict__


class ObjectType(Enum):
    Random: int = 0
    Parallelogram: int = 1
    Cross: int = 2
    Hole: int = 3
    Pi: int = 4
    InverseCross: int = 5
    Dot: int = 6
    Angle: int = 7
    Diagonal: int = 8
    Steps: int = 9
    Fish: int = 10
    Bolt: int = 11
    Spiral: int = 12
    Tie: int = 13
    Pyramid: int = 14
    Maze: int = 15
    Predefined: int = 16

    @staticmethod
    def random(_probabilities: None | DistributionOver_ObjectTypes = None) -> ObjectType:
        # This ignores the Predefined type and will never return it
        if _probabilities is None:
            probabilities = DistributionOver_ObjectTypes()
        else:
            probabilities = _probabilities
        return ObjectType(ObjectType.get_int_from_name(probabilities.sample(size=1)))

    @staticmethod
    def get_int_from_name(name: str) -> int:
        for t in ObjectType:
            if t.name == name:
                return t.value

    def generate_primitive(self, args) -> Primitive:
        return globals()[self.name](**args)


class Primitive(Object):
    def __init__(self, size: Dimension2D | np.ndarray | List, colour: None | int = None,
                 border_size: Surround = (0, 0, 0, 0),
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0), *args):
        """
        A basic class for common Primitive data and methods
        :param size: The x, y size of the Primitive
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Primitive
        :param required_dist_to_others: The [Up, Down, Left, Right] number of pixels that need to be empty around the
        object when its canvas_pos is calculated
        :param colour: The colour of the Primitive
        """

        if type(size) == Dimension2D:
            self.size = size
        elif type(size) == list or type(size) == np.ndarray:
            self.size = Dimension2D(size[0], size[1])
        elif size is None:
            size = np.random.randint(2, 20, 2)
            self.size = Dimension2D(size[0], size[1])

        self.border_size = border_size

        if colour is None:
            self.colour = Colour(random=True).index
        else:
            self.colour = colour

        self.required_dist_to_others = required_dist_to_others

    def set_new_colour(self, new_colour: int):
        self.actual_pixels[np.where(self.actual_pixels > 1)] = new_colour
        self.colour = new_colour

    def set_new_size(self, new_size: Dimension2D):
        self.size = new_size

    def create_new_primitives_from_pixels_of_colour(self, colour: int) -> List[Object]:
        pixels = copy(self.actual_pixels)
        s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]  # Allow pixels to be considered connected even when they touch diagonally
        connected_components, n = ndi.label(np.array(pixels == colour).astype(int), structure=s)

        new_objects = []
        for i in range(1, n + 1):
            comp_coords = np.where(connected_components == i)
            d0 = np.max(comp_coords[0]) - np.min(comp_coords[0]) + 1
            d1 = np.max(comp_coords[1]) - np.min(comp_coords[1]) + 1
            new_actual_pixels = self.actual_pixels[np.min(comp_coords[0]): np.min(comp_coords[0]) + d0,
                                                   np.min(comp_coords[1]): np.min(comp_coords[1]) + d1]
            new_canvas_pos = Point(x=self.canvas_pos.x + np.min(comp_coords[1]),
                                   y=self.canvas_pos.y + np.min(comp_coords[0]))
            new_object = Random(size=Dimension2D(array=np.flip(new_actual_pixels.shape)), canvas_pos=new_canvas_pos)
            new_object.actual_pixels = new_actual_pixels
            new_objects.append(new_object)

        return new_objects

    def get_str_type(self):
        return str(type(self)).split('.')[-1].split("'")[0]

    def generate_actual_pixels(self, array: np.ndarray | int) -> np.ndarray:
        """
        Embeds the array into the object actual_pixels
        :param array:
        :return:
        """

        self.size = Dimension2D(array.shape[1], array.shape[0])
        self.dimensions = self.size
        background_size = np.array([self.border_size.Up + self.size.dy + self.border_size.Down,
                                    self.border_size.Left + self.size.dx + self.border_size.Right])
        actual_pixels = np.ones(background_size)
        actual_pixels[self.border_size.Down: self.size.dy + self.border_size.Down,
                      self.border_size.Left: self.size.dx + self.border_size.Left] = array
        return actual_pixels

    def generate_symmetries(self, dirs: str = 'both'):
        """
        Generate symmetries in the centers of the primitive. This doesn't search to see if the symmetries should be there.
        It just creates symmetries whether they actually exist or not.
        :param dirs: 'both', 'x', 'y'. Create only an x only a y or both symmetries
        :return:
        """
        col_pixels_pos = self.get_coloured_pixels_positions()
        xmin = np.min(col_pixels_pos[:, 1])
        xmax = np.max(col_pixels_pos[:, 1])
        ymin = np.min(col_pixels_pos[:, 0])
        ymax = np.max(col_pixels_pos[:, 0])

        if dirs == 'both' or dirs == 'y':
            y_sym_origin = Point((xmax - xmin) / 2 + xmin, ymin)
            y_sym_length = ymax - ymin
            y_sym_or = Orientation.Up
            y_symmetry = Vector(orientation=y_sym_or, length=y_sym_length, origin=y_sym_origin)
            self.symmetries.append(y_symmetry)
        if dirs == 'both' or dirs == 'x':
            x_sym_origin = Point(xmin, (ymax - ymin) / 2 + ymin)
            x_sym_length = xmax - xmin
            x_sym_or = Orientation.Right
            x_symmetry = Vector(orientation=x_sym_or, length=x_sym_length, origin=x_sym_origin)
            self.symmetries.append(x_symmetry)

    def json_output(self):
        args = self.__dict__.copy()
        for arg in ['border_size', '_canvas_pos', 'rotation_axis', 'number_of_coloured_pixels', 'actual_pixels',
                    'required_dist_to_others', 'canvas_id', '_holes', '_center_on', '_relative_points']:
            args.pop(arg, None)
        if 'size' in args:
            args.pop('size', None)

        type_name = self.get_str_type()
        args['primitive'] = type_name
        args['id'] = self.id
        args['canvas_pos'] = [self.canvas_pos.x, self.canvas_pos.y, self.canvas_pos.z]

        args['symmetries'] = [[s.orientation.name, s.length, s.origin.x, s.origin.y] for s in self.symmetries]
        args['dimensions'] = [self.dimensions.dx, self.dimensions.dy]

        #args['transformations'] = [[t[0].value, [t[1][b] if 'axis' not in b else t[1][b].value for b in t[1]]]
        #                           for t in self.transformations]

        args['bbox'] = [[self.bbox.top_left.x, self.bbox.top_left.y],
                        [self.bbox.bottom_right.x, self.bbox.bottom_right.y]]

        if type_name == 'Hole':
            args['thickness'] = [self.thickness.Up, self.thickness.Down, self.thickness.Left, self.thickness.Right]
            args['hole_bbox'] = [[self.hole_bbox.top_left.x, self.hole_bbox.top_left.y],
                                 [self.hole_bbox.bottom_right.x, self.hole_bbox.bottom_right.y]]

        if type_name == 'Dot':
            args['border_size'] = [self.border_size.Up, self.border_size.Down,
                                   self.border_size.Left, self.border_size.Right]

        return args

    def __copy__(self):
        args = self.__dict__.copy()
        for arg in ['_actual_pixels', 'border_size', '_canvas_pos', 'id', 'actual_pixels_id', 'rotation_axis',
                    '_dimensions', 'number_of_coloured_pixels', 'symmetries', 'transformations', 'bbox', '_holes',
                    '_relative_points', '_visible_bbox', '_perimeter', '_inside']:
            args.pop(arg, None)
        args['_id'] = self.id
        args['actual_pixels_id'] = self.actual_pixels_id
        type_name = self.get_str_type()
        if type_name in np.array(['Predefined', 'InverseCross', 'Steps', 'Pyramid', 'Dot', 'Diagonal', 'Fish', 'Bolt', 'Tie']):
            args.pop('size', None)
        if type_name == 'Predefined':
            args.pop('colour', None)
            args.pop('canvas_pos', None)
            args['actual_pixels'] = self.actual_pixels
        if type_name == 'Hole':
            args.pop('hole_bbox', None)
        if type_name == 'Dot':
            args['border_size'] = self.border_size

        object = type(self)(**args)
        object.canvas_pos = Point(self.canvas_pos.x, self.canvas_pos.y, self.canvas_pos.z)
        object._holes = copy(self.holes)
        object.dimensions = Dimension2D(self.dimensions.dx, self.dimensions.dy)
        object.actual_pixels = np.ndarray.copy(self.actual_pixels)
        object.transformations = copy(self.transformations)
        object._visible_bbox = copy(self._visible_bbox)
        object.colour = copy(self.colour)
        object.symmetries = []
        for sym in self.symmetries:
            object.symmetries.append(copy(sym))
        object.border_size = Surround(Up=self.border_size.Up, Down=self.border_size.Down, Left=self.border_size.Left,
                                      Right=self.border_size.Right)
        object.rotation_axis = Point(self.rotation_axis.x, self.rotation_axis.y, self.rotation_axis.z)
        return object

    def __repr__(self):
        return json.dumps(self, cls=PrimitivesJSONEncoder, sort_keys=True, indent=4)

    def __eq__(self, other: Primitive) -> bool:
        result = True
        if self.colour != other.colour:
            result = False
        if not np.array_equal(self.actual_pixels, other.actual_pixels):
            result = False
        if self.size != other.size:
            result = False
        if self.border_size != other.border_size:
            result = False
        if self.required_dist_to_others != other.required_dist_to_others:
            result = False
        if self.dimensions != other.dimensions:
            result = False
        if self.actual_pixels_id != other.actual_pixels_id:
            result = False
        if self.bbox != other.bbox:
            result = False
        if self.visible_bbox != other.visible_bbox:
            return False
        if self.symmetries != other.symmetries:
            result = False
        if self.number_of_coloured_pixels != other.number_of_coloured_pixels:
            result = False
        if self.rotation_axis != other.rotation_axis:
            result = False
        if self.perimeter != other.perimeter:
            return False
        if self.inside != other.inside:
            return False

        return result

    def __hash__(self):
        data = copy(self.actual_pixels)
        data.flags.writeable = False
        return hash(tuple((self.id, self.canvas_id, self.canvas_pos.x, self.canvas_pos.y, data.data.tobytes())))

    def __add__(self, other):

        canvas = np.ones((32, 32))
        if self.canvas_pos.z < other.canvas_pos.z:
            canvas[self.canvas_pos.y: self.canvas_pos.y + self.dimensions.dy,
                   self.canvas_pos.x: self.canvas_pos.x + self.dimensions.dx][np.where(self.actual_pixels>1)] = self.actual_pixels[np.where(self.actual_pixels>1)]
            canvas[other.canvas_pos.y: other.canvas_pos.y + other.dimensions.dy,
                   other.canvas_pos.x: other.canvas_pos.x + other.dimensions.dx][np.where(other.actual_pixels>1)] = other.actual_pixels[np.where(other.actual_pixels>1)]
        else:
            canvas[other.canvas_pos.y: other.canvas_pos.y + other.dimensions.dy,
                   other.canvas_pos.x: other.canvas_pos.x + other.dimensions.dx][np.where(other.actual_pixels>1)] = other.actual_pixels[np.where(other.actual_pixels>1)]
            canvas[self.canvas_pos.y: self.canvas_pos.y + self.dimensions.dy,
                   self.canvas_pos.x: self.canvas_pos.x + self.dimensions.dx][np.where(self.actual_pixels>1)] = self.actual_pixels[np.where(self.actual_pixels>1)]

        pixels = np.where(canvas > 1)
        actual_pixels = canvas[pixels[0].min(): pixels[0].max() + 1, pixels[1].min():pixels[1].max() + 1]

        result = Predefined(actual_pixels=actual_pixels)
        result.translate_to_coordinates(Point(pixels[1].min(), pixels[0].min()))
        return result

    def get_features(self) -> Dict:
        features = {}

        features['Canvas Position X'] = self.canvas_pos.x
        features['Canvas Position Y'] = self.canvas_pos.y
        features['Canvas Position Z'] = self.canvas_pos.z
        features['Dimension X'] = self.dimensions.dx
        features['Dimension Y'] = self.dimensions.dy

        features['Num of Colours'] = len(self.get_used_colours())
        features['Num of Pixels with Colour'] = self.get_number_of_pixels_for_each_colour()

        features['Most common Colour'] = self.get_most_common_colour()
        colours = np.array(list(self.get_used_colours())).astype(int)
        one_hot_colours = np.zeros(10).astype(int)
        one_hot_colours[colours] = 1
        features['Used Colours'] = one_hot_colours

        features['Type'] = ObjectType.get_int_from_name(self.get_str_type())
        features['Symmetries'] = []  # TODO: fill this

        features['2x2 Shape Index'] = self.get_2x2_shape_index()
        features['3x3 Shape Index'] = self.get_3x3_shape_index()
        _, n_holes, size_of_holes = self.detect_holes()
        features['Num of Holes'] = n_holes
        ten_holes_sizes = np.zeros(10) - 1
        if size_of_holes is not None:
            ten_holes_sizes[: len(size_of_holes)] = size_of_holes
        features['Hole sizes'] = ten_holes_sizes

        return features


class Predefined(Primitive):
    def __init__(self, actual_pixels: np.ndarray, border_size: Surround = Surround(0, 0, 0, 0),
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):

        size = Dimension2D(dx=actual_pixels.shape[1], dy=actual_pixels.shape[0])
        bin_count = np.bincount(actual_pixels[np.where(actual_pixels > 1)].astype(int))
        colour = int(np.argmax(bin_count)) if len(bin_count) > 0 else 1
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)
        Object.__init__(self, actual_pixels=actual_pixels, border_size=border_size,
                        canvas_pos=Point(0, 0, 0), _id=_id, actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


class Random(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int | Colour | List[int] | List[Colour] = None,
                 occupancy_prob: float = 0.5,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):

        colours = [Colour.random([Colour.Black]).index]
        if isinstance(colour, Colour):
            colours = [colour.index]
        elif isinstance(colour, int) or isinstance(colour, np.int32):
            colours = [colour]
        elif isinstance(colour, list):
            if isinstance(colour[0], Colour):
                colours = [c.index for c in colour]
            else:
                colours = colour

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=None)

        array = np.ones((self.size.dy, self.size.dx))
        while np.all(array == 1):
            for x in range(self.size.dx):
                for y in range(self.size.dy):
                    if np.random.random() < occupancy_prob:
                        array[y, x] = np.random.choice(colours)

        self.colour = max(set(colours), key=colours.count)

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=array), border_size=border_size,
                        canvas_pos=canvas_pos,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


class Parallelogram(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0), _id: None | int = None,
                 actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A solid colour parallelogram inside a black region (border)
        :param size: The size of the parallelogram
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the parallelogram
        :param canvas_pos: The position in the canvas
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        array = np.ones((self.size.dy, self.size.dx)) * self.colour

        Object.__init__(self, canvas_pos=canvas_pos, actual_pixels=self.generate_actual_pixels(array=array),
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)

        self.generate_symmetries()


class Cross(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0), _id: None | int = None,
                 actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A single colour cross surrounded by black border
        :param size: Dimension2D. The x, y size of the cross. Since the cross has to be symmetric the dx, dy should be
        odd.
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the cross
        :param canvas_pos: The position on the canvas
        :param colour: The cross' colour
        :param _id: The id of the object
        """
    
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        cross = []
        for x in range(self.size.dx):
            temp = np.ones(self.size.dy)
            if x != self.size.dx / 2 - 0.5:
                temp[int(self.size.dy / 2)] = self.colour
            else:
                temp *= self.colour
            cross.append(temp)
        cross = np.transpose(cross)

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=cross), canvas_pos=canvas_pos,
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)

        self.generate_symmetries()


class Hole(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, thickness: Surround = Surround(1, 1, 1, 1),
                 border_size: Surround = Surround(0, 0, 0, 0), canvas_pos: Point = Point(0, 0),
                 colour: None | int = None, required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        This is a hole formed by an outside coloured parallelogram and an inside black parallelogram. The object also
        holds the position of the black hole as a self.hole_bbox Bbox.
        :param size: The x, y size of the surround
        :param thickness: The thickness of the coloured surround ([Up, Down, Left, Right[)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the object
        :param canvas_pos: The position on the canvas
        :param colour: The surround colour
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        array = np.ones((self.size.dy, self.size.dx)) * self.colour
        Object.__init__(self, canvas_pos=canvas_pos, actual_pixels=self.generate_actual_pixels(array=array),
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)

        th_up = thickness.Up
        th_down = thickness.Down
        th_left = thickness.Left
        th_right = thickness.Right

        self.thickness = thickness

        self.actual_pixels[border_size.Down + th_down: self.size.dy + border_size.Down - th_up,
                           border_size.Left + th_left: self.size.dx + border_size.Left - th_right] = 1

        self.hole_bbox = Bbox(top_left=Point(self.border_size.Left + th_left, self.size.dy + self.border_size.Down - th_up - 1),
                              bottom_right=Point(self.size.dx + self.border_size.Left - th_right - 1, self.border_size.Down + th_down))

        sym = None
        if th_up == th_down:
            sym = 'x'
        if th_left == th_right:
            sym = 'y'
        if th_up == th_down and th_left == th_right:
            sym = 'both'

        if sym is not None:
            self.generate_symmetries(sym)


class Pi(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others:  Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id : None | int = None):
        """
        A Pi shaped object.
        :param size: The x, y size of the object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Pi
        :param _canvas_pos: The position on the canvas
        :param colour: The Pi's colour
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        pi = []
        for y in range(self.size.dy):
            temp = np.ones(self.size.dx)
            if y == self.size.dy - 1:
                temp[:] = self.colour
            else:
                temp[0] = self.colour
                temp[-1] = self.colour
            pi.append(temp)
        pi = np.array(pi)

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=pi), canvas_pos=canvas_pos,
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)

        self.generate_symmetries('y')


class InverseCross(Primitive):
    def __init__(self, height: int, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 fill_colour: None | int = None, fill_height: None | int = None,
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A cross made out of a central dot with four arms 45 degrees to the vertical and perpendicular with a second
        colour (fill_colour) filling up to a point (fill_height) the inbetween pixels

        :param height: The height of the cross (it is fully symmetric so width is same as height)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the cross
        :param canvas_pos: The position on the canvas
        :param colour: The cross' colour
        :param fill_colour: The 2nd colour to fill_holes the black pixels surrounding the center dot
        :param fill_height: The number of pixels to get the 2nd colour away from the center
        :param _id: The id of the object
        """
        '''
        assert height % 2 == 1, print('To make an Inverted Cross the height must be an odd number')
    
        if fill_height is not None:
            assert fill_height % 2 == 1, print('To fill_holes an Inverted Cross the fill_height must be an odd number')
        '''

        self.height = height
        size = Dimension2D(height, height)
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        if fill_colour is None:
            fill_colour = self.colour
            while fill_colour == self.colour:
                fill_colour = Colour(random=True).index
        self.fill_colour = fill_colour

        cross = []
        for x in range(self.size.dx):
            temp = np.ones(self.size.dy)

            if fill_height is not None:
                self.fill_height = fill_height
                fill = int((self.size.dx - fill_height) / 2)
                if fill <= x < self.size.dx - fill:
                    temp[fill:-fill] = fill_colour

            temp[x] = self.colour
            temp[-x-1] = self.colour
            cross.append(temp)
        cross = np.array(cross)

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=cross), canvas_pos=canvas_pos,
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)

        self.generate_symmetries()


class Dot(Primitive):
    def __init__(self, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A solid colour Dot inside a black region (border)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Dot
        :param canvas_pos: The position in the canvas
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=Dimension2D(1, 1), border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        Object.__init__(self, canvas_pos=canvas_pos,
                        actual_pixels=self.generate_actual_pixels(array=np.array([[self.colour]])),
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)


class Angle(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A 90 degrees down left pointing angle
        :param size: The x, y size of the object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Angle
        :param canvas_pos: The position on the canvas
        :param colour: The Angle's colour
        :param _id: The id of the object
        """
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        angle = []
        for y in range(self.size.dy):
            temp = np.ones(self.size.dx)
            if y != 0:
                temp[0] = self.colour
            else:
                temp[:] = self.colour
            angle.append(temp)
        angle = np.array(angle)

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=angle), canvas_pos=canvas_pos,
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)


class Diagonal(Primitive):
    def __init__(self, height: int, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A Diagonal line
        :param height: The number of pixels
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Diagonal Line
        :param canvas_pos: The position on the canvas
        :param colour: The Diagonal Line's colour
        :param _id: The id of the object
        """

        self.height = height
        size = Dimension2D(height, height)

        super().__init__(size=size, border_size=border_size,
                         required_dist_to_others=required_dist_to_others, colour=colour)

        diagonal = self.create_diagonal_pixels(height)

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=diagonal), canvas_pos=canvas_pos,
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)

        # TODO: Add diagonal symmetries
        '''
        sym_origin = Point(canvas_pos.x + border_size[2] + length, canvas_pos.y + border_size:.Down)
        self.symmetries.append(Vector(orientation=Orientation.Up_Left, length=length, origin=sym_origin))
        '''

    def create_diagonal_pixels(self, height) -> np.ndarray:
        diagonal = np.ones((height, height))
        self.size = Dimension2D(height, height)
        for y in range(self.size.dy):
            for x in range(self.size.dx):
                if x == y:
                    diagonal[y, x] = self.colour

        diagonal = np.array(diagonal)

        return diagonal

    def change_height(self, by: Vector):
        new_canvas_pos = copy(self.canvas_pos) if by.orientation == Orientation.Up_Right \
            else copy(self.canvas_pos) - by.length
        new_height = self.height + by.length

        self.canvas_pos = new_canvas_pos

        new_pixels = self.create_diagonal_pixels(new_height)
        self.generate_actual_pixels(array=new_pixels)
        #self._reset_dimensions()

        transformations = copy(self.transformations)
        for tr in transformations:
            tranformation = tr[0]
            tr_args = tr[1]
            transform_method = getattr(self, tranformation.name)
            transform_method(**tr_args)


class Steps(Primitive):
    def __init__(self, height: int, depth: int, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id : None | int = None):
        """
        A Steps object from top left to bottom right.
        :param height: The height of the Steps (which will determine its width also)
        :param depth: The number of pixels filling the Steps
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Steps
        :param canvas_pos: The position on the canvas
        :param colour: The Steps' colour
        :param _id: The id of the object
        """

        self.height = height
        self.depth = depth

        size = Dimension2D(height, height)
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        zigzag = []
        for y in range(self.size.dy):
            temp = np.ones(self.size.dx)
            s = y-depth + 1 if y-depth + 1 > 0 else 0
            temp[s:y + 1] = self.colour
            zigzag.append(temp)
        zigzag = np.flipud(zigzag)

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=zigzag), canvas_pos=canvas_pos,
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)


class Fish(Primitive):
    def __init__(self, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A Fish like object. 50% of the times the center pixel will be black.
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Fish
        :param canvas_pos: The position on the canvas
        :param colour: The Fish's colour
        :param _id: The id of the object
        """
        size = Dimension2D(3, 3)
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        center_colour = self.colour if np.random.random() > 0.5 else 1
        fish = np.array([[1, self.colour, 1],
                         [self.colour, center_colour, self.colour],
                         [1, self.colour, self.colour]])

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=fish), canvas_pos=canvas_pos,
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)


class Bolt(Primitive):
    def __init__(self, _center_on: bool = False, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
                A Bolt like object (a 3x3 cross with its top left and bottom right filled.
                50% of the times the center pixel will be black.
                :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Bolt
                :param canvas_pos: The position on the canvas
                :param colour: The Bolt's colour
                :param _id: The id of the object
                """
        size = Dimension2D(3, 3)
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        self._center_on = _center_on
        center_colour = self.colour if _center_on else 1
        bolt = np.array([[1, self.colour, self.colour],
                         [self.colour, center_colour, self.colour],
                         [self.colour, self.colour, 1]])

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=bolt), canvas_pos=canvas_pos,
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)

    @property
    def center_on(self):
        if self.actual_pixels[2, 2] != 1:
            self._center_on = True
        else:
            self._center_on = False

        return self._center_on


class Tie(Primitive):
    def __init__(self, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A Tie like object (a 4 pixels square connected to a single pixel on its bottom left side)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Tie
        :param canvas_pos: The position on the canvas
        :param colour: The Tie's colour
        :param _id: The id of the object
        """
        size = Dimension2D(3, 3)
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        tie = np.array([[1, self.colour, self.colour],
                         [1, self.colour, self.colour],
                         [self.colour, 1, 1]])

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=tie), canvas_pos=canvas_pos, border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)


class Spiral(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, gap: int = 1,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A clockwise twisting Spiral.
        :param size: The x, y size of the object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Spiral
        :param canvas_pos: The position on the canvas
        :param colour: The Spiral's colour
        :param gap: The number of pixels between parallel arms of the Spiral
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        top = 0
        bottom = size.dy - 1
        left = 0
        right = size.dx - 1
        spiral = np.ones((size.dy, size.dx))

        turn = 0

        while top <= bottom and left <= right:
            for i in range(left, right + 1):
                spiral[top, i] = self.colour
            top += 1
            if turn > 0:
                left += gap
            for i in range(top, bottom + 1):
                spiral[i, right] = self.colour
            right -= 1

            if top <= bottom:
                for i in range(right, left - 1, -1):
                    spiral[bottom, i] = self.colour
                bottom -= 1
                top += gap

            if left <= right:
                for i in range(bottom, top - 1, -1):
                    spiral[i, left] = self.colour
                left += 1
                right -= gap

            bottom -= gap
            turn += 1

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=spiral), canvas_pos=canvas_pos,
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)


class Pyramid(Primitive):
    def __init__(self, height: int = 3, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, full: bool = True,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A Pyramid shaped object.
        :param height: height of the Pyramid
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Pyramid
        :param canvas_pos: The position on the canvas
        :param colour: The Pyramid's colour
        :param full: If Ture the Pyramid will be full, otherwise it will be just two arms of a triangle
        :param _id: The id of the object
        """

        self.height = height
        size = Dimension2D(height * 2 - 1, height)

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        pyramid = []
        for y in range(self.size.dy):
            temp = np.ones(self.size.dx)
            s = int(self.size.dx/2) - y
            e = y - int(self.size.dx/2)
            if e == 0:
                e = self.size.dx
            if full:
                temp[s: e] = self.colour
            else:
                temp[s:s+1] = self.colour
                temp[e-1:e] = self.colour
            pyramid.append(temp)
        pi = np.flipud(np.array(pyramid))

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=pi), canvas_pos=canvas_pos,
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)

        self.generate_symmetries('y')


class Maze(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                 colour: None | int = None, required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None):
        """
        A Maze.
        :param size: The x, y size of the object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Maze
        :param colour: The Maze's colour
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        self.grid_size = size
        if size.dx % 2 == 0:
            self.grid_size.dx -= 1
        if size.dy % 2 == 0:
            self.grid_size.dy -= 1
        maze = np.empty((self.grid_size.dy, self.grid_size.dx))
        maze.fill(colour)

        crow = randrange(1, self.grid_size.dy, 1)
        ccol = randrange(1, self.grid_size.dx, 1)
        track = [(crow, ccol)]
        maze[crow][ccol] = 1

        while track:
            (crow, ccol) = track[-1]
            neighbors = self._find_neighbors(crow, ccol, maze, colour)

            if len(neighbors) == 0:
                track = track[:-1]
            else:
                nrow, ncol = neighbors[0]
                maze[nrow][ncol] = 1
                maze[(nrow + crow) // 2][(ncol + ccol) // 2] = 1

                track += [(nrow, ncol)]

        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=maze), canvas_pos=Point(0, 0, 0),
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id =canvas_id)

    def _find_neighbors(self, r, c, grid, is_wall=1):
        """Find all the grid neighbors of the current position; visited, or not.

        Args:
            r (int): row of cell of interest
            c (int): column of cell of interest
            grid (np.array): 2D maze grid
            is_wall (bool): Are we looking for neighbors that are walls, or open cells?
        Returns:
            list: all neighboring cells that match our request
        """
        ns = []

        if r > 1 and grid[r - 2][c] == is_wall:
            ns.append((r - 2, c))
        if r < self.grid_size.dy - 2 and grid[r + 2][c] == is_wall:
            ns.append((r + 2, c))
        if c > 1 and grid[r][c - 2] == is_wall:
            ns.append((r, c - 2))
        if c < self.grid_size.dx - 2 and grid[r][c + 2] == is_wall:
            ns.append((r, c + 2))

        shuffle(ns)
        return ns


# TODO: Create a Primitive that looks like a snake defined by the corners it has and its front and back end
class Snake(Primitive):
    pass


# TODO: Create a grid like primitive that covers sometimes covers the whole canvas
class Grid(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                 colour: None | int = None, required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None,
                 square_size: Dimension2D = Dimension2D(2, 2), num_of_squares_in_a_line: int = 4):
        """
        A Grid.
        :param size: The x, y size of the object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Grid
        :param colour: The Grid's colour
        :param _id: The id of the Grid
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        grid = np.empty((size.dy, size.dx))


        Object.__init__(self, actual_pixels=self.generate_actual_pixels(array=grid), canvas_pos=Point(0, 0, 0),
                        border_size=border_size,
                        _id=_id, actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)



