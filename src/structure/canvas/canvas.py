

from __future__ import annotations

from copy import copy
from typing import Dict, List, Tuple, Annotated
import matplotlib.pyplot as plt
import numpy as np

from experiments.data.generation.utils import do_two_objects_overlap
from structure.object.object import Object
from structure.utils import union2d
from experiments.data.generation import constants as const
from visualization import basic_visualisation_of_data as vis
from structure.object.primitives import Primitive, Random, Dot, Predefined
from structure.geometry.basic_geometry import Point, Dimension2D, Surround, Surround_Percentage, Colour

MAX_PAD_SIZE = const.MAX_PAD_SIZE
MAX_NUM_OF_CANVAS_OBJECTS = const.MAX_NUM_OF_CANVAS_OBJECTS


class Canvas:
    def __init__(self, size: Dimension2D | np.ndarray | List | None = None, objects: List[Primitive] | None = None,
                 _id: int | None = None, actual_pixels: np.ndarray | None = None,
                 as_grid_x_y_tilesize_colour: Tuple[int, int, int, int | Colour] | None = None):
        """
        Creates a Canvas, representing an ARC's grid. The Canvas is defined by either the size or the actual_pixels or the as_grid_x_y_tilesize_colour arguments. One of them needs to have a value and the other two need to be None.
        :param size: The x, y size of the Canvas. Can be Dimension2D, np.ndarray or list. If it is not None then the actual_pixels and as_grid_x_y_tilesize_colour need to be None.
        :param objects: A list of any objects the Canvas will show.
        :param _id: The id of the Canvas
        :param actual_pixels: A predefined np.ndarray of pixels. This defines the size fo teh Canvas so if it is not None then the size and the as_grid_x_y_tilesize_colour need to be None.
        :param as_grid_x_y_tilesize_colour: This makes a grid background object on the Canvas. It is a four int Tuple. The ints are: the number of grid tiles on the x axis, the number of grid tiles on the y axis, the size of the tile (square) and the colour of the grid lines. If it is not None, the size and actual_pixels must be None.
        """

        assert not(size is None and actual_pixels is None and as_grid_x_y_tilesize_colour is None), print(f'Making a canvas with id {_id}. '
                                                                   f'Size and actual_pixels and as_grid are all None!')

        if type(size) != Dimension2D and size is not None:
            size = [int(size[0]), int(size[1])]
            self.size = Dimension2D(array=size)
        elif type(size) == Dimension2D and size is not None:
            self.size = Dimension2D(int(size.dx), int(size.dy))

        self.id = _id
        self.objects = []
        self.grid = False

        if actual_pixels is None and as_grid_x_y_tilesize_colour is None:
            self.actual_pixels = np.ones((size.dy, size.dx))
        elif actual_pixels is not None and as_grid_x_y_tilesize_colour is None:
            self.actual_pixels = actual_pixels
            self.size = Dimension2D(self.actual_pixels.shape[1], self.actual_pixels.shape[0])

        self.full_canvas = np.zeros((MAX_PAD_SIZE, MAX_PAD_SIZE))

        if as_grid_x_y_tilesize_colour is not None and size is None and actual_pixels is None:
            self.grid = True
            x = as_grid_x_y_tilesize_colour[0]
            y = as_grid_x_y_tilesize_colour[1]
            s = as_grid_x_y_tilesize_colour[2]
            c = as_grid_x_y_tilesize_colour[3]
            if isinstance(c, Colour):
                c = c.index
            self.grid_tiles_coordinates: Dict[Tuple, Point] = {}
            """Keeps the coordinates of each tile. The key Tuple is the (x, y) tile coordinates and the value Point is 
            the coordinates of the bottom left pixel of this tile."""
            self.size_of_tiles: int = s
            self.grid_shape: Tuple[int, int] = (x, y)
            self.grid_colour = c
            self.size = Dimension2D(x * (s + 1) - 1, y * (s + 1) - 1)
            self.make_grid_background()
            self.actual_pixels = self.background_pixels

        self.full_canvas[0: self.size.dy, 0:self.size.dx] = self.actual_pixels
        self.background_pixels = np.ndarray.copy(self.actual_pixels)
        self.objects_features = {}
        self.obj_id_to_index_hashmap = {}
        self.binary_features = {}

        if objects is not None:
            for o in objects:
                self.add_new_object(o)

    def __repr__(self):
        return f'ID: {self.id},   {self.size.dx}X{self.size.dy},   {len(self.objects)} Primitives'

    def __copy__(self):
        new_canvas = Canvas(size=self.size, _id=None)
        for o in self.objects:
            new_canvas.add_new_object(copy(o))
        new_canvas.full_canvas = copy(self.full_canvas)
        new_canvas.actual_pixels = copy(self.actual_pixels)
        new_canvas.id = self.id
        try:
            new_canvas.grid = copy(self.grid)
            new_canvas.size_of_tiles = copy(self.size_of_tiles)
            new_canvas.grid_shape = copy(self.grid_shape)
            new_canvas.grid_colour = copy(self.grid_colour)
            new_canvas.grid_tiles_coordinates = copy(self.grid_tiles_coordinates)
            self.make_grid_background()
        except:
            pass
        return new_canvas

    @staticmethod
    def actual_pixels_index_to_canvas_coordinates(actual_pixels_index: Tuple[int, int]) -> Point:
        return Point(actual_pixels_index[1], actual_pixels_index[0])

    def resize_canvas(self, new_size: Dimension2D):
        """
        Resize the Canvas. This will remove any actual_pixels info that is not in the canvas' object.
        :param new_size: The new size
        :return:
        """
        self.size = Dimension2D(int(new_size.dx), int(new_size.dy))
        self.actual_pixels = np.ones((self.size.dy, self.size.dx))
        self.full_canvas[0: self.size.dy, 0:self.size.dx] = self.actual_pixels
        self.background_pixels = np.ndarray.copy(self.actual_pixels)
        self.embed_objects()

    def sort_objects_by_size(self, used_dim: str = 'area') -> List[Primitive]:
        """
        Returns a list of all the Object on Canvas sorted from largest to smallest according to the dimension used
        :param used_dim: The dimension to use to sort the Objects. It can be 'area', 'height', 'width', 'coloured_pixels'
        :return:
        """
        sorted_objects = np.array(copy(self.objects))
        dim = []
        for o in sorted_objects:
            if used_dim == 'area':
                metric = o.dimensions.dx * o.dimensions.dy
            elif used_dim == 'height':
                metric = o.dimensions.dy
            elif used_dim == 'length':
                metric = o.dimensions.dx
            elif used_dim == 'coloured_pixels':
                metric = len(o.get_coloured_pixels_positions())
            dim.append(metric)
        dim = np.array(dim)
        sorted_indices = dim.argsort()
        sorted_objects = sorted_objects[sorted_indices]

        return sorted_objects

    def find_objects_of_colour(self, colour: int):
        result = []
        for obj in self.objects:
            if obj.colour == colour:
                result.append(obj)

        return result

    def find_object_at_canvas_pos(self, cp: Point) -> Primitive | None:
        for obj in self.objects:
            if cp == obj.canvas_pos:
                return obj

        return None

    def get_coloured_pixels_positions(self) -> np.ndarray:
        """
        Returns the Union of the positions of the coloured pixels of all the object in the self.object list
        :return: np.ndarray of the union of all the coloured pixels of all object
        """
        result = self.objects[0].get_coloured_pixels_positions()
        for obj in self.objects[1:]:
            result = union2d(result, obj.get_coloured_pixels_positions())

        return result

    def where_object_fits_on_canvas(self, obj: Primitive,
                                    allowed_canvas_limits: Surround_Percentage | Surround =
                                    Surround_Percentage(Up=0.25, Down=0.25, Left=0.25, Right=0.25),
                                    excluded_points: List[Point] = ()) -> List[Point]:
        """
        Finds all the points on the Canvas that an Object can be placed (Object.canvas_pos) so that it is at least
        2/3 within the Canvas and that it is over and under other Objects on the Canvas by their required_dist_to_others
        :param excluded_points: A list of Points where the Object is not allowed to have its canvas_pos.
        :param allowed_canvas_limits: The limits out of which an object cannot go. If it is a Surround then the values are in canvas pixels. If it is Surround_Percentage then the values are floats denoting the percentage of the object's size that can be outside the canvas.
        :param obj: The Object to check
        :return:
        """
        available_canvas_points = []
        if np.any((self.size - obj.dimensions).to_numpy() < [0, 0]):
            return available_canvas_points

        x_range = [allowed_canvas_limits.Left,
                   allowed_canvas_limits.Right] \
            if isinstance(allowed_canvas_limits, Surround) \
            else [-int(obj.dimensions.dx * allowed_canvas_limits.Left),
                  int(self.size.dx - (1-allowed_canvas_limits.Right) * obj.dimensions.dx)]

        y_range = [allowed_canvas_limits.Down,
                   allowed_canvas_limits.Up] \
            if isinstance(allowed_canvas_limits, Surround) \
            else [-int(obj.dimensions.dx * allowed_canvas_limits.Down),
                  int(self.size.dx - (1 - allowed_canvas_limits.Up) * obj.dimensions.dy)]

        for x in range(x_range[0], x_range[1]):
            for y in range(y_range[0], y_range[1]):
                if Point(x, y, 0) not in excluded_points:
                    obj.canvas_pos = Point(x, y, 0)
                    overlap = False
                    for obj_b in self.objects:
                        if do_two_objects_overlap(obj, obj_b):
                            overlap = True
                            #print(x, y)
                            #print(obj.dimensions, obj.required_dist_to_others, obj.canvas_pos)
                            #print(obj_b.dimensions, obj_b.required_dist_to_others, obj_b.canvas_pos)
                    if not overlap:
                        available_canvas_points.append(Point(x, y, 0))
        return available_canvas_points

    def embed_objects(self):
        """
        Embeds all object in the self.objects list onto the self.actual_pixels of the canvas. It uses the object
        canvas_pos.z to define the order (object with smaller z go first thus end up behind object with larger z)
        :return:
        """
        if self.grid:
            self.make_grid_background()

        self.actual_pixels = np.ndarray.copy(self.background_pixels)

        self.objects = sorted(self.objects, key=lambda obj: obj.canvas_pos.z)

        for i, obj in enumerate(self.objects):
            obj.canvas_pos = Point(obj.canvas_pos.x, obj.canvas_pos.y, i)

        for i, obj in enumerate(self.objects):
            xmin = 0
            xmin_canv = obj.canvas_pos.x
            if xmin_canv >= self.actual_pixels.shape[1]:
                continue
            if xmin_canv < 0:
                xmin = np.abs(xmin_canv)
                xmin_canv = 0

            xmax = obj.dimensions.dx
            xmax_canv = obj.canvas_pos.x + obj.dimensions.dx
            if xmax_canv >= self.actual_pixels.shape[1]:
                xmax -= xmax_canv - self.actual_pixels.shape[1]
                xmax_canv = self.actual_pixels.shape[1]

            ymin = 0
            ymin_canv = obj.canvas_pos.y
            if ymin_canv >= self.actual_pixels.shape[0]:
                continue
            if ymin_canv < 0:
                ymin = np.abs(ymin_canv)
                ymin_canv = 0

            ymax = obj.dimensions.dy
            ymax_canv = obj.canvas_pos.y + obj.dimensions.dy
            if ymax_canv >= self.actual_pixels.shape[0]:
                ymax -= ymax_canv - self.actual_pixels.shape[0]
                ymax_canv = self.actual_pixels.shape[0]

            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)

            ymin_canv = int(ymin_canv)
            ymax_canv = int(ymax_canv)
            xmin_canv = int(xmin_canv)
            xmax_canv = int(xmax_canv)

            # The following will sum to the canvas only the object's pixels that are not 1
            bbox_to_embed = copy(obj.actual_pixels[ymin:ymax, xmin:xmax])
            bbox_to_embed_in = copy(self.actual_pixels[ymin_canv: ymax_canv, xmin_canv: xmax_canv])
            bbox_to_embed_in[np.where(bbox_to_embed > 1)] = bbox_to_embed[np.where(bbox_to_embed > 1)]
            self.actual_pixels[ymin_canv: ymax_canv, xmin_canv: xmax_canv] = bbox_to_embed_in
        self.full_canvas[0: self.size.dy, 0:self.size.dx] = self.actual_pixels

    def make_grid_background(self):
        colour = self.grid_colour
        size_of_tiles = self.size_of_tiles

        actual_pixels = np.ones((self.size.dy, self.size.dx))
        for y in range(size_of_tiles, actual_pixels.shape[0], size_of_tiles + 1):
            actual_pixels[y, :] = colour
        for x in range(size_of_tiles, actual_pixels.shape[1], size_of_tiles + 1):
            actual_pixels[:, x] = colour

        for j, y in enumerate(range(0, actual_pixels.shape[0], size_of_tiles + 1)):
            for i, x in enumerate(range(0, actual_pixels.shape[1], size_of_tiles + 1)):
                self.grid_tiles_coordinates[(i, j)] = Point(x, y)
        self.background_pixels = actual_pixels

    def add_new_object(self, obj: Primitive):
        if obj.id == None:
            if len(self.objects) == 0:
                obj.id = 0
            else:
                obj.id = np.max([o.id for o in self.objects]) + 1
        obj.canvas_id = self.id
        self.objects.append(obj)
        obj.canvas_id = self.id
        self.embed_objects()

    def remove_object(self, obj: Primitive):
        self.objects.remove(obj)
        self.embed_objects()

    @staticmethod
    def and_objects(obj_a: Primitive, obj_b: Primitive, result_colour: Colour | int,
                    canvas_pos: Point = Point(0, 0)) -> Predefined:
        assert obj_a.size == obj_b.size
        result_colour = result_colour.index if isinstance(result_colour, Colour) else result_colour

        pa = copy(obj_a.actual_pixels)
        pa[pa == 1] = -1
        pb = copy(obj_b.actual_pixels)
        pb[pb == 1] = -2

        new_actual_pixels = np.ones((obj_a.dimensions.dx, obj_b.dimensions.dy))
        new_actual_pixels[pa == pb] = result_colour

        result = Predefined(actual_pixels=new_actual_pixels)
        result.canvas_pos = canvas_pos

        return result

    def or_objects(self, obj_a: Primitive, obj_b: Primitive, result_colour: Colour | int,
                    canvas_pos: Point = Point(0, 0)) -> Predefined:
        pass

    def clear(self):
        for o in self.objects:
            self.objects.remove(o)
        self.embed_objects()

    def split_object_by_colour(self, obj: Object) -> Dict:

        resulting_ids = {'id': [], 'actual_pixels_id': [], 'index': []}
        object_id = np.max([obj.id for obj in self.objects])
        actual_pixels_id = np.max([obj.actual_pixels_id for obj in self.objects])

        colours_in_places = obj.get_colour_groups()
        for col in colours_in_places:
            object_id += 1
            actual_pixels_id += 1
            canvas_pos = Point(colours_in_places[col][:, 1].min(), colours_in_places[col][:, 0].min())
            new_pixels_index = colours_in_places[col] - np.array([colours_in_places[col][:, 0].min(), colours_in_places[col][:, 1].min()])
            actual_pixels_index = colours_in_places[col] - np.array([obj.canvas_pos.y, obj.canvas_pos.x])

            size = Dimension2D(colours_in_places[col][:, 1].max() - colours_in_places[col][:, 1].min() + 1,
                               colours_in_places[col][:, 0].max() - colours_in_places[col][:, 0].min() + 1)

            if colours_in_places[col].shape[0] == 1:
                new_primitive = Dot(colour=col, canvas_pos=canvas_pos,
                                    _id=object_id, actual_pixels_id=actual_pixels_id)
            else:
                new_pixels = np.ones((size.dy, size.dx))
                new_pixels[new_pixels_index[:, 0], new_pixels_index[:, 1]] = \
                    obj.actual_pixels[actual_pixels_index[:, 0], actual_pixels_index[:, 1]]

                new_primitive = Random(size=Dimension2D(new_pixels.shape[1], new_pixels.shape[0]), canvas_pos=canvas_pos,
                                       _id=object_id, actual_pixels_id=actual_pixels_id)
                new_primitive.set_colour_to_most_common()
                new_primitive.actual_pixels[:, :] = new_pixels[:, :]
            resulting_ids['id'].append(object_id)
            resulting_ids['actual_pixels_id'].append(actual_pixels_id)
            self.add_new_object(new_primitive)
            resulting_ids['index'].append(len(self.objects) - 1)

        self.remove_object(obj)
        resulting_ids['index'] = np.array(resulting_ids['index']) - 1

        return resulting_ids

    def create_background_from_object(self, obj: Object):
        xmin = int(obj.canvas_pos.x)
        if xmin >= self.actual_pixels.shape[1]:
            return
        if xmin < 0:
            xmin = 0
        xmax = int(obj.canvas_pos.x + obj.dimensions.dx)
        if xmax >= self.actual_pixels.shape[1]:
            xmax = self.actual_pixels.shape[1]
        ymin = int(obj.canvas_pos.y)
        if ymin >= self.actual_pixels.shape[0]:
            return
        if ymin < 0:
            ymin = 0
        ymax = int(obj.canvas_pos.y + obj.dimensions.dy)
        if ymax >= self.actual_pixels.shape[0]:
            ymax = self.actual_pixels.shape[0]

        self.background_pixels[ymin: ymax, xmin: xmax] = obj.actual_pixels[: ymax - ymin, : xmax - xmin]
        self.embed_objects()

    def position_object(self, index: int, canvas_pos: Point):
        """
        Positions the object (with id = index) to the canvas_pos specified (the bottom left pixel of the object is
        placed to that canvas_pos)
        :param index: The id of the object
        :param canvas_pos: The Point specifying the coordinates on the canvas of the bottom left pixel of the object
        :return:
        """
        self.objects[index].canvas_pos = canvas_pos
        self.embed_objects()

    def get_object_by_id(self, id: int) -> Primitive | None:
        for o in self.objects:
            if o.id == id:
                return o
        return None

    def add_relational_features_to_canvas_objects(self):
        MAX_NUM_OF_CANVAS_OBJECTS = 10
        for obj in self.objects:
            other_objects = copy(self.objects)
            other_objects.remove(obj)

            #features = obj.get_features()
            features = {}

            features['Touched Objects'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for i, oo in enumerate(other_objects):
                if obj.is_object_touching(oo):
                    features['Touched Objects'][j] = oo.id
                    j += 1

            features['Overlaped Objects'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for i, oo in enumerate(other_objects):
                if obj.is_object_overlapped(oo):
                    features['Overlaped Objects'][j] = oo.id
                    j += 1

            features['Underlaped Objects'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for i, oo in enumerate(other_objects):
                if obj.is_object_underlapped(oo):
                    features['Underlaped Objects'][j] = oo.id
                    j += 1

            features['Matched Objects'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_matching_to_object(oo):
                    features['Matched Objects'][j] = oo.id
                    j += 1

            features['Matched Objects Only By Shape'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_matching_to_object(oo, match_shape_only=True):
                    features['Matched Objects Only By Shape'][j] = oo.id
                    j += 1

            features['Matched Objects if Rotated'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_matching_to_object(oo, transformations=['rotate']):
                    features['Matched Objects if Rotated'][j] = oo.id
                    j += 1

            features['Matched Objects if Rotated Only By Shape'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_matching_to_object(oo, match_shape_only=True, transformations=['rotate']):
                    features['Matched Objects if Rotated Only By Shape'][j] = oo.id
                    j += 1

            features['Matched Objects if Scaled'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_matching_to_object(oo, transformations=['scale']):
                    features['Matched Objects if Scaled'][j] = oo.id
                    j += 1

            features['Matched Objects if Scaled Only By Shape'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_matching_to_object(oo, match_shape_only=True, transformations=['scale']):
                    features['Matched Objects if Scaled Only By Shape'][j] = oo.id
                    j += 1

            features['Matched Objects if Flipped'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_matching_to_object(oo, transformations=['flip']):
                    features['Matched Objects if Flipped'][j] = oo.id
                    j += 1

            features['Matched Objects if Flipped Only By Shape'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_matching_to_object(oo, match_shape_only=True, transformations=['flip']):
                    features['Matched Objects if Flipped Only By Shape'][j] = oo.id
                    j += 1

            features['Matched Objects if Inverted'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_matching_to_object(oo, transformations=['invert']):
                    features['Matched Objects if Inverted'][j] = oo.id
                    j += 1

            features['Matched Objects if Inverted Only By Shape'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_matching_to_object(oo, match_shape_only=True, transformations=['invert']):
                    features['Matched Objects if Inverted Only By Shape'][j] = oo.id
                    j += 1

            features['Objects Along X'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_along_x_to_object(oo):
                    features['Objects Along X'][j] = oo.id
                    j += 1

            features['Objects Along Y'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_along_y_to_object(oo):
                    features['Objects Along Y'][j] = oo.id
                    j += 1

            features['Objects Along XY'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_along_xy_to_object(oo):
                    features['Objects Along XY'][j] = oo.id
                    j += 1

            features['Objects Along XminusY'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_along_xminusy_to_object(oo):
                    features['Objects Along XminusY'][j] = oo.id
                    j += 1

            features['Objects Over'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_under_object(oo):
                    features['Objects Over'][j] = oo.id
                    j += 1

            features['Objects Under'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_over_object(oo):
                    features['Objects Under'][j] = oo.id
                    j += 1

            features['Objects to the Left'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_right_of_object(oo):
                    features['Objects to the Left'][j] = oo.id
                    j += 1

            features['Objects to the Right'] = np.zeros(MAX_NUM_OF_CANVAS_OBJECTS) - 1
            j = 0
            for oo in other_objects:
                if obj.is_object_left_of_object(oo):
                    features['Objects to the Right'][j] = oo.id
                    j += 1

            num_of_features = {}
            for f in features:
                size = len(np.where(features[f] > -1)[0])
                num_of_features[f'Num of {f}'] = size

            features = features | num_of_features
            features = features | obj.get_features()

            self.objects_features[obj.id] = features
            self.transform_object_features_to_matrix_form()

    def transform_object_features_to_matrix_form(self):
        binary_features_names = ['Touched Objects', 'Overlaped Objects', 'Underlaped Objects', 'Matched Objects',
                                 'Matched Objects Only By Shape', 'Matched Objects if Rotated',
                                 'Matched Objects if Rotated Only By Shape', 'Matched Objects if Scaled',
                                 'Matched Objects if Scaled Only By Shape', 'Matched Objects if Flipped',
                                 'Matched Objects if Flipped Only By Shape', 'Matched Objects if Inverted',
                                 'Matched Objects if Inverted Only By Shape', 'Objects Along X', 'Objects Along Y',
                                 'Objects Along XY', 'Objects Along XminusY', 'Objects Over',  'Objects Under',
                                 'Objects to the Left', 'Objects to the Right']

        binary_features_initial_matrices = [np.zeros((MAX_NUM_OF_CANVAS_OBJECTS, MAX_NUM_OF_CANVAS_OBJECTS))
                                            for i in range(len(binary_features_names))]

        for i, o in enumerate(self.objects):
            self.obj_id_to_index_hashmap[o.id] = i

        binary_features = {k: v for (k, v) in zip(binary_features_names, binary_features_initial_matrices)}
        for obj_id in self.objects_features:
            index = self.obj_id_to_index_hashmap[obj_id]
            for feature_name in self.objects_features[obj_id]:
                feature = self.objects_features[obj_id][feature_name]
                if feature_name in binary_features_names:
                    feature_indices = [self.obj_id_to_index_hashmap[f] if f > -1 else -1 for f in feature]
                    for i in feature_indices:
                        binary_features[feature_name][index, i] = 1 if i > -1 else 0

        self.binary_features = binary_features

    def json_output(self, with_pixels: bool = False) -> dict:
        result = {'object': []}
        if with_pixels:
            result['actual_pixels'] = self.actual_pixels.tolist()
            result['full_canvas'] = self.full_canvas.tolist()
        for o in self.objects:
            o_json = o.json_output()
            o_json.pop('id', None)
            o_json.pop('actual_pixels_id', None)
            o_json['actual_pixels'] = o.actual_pixels.tolist()
            result['object'].append(o_json)

        return result

    def get_used_colours(self) -> List[int]:
        colours = set(np.unique(
            self.actual_pixels))
        colours -= {0, 1}
        if self.grid:
            colours -= {self.grid_colour}
        return list(np.array(list(colours)).astype(int))

    def swap_colours(self, colour_swap_map: dict[int, int]):
        temp_pixels = copy(self.full_canvas)
        for colour in colour_swap_map:
            temp_pixels[np.where(self.full_canvas == colour)] = colour_swap_map[colour]
        self.full_canvas = copy(temp_pixels)
        self.actual_pixels = self.full_canvas[:self.actual_pixels.shape[0], :self.actual_pixels.shape[1]]

        for obj in self.objects:
            obj.replace_all_colours(colours_hash=colour_swap_map)

    def show(self, full_canvas=True, fig_to_add: None | plt.Figure = None, nrows: int = 0, ncoloumns: int = 0,
             index: int = 1, save_as: str | None = None, thin_lines: bool = False):

        if full_canvas:
            xmin = - 0.5
            xmax = self.full_canvas.shape[1] - 0.5
            ymin = - 0.5
            ymax = self.full_canvas.shape[0] - 0.5
            extent = [xmin, xmax, ymin, ymax]
            if fig_to_add is None:
                fig, _ = vis.plot_data(self.full_canvas, extent=extent, thin_lines=thin_lines)
            else:
                ax = fig_to_add.add_subplot(nrows, ncoloumns, index)
                _ = vis.plot_data(self.full_canvas, extent=extent, axis=ax, thin_lines=thin_lines)
        else:
            xmin = - 0.5
            xmax = self.actual_pixels.shape[1] - 0.5
            ymin = - 0.5
            ymax = self.actual_pixels.shape[0] - 0.5
            extent = [xmin, xmax, ymin, ymax]
            if fig_to_add is None:
                fig, _ = vis.plot_data(self.actual_pixels, extent=extent, thin_lines=thin_lines)
            else:
                ax = fig_to_add.add_subplot(nrows, ncoloumns, index)
                _ = vis.plot_data(self.actual_pixels, extent=extent, axis=ax, thin_lines=thin_lines)

        if fig_to_add is None and save_as is not None:
            fig.savefig(save_as)
            plt.close(fig)

