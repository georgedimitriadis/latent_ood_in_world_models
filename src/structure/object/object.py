
from __future__ import annotations

from typing import List, Union, Dict, Set
from copy import copy, deepcopy
from typing import Tuple
import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import skimage

from structure.object.transformation import ObjectTransformations
from visualization import basic_visualisation_of_data as vis
from structure.geometry.basic_geometry import Point, Vector, Orientation, Surround, Bbox, \
    Dimension2D, RelativePoint, Colour

from experiments.data.generation import constants as const

MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Object:

    def __init__(self, actual_pixels: np.ndarray, _id: None | int = None,
                 actual_pixels_id: int | None = None, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: List | np.ndarray | Point = (0, 0, 0), canvas_id: int | None = None):

        self.id = _id
        self.actual_pixels_id = actual_pixels_id
        self.canvas_id = canvas_id
        self.border_size = border_size

        self._actual_pixels = actual_pixels
        self._canvas_pos = Point.point_from_numpy(np.array(canvas_pos)) if type(canvas_pos) != Point else canvas_pos
        self._holes = None
        self._relative_points = {}
        self._perimeter = {}
        self._inside = {}
        self._visible_bbox = None
        self._dimensions = Dimension2D(self.actual_pixels.shape[1], self.actual_pixels.shape[0])

        self.rotation_axis = deepcopy(self._canvas_pos)
        self.symmetries: List = []
        self.transformations: List = []

        self._reset_dimensions()

    @property
    def canvas_pos(self):
        """
        The getter of the canvas_pos
        :return: the canvas_pos
        """
        return self._canvas_pos

    @canvas_pos.setter
    def canvas_pos(self, new_pos: Point):
        """
        The setter of the canvas_pos
        :param new_pos: The new canvas_pos
        :return:
        """
        move = new_pos - self._canvas_pos
        self._canvas_pos = new_pos
        for sym in self.symmetries:
            sym.origin += move
        self._reset_dimensions()

    @property
    def number_of_coloured_pixels(self):
        """
        The getter of the number_of_coloured_pixels
        :return: the number_of_coloured_pixels
        """
        return int(np.sum([1 for i in self.actual_pixels for k in i if k > 1]))

    @property
    def holes(self):
        """
        The getter of the holes of the Object.
        :return: the holes
        """
        holes, n, _ = self.detect_holes(self.actual_pixels)
        if n == 0:
            return None
        self._holes = holes
        return self._holes

    @property
    def relative_points(self):
        """
        Generates the relative_points for the Object, that is the coordinates of the 9 RelativePoints.
        :return: the relative_points
        """
        self._relative_points[RelativePoint.Bottom_Left] = self.canvas_pos
        self._relative_points[RelativePoint.Bottom_Right] = Point(self.canvas_pos.x + self.dimensions.dx,
                                                                  self.canvas_pos.y)
        self._relative_points[RelativePoint.Top_Left] = self.bbox.top_left
        self._relative_points[RelativePoint.Top_Right] = Point(self.bbox.bottom_right.x, self.bbox.top_left.y)

        self._relative_points[RelativePoint.Top_Center] = Point(self.canvas_pos.x + self.dimensions.dx / 2 - 0.5,
                                                                self.bbox.top_left.y)
        self._relative_points[RelativePoint.Bottom_Center] = Point(self.canvas_pos.x + self.dimensions.dx / 2 - 0.5,
                                                                   self.canvas_pos.y)
        self._relative_points[RelativePoint.Middle_Left] = Point(self.canvas_pos.x,
                                                                 self.canvas_pos.y + self.dimensions.dy / 2 - 0.5)
        self._relative_points[RelativePoint.Middle_Right] = Point(self.bbox.bottom_right.x,
                                                                  self.canvas_pos.y + self.dimensions.dy / 2 - 0.5)
        self._relative_points[RelativePoint.Middle_Center] = Point(self.canvas_pos.x + self.dimensions.dx / 2 - 0.5,
                                                                   self.canvas_pos.y + self.dimensions.dy / 2 - 0.5)

        return self._relative_points

    @property
    def actual_pixels(self) -> np.ndarray:
        """
        The getter of the actual_pixels
        :return: he actual_pixels
        """
        return self._actual_pixels

    @actual_pixels.setter
    def actual_pixels(self, new_pixels: np.ndarray):
        """
        The setter of the actual_pixels
        :param new_pixels: The new actual_pixels
        :return:
        """
        self._actual_pixels = new_pixels
        self._reset_dimensions()

    @property
    def dimensions(self) -> Dimension2D:
        """
        Getter of the dimensions property
        :return: The dimensions property
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, new_dim: Dimension2D):
        """
        Setter of the dimensions property
        :param new_dim:
        :return:
        """
        self._dimensions = new_dim

    @property
    def visible_bbox(self) -> Bbox:
        """
        This returns the bbox that encompasses only the visible pixels of the object.
        :return: The visible bbox
        """
        self._reset_dimensions()
        return self._visible_bbox

    @property
    def perimeter(self) -> Dict[str, Set]:
        """
        Creates a dictionary with keys 'Left', 'Right', 'Top' and 'Bottom'. The value of each key is a set of k 2-element
        Lists. Each 2-element List is a position (on the Canvas - (x,y)) of the pixels that have a colour and would be
        accessible from the key side of the Object. This generates a dictionary holding the positions of the surround
        (perimeter) pixels of the Object.
        :return: The perimeter dictionary.
        """
        pixs = self.actual_pixels

        result = {'Left': [], 'Right': [], 'Top': [], 'Bottom': []}
        pos_indices = []
        for left_to_right in range(self.dimensions.dx):
            pos = np.argwhere(pixs[:, left_to_right] > 1).squeeze()
            if len(pos.shape) == 0:
                pos = np.array([pos])
            good_pos = [p for p in pos if p not in pos_indices]
            for gp in good_pos:
                result['Left'].append((left_to_right + self.canvas_pos.x, gp + self.canvas_pos.y))
                pos_indices.append(gp)

        pos_indices = []
        for right_to_left in range(self.dimensions.dx):
            pos = np.argwhere(pixs[:, self.dimensions.dx - right_to_left - 1] > 1).squeeze()
            if len(pos.shape) == 0:
                pos = np.array([pos])
            good_pos = [p for p in pos if p not in pos_indices]
            for gp in good_pos:
                result['Right'].append((self.dimensions.dx - right_to_left - 1 + self.canvas_pos.x, gp + self.canvas_pos.y))
                pos_indices.append(gp)

        pos_indices = []
        for top_to_bottom in range(self.dimensions.dy):
            pos = np.argwhere(pixs[top_to_bottom, :] > 1).squeeze()
            if len(pos.shape) == 0:
                pos = np.array([pos])
            good_pos = [p for p in pos if p not in pos_indices]
            for gp in good_pos:
                result['Bottom'].append((gp + self.canvas_pos.x, top_to_bottom + self.canvas_pos.y))
                pos_indices.append(gp)

        pos_indices = []
        for bottom_to_top in range(self.dimensions.dy):
            pos = np.argwhere(pixs[self.dimensions.dy - bottom_to_top - 1, :] > 1).squeeze()
            if len(pos.shape) == 0:
                pos = np.array([pos])
            good_pos = [p for p in pos if p not in pos_indices]
            for gp in good_pos:
                result['Top'].append((gp + self.canvas_pos.x, self.dimensions.dy - bottom_to_top - 1 + self.canvas_pos.y))
                pos_indices.append(gp)

        for dir in result:
            temp = result[dir]
            set_temp = set(tuple(i) for i in temp)
            result[dir] = set_temp

        self._perimeter = result
        return result

    @property
    def inside(self) -> Set:
        """
        Returns a set of the positions (x, y) of all the visible pixels of the Object that are not part of the perimeter.
        :return: The inside pixels set
        """
        all_coloured_pixels = set((i[1], i[0]) for i in self.get_coloured_pixels_positions())
        result = all_coloured_pixels - self.perimeter['Left'] - self.perimeter['Right'] - self.perimeter['Top'] - \
                    self.perimeter['Bottom']
        return result

    @staticmethod
    def _match_filter_obj_to_background_obj(background_obj: Object, filter_obj: Object,
                                            padding: Surround | None = None, try_unique: bool = True,
                                            match_shape_only: bool = False) -> Tuple[List[Point], np.ndarray]:
        """
        The basic match_to_background between a background Object and a filter Object. It calculates the list of coordinates
        (Points) that the filter Object should translate to so that it gives the best match_to_background with background Object.
        This is a list if multiple positions result in the same match_to_background quality. It also returns the match_to_background quality.
        :param background_obj: The object to be matched
        :param filter_obj: The object to match_to_background
        :param padding: Any padding (Surround) that would help with the matching. If (0, 0, 0, 0) doesn't work try (1, 1, 1, 1)
        :param match_shape_only: If True then ignore colours and match_to_background only for the shapes the visible pixels generates.
        :return: The best position(s) the filter should move to in order to match_to_background background, The quality of the match_to_background.
        """

        if padding is None:
            padding = Surround(0, 0, 0, 0)

        background_obj_size = Dimension2D(background_obj.dimensions.dx + padding.Left + padding.Right,
                                          background_obj.dimensions.dy + padding.Up + padding.Down)

        filter_obj_size = Dimension2D(filter_obj.dimensions.dx + padding.Left + padding.Right,
                                      filter_obj.dimensions.dy + padding.Up + padding.Down)

        filter_obj_pixels = np.ones((filter_obj_size.dy, filter_obj_size.dx))
        filter_obj_pixels[padding.Down: filter_obj_size.dy - padding.Up,
                          padding.Left: filter_obj_size.dx - padding.Right] = copy(filter_obj.actual_pixels)

        # The following does a 2D convolution of the filter Object over the background Object assuming 0 values for the
        # background's padding (equal to the size of the filter plus the filter's padding).

        x_base_dim = background_obj_size.dx + 2 * filter_obj_size.dx - 2
        y_base_dim = background_obj_size.dy + 2 * filter_obj_size.dy - 2
        background = np.zeros((y_base_dim, x_base_dim))

        x_res_dim = background_obj_size.dx + filter_obj_size.dx - 1
        y_res_dim = background_obj_size.dy + filter_obj_size.dy - 1
        fit_value = np.zeros((y_res_dim, x_res_dim))

        background_pixels = np.zeros((background_obj_size.dy, background_obj_size.dx))
        background_pixels[padding.Down: background_obj_size.dy - padding.Up,
                          padding.Left: background_obj_size.dx - padding.Right] = background_obj.actual_pixels
        background[filter_obj_size.dy - 1: filter_obj_size.dy - 1 + background_obj_size.dy,
                   filter_obj_size.dx - 1: filter_obj_size.dx - 1 + background_obj_size.dx] = background_pixels

        for x in range(x_res_dim):
            for y in range(y_res_dim):
                part_of_background = copy(background[y: filter_obj_size.dy + y, x: filter_obj_size.dx + x])
                '''
                filter_obj_pixels = np.ones((filter_obj_size.dy, filter_obj_size.dx))
                filter_obj_pixels[padding.Down: filter_obj_size.dy - padding.Up,
                                  padding.Left: filter_obj_size.dx - padding.Right] = copy(filter_obj.actual_pixels)
                '''
                if match_shape_only:
                    filter_obj_pixels_cp = copy(filter_obj_pixels)
                    part_of_background[np.where(part_of_background == 1)] = 0
                    part_of_background[np.where(part_of_background > 1)] = 1
                    filter_obj_pixels_cp[np.where(filter_obj_pixels_cp == 1)] = 0
                    filter_obj_pixels_cp[np.where(filter_obj_pixels_cp > 1)] = 1
                    comp = (part_of_background == filter_obj_pixels_cp).astype(int)
                else:
                    comp = np.zeros(part_of_background.shape)
                    for i in range(part_of_background.shape[0]):
                        for j in range(part_of_background.shape[1]):
                            if filter_obj_pixels[i, j] != 1 and part_of_background[i, j] != 1:
                                if part_of_background[i, j] == filter_obj_pixels[i, j]:
                                    comp[i, j] = 1
                                else:
                                    comp[i, j] = -1
                            elif filter_obj_pixels[i, j] == 1 or part_of_background[i, j] == 1:
                                comp[i, j] = 0
                fit_value[y, x] = comp.sum() / len(filter_obj.get_coloured_pixels_positions()) \
                    if len(filter_obj.get_coloured_pixels_positions()) > 0 else comp.sum()

        best_relative_positions = np.argwhere(fit_value == np.amax(fit_value))

        # This uses a center surround kind of algorithm to distinguish between multiple positives
        if len(best_relative_positions > 1) and try_unique:
            center_surround = []
            for brp in best_relative_positions:
                x = brp[1]
                y = brp[0]
                s = 0
                surround = 0
                if y > 0:
                    surround += fit_value[y - 1, x]
                    s += 1
                if y < y_res_dim - 1:
                    surround += fit_value[y + 1, x]
                    s += 1
                if x > 0:
                    surround += fit_value[y, x - 1]
                    s += 1
                    if y > 0:
                        surround += fit_value[y - 1, x - 1]
                        s += 1
                    if y < y_res_dim - 1:
                        surround += fit_value[y + 1, x - 1]
                        s += 1
                if x < x_res_dim - 1:
                    surround += fit_value[y, x + 1]
                    s += 1
                    if y > 0:
                        surround += fit_value[y - 1, x + 1]
                        s += 1
                    if y < y_res_dim - 1:
                        surround += fit_value[y + 1, x + 1]
                        s += 1
                center_surround.append(fit_value[y, x] - surround / s)

            better_index = np.where(center_surround == np.amin(center_surround))
            best_relative_positions = best_relative_positions[better_index]

        best_positions = [Point(x=brp[1] - filter_obj_size.dx + 1 + background_obj.canvas_pos.x,
                                y=brp[0] - filter_obj_size.dy + 1 + background_obj.canvas_pos.y)
                          for brp in best_relative_positions]

        return best_positions, np.amax(fit_value)

    def __copy__(self):
        """
        Copys the whole Object
        :return:
        """
        new_obj = Object(actual_pixels=self.actual_pixels, _id=self.id, border_size=self.border_size,
                         canvas_pos=self.canvas_pos)
        new_obj.dimensions = copy(self.dimensions)
        new_obj.border_size = copy(self.border_size)
        new_obj.bbox = copy(self.bbox)
        new_obj.rotation_axis = copy(self.rotation_axis)
        for sym in self.symmetries:
            new_obj.symmetries.append(copy(sym))
        return new_obj

    '''
    def __add__(self, other: Object) -> Object:
        """
        canvas = Canvas(size=Dimension2D(32, 32))
        canvas.add_new_object(self)
        canvas.add_new_object(other)
        w = np.where(canvas.full_canvas > 1)

        actual_pixels = canvas.full_canvas[w[0].min(): w[0].max()+1, w[1].min():w[1].max()+1]
        
        Superimposes the other Object on top of the self Object. The black pixels are treated as transparent.
        The superposition ignores the canvas_pos of the two objects. The final Object has the canvas_pos of the self
        object.
        :param other: The other Object
        :return: The result of the superposition
        """
        #result = copy(other)
        
        actual_pixels = np.ones((np.max([self.dimensions.dy, other.dimensions.dy]),
                                 np.max([self.dimensions.dx, other.dimensions.dx])))


        bottom = self if self.canvas_pos.z < other.canvas_pos.z else other
        top = other if self.canvas_pos.z < other.canvas_pos.z else self

        actual_pixels[:bottom.dimensions.dy, :bottom.dimensions.dx] = bottom.actual_pixels
        actual_pixels[:top.dimensions.dy, :top.dimensions.dx] = top.actual_pixels

        for black_pos in top.get_background_pixels_positions():
            try:
                actual_pixels[black_pos[0], black_pos[1]] = bottom.actual_pixels[black_pos[0], black_pos[1]]
            except:
                pass

        result = Object(actual_pixels=actual_pixels)
        result.translate_to_coordinates(self.canvas_pos)

        #black_indices = np.argwhere(result.actual_pixels == 1)
        #result.actual_pixels[black_indices[:, 0], black_indices[:, 1]] = \
        #    self.actual_pixels[black_indices[:, 0], black_indices[:, 1]]

        return result
        '''
    def __sub__(self, other: Object) -> Object:
        """
        Subtracts the other Object from the self ignoring the colours of the two Objects.
        That means that all coloured pixel positions of the other will become black in the self.
        :param other: The Object to subtract
        :return: The new Object
        """
        result = copy(self)
        result.translate_to_coordinates(Point(0, 0, 0))
        other_copy = copy(other)
        other_copy.translate_to_coordinates(other.canvas_pos - self.canvas_pos)
        coloured_indices = np.argwhere(other_copy.actual_pixels != 1)
        result.actual_pixels[coloured_indices[:, 0]+other_copy.canvas_pos.y,
                             coloured_indices[:, 1]+other_copy.canvas_pos.x] = 1
        result.translate_to_coordinates(self.canvas_pos)

        return result

    def __mul__(self, multiplier: Dimension2D | int) -> Object:
        """
        Copies the self Object many times (multiplier) along the x and y axis (from left to right and from bottom to top).
        :param multiplier: How many times to copy the Object. If it is Dimension2D then the x and y can be different. If it is a single int then the x and y are the same.
        :return: The multiplied Object
        """
        x = multiplier.dx if isinstance(multiplier, Dimension2D) else multiplier
        y = multiplier.dy if isinstance(multiplier, Dimension2D) else multiplier
        result = copy(self)
        new_pixels = np.ones((y * self.actual_pixels.shape[0], x * self.actual_pixels.shape[1]))
        for i in range(y):
            for j in range(x):
                new_pixels[i * self.actual_pixels.shape[0]: (i + 1) * self.actual_pixels.shape[0],
                           j * self.actual_pixels.shape[1]: (j + 1) * self.actual_pixels.shape[1]] = self.actual_pixels
        result.actual_pixels = new_pixels

        return result

    # <editor-fold desc="TRANSFORMATION METHODS">
    def translate_to_coordinates(self, target_point: Point, object_point: Point | None = None):
        """
        Moves the Object's canvas_pos or object_point (if not None) to the target_point coordinates
        :param target_point: The target point to move the object point (or the canvas position)
        :param object_point: The point of the object to move to target point (becomes equal to canvas position if set to None).
        :return:
        """
        if object_point is None:
            object_point = self.canvas_pos

        self.transformations.append([ObjectTransformations.translate_to_coordinates.name,
                                     {'distance': (target_point - object_point).to_numpy().tolist()}])

        object_point = self.canvas_pos - object_point
        self.canvas_pos = target_point + object_point

    def translate_by(self, distance: Dimension2D):
        """
        Move the Object by distance.
        :param distance: The distance to move the Object by.
        :return:
        """
        self.canvas_pos += Point(distance.dx, distance.dy)

        self.transformations.append([ObjectTransformations.translate_by.name,
                                     {'distance': distance.to_numpy().tolist()}])

    def translate_along(self, direction: Vector):
        """
        Translate the Object along a given Vector
        :param direction: The Vector to translate along
        :return:
        """
        initial_canvas_pos = copy(self.canvas_pos)
        orient = direction.orientation
        if orient in [Orientation.Up, Orientation.Up_Left, Orientation.Up_Right]:
            self.canvas_pos.y += direction.length
        if orient in [Orientation.Down, Orientation.Down_Left, Orientation.Down_Right]:
            self.canvas_pos.y -= direction.length
        if orient in [Orientation.Left, Orientation.Up_Left, Orientation.Down_Left]:
            self.canvas_pos.x -= direction.length
        if orient in [Orientation.Right, Orientation.Up_Right, Orientation.Down_Right]:
            self.canvas_pos.x += direction.length

        self.transformations.append([ObjectTransformations.translate_along.name,
                                     {'distance': (self.canvas_pos - initial_canvas_pos).to_numpy().tolist()}])

        self._reset_dimensions()

    def translate_relative_point_to_point(self, relative_point: RelativePoint, other_point: Point):
        """
        Translate the Object so that its relative point with key RelativePoint ends up on other_point.
        :param relative_point: The RelativePoint
        :param other_point: The target point
        :return:
        """

        this_point = self.relative_points[relative_point]
        if this_point is not None:
            difference = other_point - this_point
            self.canvas_pos += difference

            self.transformations.append([ObjectTransformations.translate_relative_point_to_point.name,
                                         {'distance': difference.to_numpy().tolist()}])

        self._reset_dimensions()

    def translate_until_touch(self, other: Object):
        """
        Translates the Object on a straight line (along one of the 8 Orientations) in order to just touch the other
        Object.

        :param other: The other Object
        :return:
        """
        distance = self.get_straight_distance_to_object(other, up_to_corner=True)
        if distance is not None:
            self.translate_along(distance)

            self.transformations.append([ObjectTransformations.translate_until_touch.name,
                                         {'other': other.id}])

    def translate_until_fit(self, other: Object):
        """
        Translates the Object on a straight line (along one of the 8 Orientations) in order to touch the other Object in
        a way that makes the Objects fit as well as possible

        :param other: The other Object
        :return:
        """
        distance = self.get_straight_distance_to_object(other, up_to_corner=False)
        if distance is not None:
            self.translate_along(distance)

            self.transformations.append([ObjectTransformations.translate_until_fit.name,
                                         {'other': other.id}])

    def scale(self, factor: int):
        """
        Scales the object. A positive factor adds pixels and a negative factor removes pixels.
        :param factor: Integer
        :return: Nothing
        """

        if factor == 0:  # Factor cannot be 0 so in this case nothing happens
            return

        pic = self.actual_pixels

        if factor > 0:
            # If factor is > 0 it cannot blow up the object to more than MAX_PAD_SIZE
            if np.max(pic.shape) * factor > const.MAX_PAD_SIZE - 2:
                return
            scaled = np.ones(np.array(pic.shape) * factor)
            for x in range(pic.shape[0]):
                for y in range(pic.shape[1]):
                    scaled[x * factor:(x + 1) * factor, y * factor:(y + 1) * factor] = pic[x, y]
        else:
            # If factor is <0 it cannot shrink the object to something smaller than 2x2
            if np.abs(1/factor) * np.min(pic.shape) < 2:
                return
            scaled = np.ones(np.ceil(np.array(pic.shape) / np.abs(factor)).astype(np.int32))
            for x in range(scaled.shape[0]):
                for y in range(scaled.shape[1]):
                    scaled[x, y] = pic[x * np.abs(factor), y * np.abs(factor)]

        self.actual_pixels = scaled

        self._reset_dimensions()

        for sym in self.symmetries:
            if factor > 0:
                #sym.length *= 2 * factor - 0.5 * (factor + 1)
                sym.length = sym.length * factor + (factor - 1)

                d = np.abs(sym.origin - self.canvas_pos)
                multiplier = Point(factor, factor) if sym.orientation == Orientation.Up else Point(factor + 1, factor)
                sym.origin.x = d.x * multiplier.x + self.canvas_pos.x if d.x > 0 else sym.origin.x
                sym.origin.y = d.y * multiplier.y + self.canvas_pos.y if d.y > 0 else sym.origin.y

                sym.origin.x += 0.5 * (factor -1) if sym.orientation == Orientation.Up\
                    else -(factor -1) + (factor - 2) *2
                sym.origin.y += (factor - 1) * 0.5 if sym.orientation == Orientation.Left\
                    else (factor - 1) * 0.5 - (factor - 2) * 0.5 - 0.5
            else:
                sym.length = sym.length / np.abs(factor) - 1/np.abs(factor)
                factor = 1/np.abs(factor)
                sym.origin = (sym.origin - self.canvas_pos) * factor + self.canvas_pos

        self.transformations.append([ObjectTransformations.scale.name, {'factor': factor}])

    def rotate(self, times: Union[1, 2, 3], center: np.ndarray | List = (0, 0)):
        """
        Rotate the object counter-clockwise by times multiple of 90 degrees

        :param times: 1, 2 or 3 times
        :param center: The point of the axis of rotation
        :return:
        """
        radians = np.pi/2 * times
        degrees = -(90 * times)

        old_canvas_pos = copy(self.canvas_pos).to_numpy()

        '''
        if times == 1:
            self.canvas_pos.x = self.canvas_pos.x - self.dimensions.dy + 1
        elif times == 2:
            self.canvas_pos.x = self.canvas_pos.x - self.dimensions.dx + 1
            self.canvas_pos.y = self.canvas_pos.y - self.dimensions.dy + 1
        elif times == 3:
            self.canvas_pos.y = self.canvas_pos.y - self.dimensions.dx + 1
        '''
        self.actual_pixels = skimage.transform.rotate(self.actual_pixels, degrees, resize=True, order=0, center=[0, 0])

        for sym in self.symmetries:
            sym.transform(translation=-old_canvas_pos)
            sym.transform(rotation=radians)
            sym.transform(translation=old_canvas_pos)

        self.transformations.append([ObjectTransformations.rotate.name, {'times': times}])

    def shear(self, _shear: np.ndarray | List):
        """
        Shears the actual pixels

        :param _shear: Shear percentage (0 to 100)
        :return:
        """
        self.transformations.append([ObjectTransformations.shear.name, {'_shear': _shear}])

        self.flip(Orientation.Left)
        _shear = _shear / 100
        transform = skimage.transform.AffineTransform(shear=_shear)

        temp_pixels = self.actual_pixels[self.border_size.Down: self.dimensions.dy - self.border_size.Up,
                                         self.border_size.Right: self.dimensions.dx - self.border_size.Left]

        large_pixels = np.ones((300, 300))
        large_pixels[30: 30 + temp_pixels.shape[0], 170: 170 + temp_pixels.shape[1]] = temp_pixels
        large_pixels_sheared = skimage.transform.warp(large_pixels, inverse_map=transform.inverse, order=0)
        coloured_pos = np.argwhere(large_pixels_sheared > 1)

        if len(coloured_pos) == 0:
            self.show()
        else:
            top_left = coloured_pos.min(0)
            bottom_right = coloured_pos.max(0)
            new_pixels = large_pixels_sheared[top_left[0]:bottom_right[0] + 1, top_left[1]: bottom_right[1] + 1]
            self.actual_pixels = np.ones((new_pixels.shape[0] + self.border_size.Up + self.border_size.Down,
                                           new_pixels.shape[1] + self.border_size.Left + self.border_size.Right))
            self.actual_pixels[self.border_size.Down: new_pixels.shape[0] + self.border_size.Down,
                               self.border_size.Right: new_pixels.shape[1] + self.border_size.Right] = new_pixels

        self.flip(Orientation.Right)
        self._reset_dimensions()
        self.symmetries = []  # Loose any symmetries

    def mirror(self, axis: Orientation | None, on_axis=False):
        """
        Mirrors to object (copy, flip and move) along one of the edges (up, down, left or right). If on_axis is True
        the pixels along the mirror axis do not get copied

        :param axis: The axis of mirroring (e.g. Orientation.Up means along the top edge of the object)
        :param on_axis: If it is True the pixels along the mirror axis do not get copied
        :return:
        """

        if axis is not None:
            if axis == Orientation.Up or axis == Orientation.Down:
                concat_pixels = np.flipud(self.actual_pixels)
                if on_axis:
                    concat_pixels = concat_pixels[:-1, :] if axis == Orientation.Down else concat_pixels[1:, :]

                self.actual_pixels = np.concatenate((self.actual_pixels, concat_pixels), axis=0) \
                    if axis == Orientation.Up else np.concatenate((concat_pixels, self.actual_pixels), axis=0)
                if axis == Orientation.Down:
                    self.canvas_pos = Point(self.canvas_pos.x, self.canvas_pos.y - self.dimensions.dy // 2)

                new_symmetry_axis_origin = Point(self.canvas_pos.x, self.dimensions.dy / 2 + self.canvas_pos.y)

                new_symmetry_axis_origin.y -= 0.5

                if on_axis and axis == Orientation.Up:
                    for sym in self.symmetries:
                        if sym.orientation == Orientation.Right and sym.origin.y > new_symmetry_axis_origin.y:
                            sym.origin.y -= 1

                symmetry_vector = Vector(orientation=Orientation.Right, origin=new_symmetry_axis_origin,
                                         length=self.actual_pixels.shape[1] - 1)

            elif axis == Orientation.Left or axis == Orientation.Right:
                concat_pixels = np.fliplr(self.actual_pixels)
                if on_axis:
                    concat_pixels = concat_pixels[:, 1:] if axis == Orientation.Right else concat_pixels[:, :-1]

                self.actual_pixels = np.concatenate((self.actual_pixels, concat_pixels), axis=1) if axis == Orientation.Right else \
                    np.concatenate((concat_pixels, self.actual_pixels), axis=1)
                if axis == Orientation.Left:
                    self.canvas_pos = Point(self.canvas_pos.x - self.dimensions.dx // 2, self.canvas_pos.y)

                new_symmetry_axis_origin = Point(self.dimensions.dx / 2 + self.canvas_pos.x, self.canvas_pos.y)\
                   # if axis == Orientation.Right else Point(self.canvas_pos.x, self.canvas_pos.y)

                new_symmetry_axis_origin.x -= 0.5

                if on_axis and axis == Orientation.Left:
                    for sym in self.symmetries:
                        if sym.orientation == Orientation.Up and sym.origin.x > new_symmetry_axis_origin.x:
                            sym.origin.x -= 1

                symmetry_vector = Vector(orientation=Orientation.Up, origin=new_symmetry_axis_origin,
                                         length=self.actual_pixels.shape[0] - 1)

            self.symmetries.append(symmetry_vector)
            self.transformations.append([ObjectTransformations.mirror.name, {'axis': axis}])

    def flip(self, axis: Orientation | None, translate: bool = False):
        """
        Flips the object along an axis. If the Orientation is diagonal it will flip twice
        :param axis: The direction to flip towards. The edge of the bbox toward that direction becomes the axis of the flip.
        :param translate: If True then move the Object by its flip, otherwise keep it where it was.
        :return: Nothing
        """

        if axis is not None:
            if axis == Orientation.Up or axis == Orientation.Down:
                self.actual_pixels = np.flipud(self.actual_pixels)
                if translate:
                    self.canvas_pos = Point(self.canvas_pos.x, self.canvas_pos.y + self.dimensions.dy) \
                        if axis == Orientation.Up else Point(self.canvas_pos.x, self.canvas_pos.y - self.dimensions.dy)
            elif axis == Orientation.Left or axis == Orientation.Right:
                self.actual_pixels = np.fliplr(self.actual_pixels)
                if translate:
                    self.canvas_pos = Point(self.canvas_pos.x+ self.dimensions.dx, self.canvas_pos.y) \
                        if axis == Orientation.Right else Point(self.canvas_pos.x - self.dimensions.dx, self.canvas_pos.y)
            else:
                self.actual_pixels = np.flipud(self.actual_pixels)
                self.actual_pixels = np.fliplr(self.actual_pixels)
                if translate:
                    if axis in [Orientation.Up_Right, Orientation.Up_Left]:
                        self.canvas_pos = Point(self.canvas_pos.x, self.canvas_pos.y + self.dimensions.dy)
                    elif axis in [Orientation.Down_Right, Orientation.Down_Left]:
                        self.canvas_pos = Point(self.canvas_pos.x, self.canvas_pos.y - self.dimensions.dy)
                    if axis in [Orientation.Up_Right, Orientation.Down_Right]:
                        self.canvas_pos = Point(self.canvas_pos.x + self.dimensions.dx, self.canvas_pos.y)
                    elif axis in [Orientation.Up_Left, Orientation.Down_Left]:
                        self.canvas_pos = Point(self.canvas_pos.x - self.dimensions.dx, self.canvas_pos.y)

            self.transformations.append([ObjectTransformations.flip.name, {'axis': axis}])

    def delete(self):
        """
        Set all the pixels of the object to 1 (black).
        :return:
        """
        c = self.get_coloured_pixels_positions()
        self.actual_pixels[c[:, 0], c[:, 1]] = 1

    def fill_holes(self, colour: int):
        """
        Fill all holes of the Object with colour
        :param colour: The colour to fill_holes with
        :return:
        """
        if self.holes is not None:
            h = np.argwhere(self.holes > 0)
            if len(h) > 0:
                self.actual_pixels[h[:, 0], h[:, 1]] = colour

    def fill(self, colour: int):
        """
        Changes the colour of the inside pixels to colour
        :param colour: The colour to change the inside pixels to
        :return:
        """
        self.fill_holes(colour)

        pixels = self.inside
        for p in pixels:
            index = np.array(p) - self.canvas_pos.to_numpy()[:2]
            self.actual_pixels[index[1], index[0]] = colour

    def negate_colour(self):
        """
        Swaps between the coloured and the black pixels but uses the self.colour for the new colour (so any colour
        changes get lost).
        :return:
        """
        temp = copy(self.actual_pixels)
        self.actual_pixels = np.ones(self.actual_pixels.shape)
        self.actual_pixels[np.where(temp == 1)] = self.colour

    def replace_colour(self, final_colour: int, initial_colour: int | None = None):
        """
        Replace one colour with another
        :param initial_colour: The colour to change
        :param final_colour: The colour to change to
        :return:
        """
        if initial_colour is None:
            initial_colour = self.colour

        self.actual_pixels[np.where(self.actual_pixels == initial_colour)] = final_colour
        if self.colour == initial_colour:
            self.colour = final_colour
        self.transformations.append([ObjectTransformations.replace_colour.name, {'initial_colour': initial_colour,
                                                                            'final_colour': final_colour}])

    def colour_pixels(self, pixel_positions: List[Point] | List[np.ndarray], colour: int | Colour):
        if isinstance(colour, Colour):
            colour = colour.index

        array_pixel_positions = []
        if isinstance(pixel_positions[0], Point):
            for pos in pixel_positions:
                array_pixel_positions.append(np.array([(pos - self.canvas_pos).to_numpy()[1],
                                                       (pos - self.canvas_pos).to_numpy()[0]]).astype(int))
        else:
            for pos in pixel_positions:
                array_pixel_positions.append(np.array([pos[0] - self.canvas_pos.y,
                                                       pos[1] - self.canvas_pos.x]))

        for pix in array_pixel_positions:
            self.actual_pixels[pix[0], pix[1]] = colour


    def replace_all_colours(self, colours_hash: dict[int, int]):
        """
        Swaps all the colours of the Object according to the colours_hash dict
        :param colours_hash: The dict that defines what colour will become what (old colour = key, new colour = value)
        :return:
        """
        colours = self.get_used_colours()
        temp_pixels = copy(self.actual_pixels)
        for c in colours:
            if c in colours_hash:
                temp_pixels[np.where(self.actual_pixels == c)] = colours_hash[c]
                if self.colour == c:
                    self.colour = colours_hash[c]
        self.actual_pixels = copy(temp_pixels)
        self.transformations.append([ObjectTransformations.replace_all_colours, {'colours_hash': colours_hash}])

    def create_new_object_from_colour(self, colour: int) -> Object:
        new_actual_pixels = np.ones(self.actual_pixels.shape)
        positions = np.array([np.array(pix) - np.array([self.canvas_pos.y, self.canvas_pos.x]) for
                              pix in self.get_coloured_pixels_positions(colour)])
        new_actual_pixels[positions[:, 0], positions[:, 1]] = colour
        new_object = copy(self)
        new_object.actual_pixels = new_actual_pixels

        return new_object

    def split_object_along_axis(self, cut_orientation: Orientation, percentage: float | None = None,
                                pixels: int | None = None) -> Tuple[Object, Object]:
        """
        It splits the object in two new objects along an axis.
        :param cut_orientation:
        :param percentage:
        :param pixels:
        :return:
        """
        if cut_orientation == Orientation.Left or cut_orientation == Orientation.Right:  # Cut up/down
            cut_point = pixels if pixels is not None else int(self.dimensions.dy * percentage)
            new_actual_pixels_bottom = self.actual_pixels[:cut_point, :]
            new_actual_pixels_top = self.actual_pixels[cut_point:, :]

            obj_b = Object(actual_pixels=new_actual_pixels_bottom, canvas_pos=self.canvas_pos)
            cut_point = cut_point if cut_point > 0 else self.dimensions.dy + cut_point
            obj_t = Object(actual_pixels=new_actual_pixels_top, canvas_pos=Point(self.canvas_pos.x,
                                                                                 self.canvas_pos.y + cut_point))
            return obj_b, obj_t

        if cut_orientation == Orientation.Up or cut_orientation == Orientation.Down:  # Cut left/right
            cut_point = pixels if pixels is not None else int(self.dimensions.dx * percentage)
            new_actual_pixels_left = self.actual_pixels[:, :cut_point]
            new_actual_pixels_right = self.actual_pixels[:, cut_point:]

            obj_l = Object(actual_pixels=new_actual_pixels_left, canvas_pos=self.canvas_pos)
            cut_point = cut_point if cut_point > 0 else self.dimensions.dx + cut_point
            obj_r = Object(actual_pixels=new_actual_pixels_right, canvas_pos=Point(self.canvas_pos.x + cut_point,
                                                                                   self.canvas_pos.y))
            return obj_l, obj_r

    def split_object_in_quarters(self, round_to_include: bool = True) -> Tuple[Object, Object, Object, Object]:
        left_cut = self.dimensions.dx // 2 if self.dimensions.dx % 2 == 0 or \
            round_to_include is False else self.dimensions.dx // 2 + 1
        right_cut = self.dimensions.dx // 2 if self.dimensions.dx % 2 == 0 or \
            round_to_include is True else self.dimensions.dx // 2 + 1
        up_cut = self.dimensions.dy // 2 if self.dimensions.dy % 2 == 0 or \
            round_to_include is True else self.dimensions.dy // 2 + 1
        down_cut = self.dimensions.dy // 2 if self.dimensions.dy % 2 == 0 or \
            round_to_include is False else self.dimensions.dy // 2 + 1

        ul_actual_pixels = self.actual_pixels[up_cut:, :left_cut]
        ul_obj = Object(actual_pixels=ul_actual_pixels)
        ul_obj.canvas_pos = Point(self.canvas_pos.x, self.canvas_pos.y + up_cut)
        ur_actual_pixels = self.actual_pixels[up_cut:, right_cut:]
        ur_obj = Object(actual_pixels=ur_actual_pixels)
        ur_obj.canvas_pos = Point(self.canvas_pos.x + right_cut, self.canvas_pos.y + up_cut)
        dl_actual_pixels = self.actual_pixels[:down_cut, :left_cut]
        dl_obj = Object(actual_pixels=dl_actual_pixels)
        dl_obj.canvas_pos = copy(self.canvas_pos)
        dr_actual_pixels = self.actual_pixels[:down_cut, right_cut:]
        dr_obj = Object(actual_pixels=dr_actual_pixels)
        dr_obj.canvas_pos = Point(self.canvas_pos.x + right_cut, self.canvas_pos.y)

        return ul_obj, ur_obj, dl_obj, dr_obj

    # </editor-fold>

    # <editor-fold desc="RANDOMISATION METHODS">
    def randomise_colour(self, ratio: int = 10, colour: str = 'random'):
        """
        Changes the colour of ratio of the coloured pixels (picked randomly) to a new random (not already there) colour
        :param ratio: The percentage of the coloured pixels to be recoloured (0 to 100)
        :param colour: The colour to change the pixels to. 'random' means a random colour (not already on the object), 'x' means use the colour number x
        :return:
        """
        new_pixels_pos = self.pick_random_pixels(coloured_or_background='coloured', ratio=ratio)

        if new_pixels_pos is not None:
            if colour == 'random':
                colours = self.get_used_colours()
                new_colour = np.setdiff1d(np.arange(2, 11), colours)
                new_colour = np.random.choice(new_colour, size=1)
            else:
                new_colour = int(colour)

            self.actual_pixels[new_pixels_pos[:, 0], new_pixels_pos[:, 1]] = new_colour

            self.symmetries = []

            self.transformations.append([ObjectTransformations.randomise_colour.name, {'ratio': ratio}])

    def randomise_shape(self, add_or_subtract: str = 'sum', ratio: int = 10, colour: str = 'common'):
        """
        Adds or subtracts coloured pixels to the object
        :param add_or_subtract: To sum or subtract pixels. 'sum' or 'subtract'
        :param ratio: The percentage (ratio) of pixels to be added or subtracted
        :param colour: Whether the colour used for added pixels should be the most common one used or a random one or a specific one. 'common' or 'random' or 'x' where x is the colour number (from 2 to 10)
        :return:
        """
        coloured_or_background = 'background' if add_or_subtract == 'sum' else 'coloured'
        new_pixels_pos = self.pick_random_pixels(coloured_or_background=coloured_or_background, ratio=ratio)

        if new_pixels_pos is not None:
            if add_or_subtract == 'sum':
                if colour == 'common':
                    colours = self.actual_pixels[np.where(self.actual_pixels > 1)].astype(int)
                    if len(colours) == 0:
                        new_colour = np.random.randint(2, 10, 1)
                    else:
                        new_colour = int(np.argmax(np.bincount(colours)))
                elif colour == 'random':
                    new_colour = np.random.randint(2, 10, 1)
                else:
                    new_colour = int(colour)

            elif add_or_subtract == 'subtract':
                if len(new_pixels_pos) == self.actual_pixels.size:  # Stop all pixels becoming black
                    ratio /= 2
                    new_pixels_pos = self.pick_random_pixels(coloured_or_background=coloured_or_background, ratio=ratio)
                new_colour = 1
                # If all pixels become black then turn one back on
                if np.all(self.actual_pixels == 1):
                    self.actual_pixels[np.random.choice(range(self.actual_pixels.shape[0])), np.random.choice(
                        range(self.actual_pixels.shape[1]))] = self.colour


            self.actual_pixels[new_pixels_pos[:, 0], new_pixels_pos[:, 1]] = new_colour

            self.symmetries = []

            self.transformations.append([ObjectTransformations.randomise_shape.name, {'add_or_subtract': add_or_subtract,
                                                                                      'ratio': ratio}])

    def create_random_hole(self, hole_size: int) -> bool:
        """
        Tries to create a hole in the Object of total empty pixels = hole_size or smaller. If it succeeds it returns True,
        otherwise it returns False. The hole creation process is totally random and cannot be controlled other than
        setting the maximum number of empty pixels generated.
        :param hole_size: The maximum possible number of connected empty pixels the hole is made out of
        :return: True if the process was successful (there is a hole) and False if not
        """
        pixels = copy(self.actual_pixels)
        initial_holes, initial_n_holes, _ = self.detect_holes()
        connected_components, n = ndi.label(np.array(pixels > 1).astype(int))
        largest_component = 0
        for i in range(1, np.max(connected_components) + 1):
            if len(np.where(connected_components == i)[0]) > len(np.where(connected_components == largest_component)[0]):
                largest_component = i

        if largest_component > 0:
            for _ in range(1000):  # Try 1000 times
                pixels = copy(self.actual_pixels)
                current_hole_size = 0
                hole_points = []
                focus_index = np.random.randint(0, len(np.where(connected_components == largest_component)[0]))
                focus_point = np.array([np.where(connected_components == largest_component)[0][focus_index],
                                        np.where(connected_components == largest_component)[1][focus_index]])
                hole_points.append(focus_point)
                current_hole_size += 1
                while hole_size > current_hole_size:
                    new_points = np.array([focus_point + np.array([-1, 0]), focus_point + np.array([1, 0]),
                                           focus_point + np.array([0, -1]), focus_point + np.array([0, 1])])
                    new_point_found = False
                    np.random.shuffle(new_points)
                    for p in new_points:
                        try:
                            if pixels[p[0], p[1]] == pixels[focus_point[0], focus_point[1]]:
                                hole_points.append(p)
                                current_hole_size += 1
                                new_point_found = True
                                break
                        except IndexError:
                            pass

                    if new_point_found:
                        focus_point = p
                    else:
                        break

                for hp in hole_points:
                    pixels[hp[0], hp[1]] = 1
                final_holes, final_n_holes, _ = self.detect_holes(pixels)

                if final_n_holes > initial_n_holes:
                    self.actual_pixels = pixels
                    self.symmetries = []
                    return True

        return False
    # </editor-fold>

    # <editor-fold desc="UTILITY METHODS">
    @staticmethod
    def actual_pixels_index_to_canvas_coordinates(actual_pixels_index: Tuple[int, int]) -> Point:
        return Point(actual_pixels_index[1], actual_pixels_index[0])

    def _reset_dimensions(self):
        """
        Reset the self.dimensions and the self.bbox top left and bottom right points to fit the updated actual_pixels
        :return:
        """
        self.dimensions.dx = self.actual_pixels.shape[1]
        self.dimensions.dy = self.actual_pixels.shape[0]

        bb_top_left = Point(self.canvas_pos.x, self.canvas_pos.y + self.dimensions.dy - 1, self.canvas_pos.z)
        bb_bottom_right = Point(bb_top_left.x + self.dimensions.dx - 1, self.canvas_pos.y, self.canvas_pos.z)
        self.bbox = Bbox(top_left=bb_top_left, bottom_right=bb_bottom_right)

        coloured_pixels = self.get_coloured_pixels_positions()
        if len(coloured_pixels) > 0:
            bb_top_left = Point(np.min(coloured_pixels, axis=0)[1], np.max(coloured_pixels, axis=0)[0])
            bb_bottom_right = Point(np.max(coloured_pixels, axis=0)[1], np.min(coloured_pixels, axis=0)[0])
            self._visible_bbox = Bbox(top_left=bb_top_left, bottom_right=bb_bottom_right)
        else:
            self._visible_bbox = None

    def get_distance_to_object(self, other: Object, dist_type: str = 'min') -> Vector | None:
        """
        Calculates the Vector that defines the distance (in pixels) between this and the other Object. The exact
        calculation depends on the type asked for. If type is 'min' then this is the distance between the two nearest
        pixels of the Objects. If it is 'max' it is between the two furthest_point_to_point. If it is 'canvas_pos' then
        it is the distance between the two canvas positions of the Objects. If it is 'straight_line' then it is the
        minimum straight line distance between the objects, i.e. the smallest distance one object needs to move in a
        straight line (along an Orientation) to touch the other object.
        If the two Points that are being compared lie along a Direction then the returned Vector also has this Direction.
        If more than two pairs of points qualify to calculate the distance then one with a Direction is chosen (with
        a preference to Directions Up, Down, Left and Right).
        :param other: The other Object
        :param dist_type: str. Can be 'max', 'min', 'canvas_pos', 'straight_line'
        :return: A Vector whose length is the distance calculated, whose origin in the Point in this Object used to calculate the distance and whose orientation is the Orientation between the two Points used if it exists
        (otherwise it is None).
        """
        try:
            if dist_type == 'min' or dist_type == 'max' or dist_type == 'straight_line':
                all_self_colour_points = [Point(pos[1], pos[0]) for pos in self.get_coloured_pixels_positions()]
                all_other_colour_points = [Point(pos[1], pos[0]) for pos in other.get_coloured_pixels_positions()]

                all_distances = []
                all_distance_lengths = []
                for sp in all_self_colour_points:
                    for op in all_other_colour_points:
                        all_distances.append(sp.euclidean_distance(op))
                        all_distance_lengths.append(all_distances[-1].length)

                if dist_type == 'max':
                    indices = np.argwhere(np.array(all_distance_lengths) == np.amax(all_distance_lengths)).squeeze()
                    distances = np.take(all_distances, indices)
                elif dist_type == 'min':
                    indices = np.argwhere(np.array(all_distance_lengths) == np.amin(all_distance_lengths)).squeeze()
                    distances = np.take(all_distances, indices)
                elif dist_type == 'straight_line':
                    distances_with_orientation = [dist for dist in all_distances if dist.orientation is not None]
                    distance_lengths_with_orientation = [dist.length for dist in all_distances if dist.orientation is not None]
                    if len(distance_lengths_with_orientation) > 0:
                        indices = np.argwhere(np.array(distance_lengths_with_orientation) == np.amin(distance_lengths_with_orientation)).squeeze()
                        distances = np.take(distances_with_orientation, indices)
                    else:
                        indices = np.argwhere(np.array(all_distance_lengths) == np.amin(all_distance_lengths)).squeeze()
                        distances = np.take(all_distances, indices)

                if type(distances) != np.ndarray:
                    distances = [distances]

                distances_with_ori = [dist for dist in distances if dist.orientation is not None]
                if len(distances_with_ori) > 0:
                    distances_with_good_ori = [dist for dist in distances_with_ori if (dist.orientation == Orientation.Up
                                                                                       or dist.orientation == Orientation.Down
                                                                                       or dist.orientation == Orientation.Right
                                                                                       or dist.orientation == Orientation.Left)]
                    distance = distances_with_good_ori[0] if len(distances_with_good_ori) > 0 else distances_with_ori[0]
                else:
                    distance = distances[0]

            if dist_type == 'canvas_pos':
                distance = self.canvas_pos.euclidean_distance(other.canvas_pos)
        except:
            return None

        return distance

    def get_straight_distance_to_object(self, other: Object, up_to_corner: bool = True) -> Vector | None:
        """
        Returns a Vector that show the Orientation and the number of pixels the Object should be moved in a straight
        line in order to touch the other Object. If the Object cannot touch the other Object by moving along one of the
        8 Orientations then it returns None. It also returns None if the Objects are on top of each other.
        :param up_to_corner: If True the distance is calculated for the two Objects to touch up to their corners. If False then up until they fit together.
        :param other: The other Object
        :return: The Vector straight line distance
        """
        temp_self = copy(self)

        dir = self.get_direction_to_object(other)
        if dir is None:
            return None

        step = dir.get_step_towards_orientation()

        func = temp_self.is_object_touching if up_to_corner else temp_self.is_object_superimposed
        num_of_steps = 0
        while not func(other):
            temp_self.translate_by(step)
            num_of_steps += 1
            if num_of_steps > 40:
                return None

        num_of_steps = num_of_steps - 1 if not up_to_corner else num_of_steps
        distance = Vector(orientation=self.get_direction_to_object(other), length=num_of_steps,  origin=self.canvas_pos)

        return distance

    def get_direction_to_object(self, other: Object) -> Orientation | None:
        """
        Return the Orientation the other Object is with respect to this one.
        :param other: The other Object
        :return: The Orientation
        """
        if self.is_object_superimposed(other):
            return None

        if self.canvas_pos.x > other.canvas_pos.x:
            if self.canvas_pos.y > other.canvas_pos.y:
                if other.canvas_pos.y + other.dimensions.dy < self.canvas_pos.y:
                    if other.canvas_pos.x + other.dimensions.dx > self.canvas_pos.x:
                        return Orientation.Down
                    else:
                        return Orientation.Down_Left
                else:
                    return Orientation.Left
            else:
                if self.canvas_pos.y + self.dimensions.dy < other.canvas_pos.y:
                    if other.canvas_pos.x + other.dimensions.dx > self.canvas_pos.x:
                        return Orientation.Up
                    else:
                        return Orientation.Up_Left
                else:
                    return Orientation.Left
        else:
            if self.canvas_pos.y > other.canvas_pos.y:
                if other.canvas_pos.y + other.dimensions.dy < self.canvas_pos.y:
                    if self.canvas_pos.x + self.dimensions.dx > other.canvas_pos.x:
                        return Orientation.Down
                    else:
                        return Orientation.Down_Right
                else:
                    return Orientation.Right
            else:
                if self.canvas_pos.y + self.dimensions.dy < other.canvas_pos.y:
                    if self.canvas_pos.x + self.dimensions.dx > other.canvas_pos.x:
                        return Orientation.Up
                    else:
                        return Orientation.Up_Right
                else:
                    return Orientation.Right

    def is_object_superimposed(self, other: Object) -> bool:
        """
        Returns whether some of the visible pixels of this Object and of the other Object are on the same coordinates
        :param other: The other Object
        :return: True or False
        """
        self_pixs = set((i[1], i[0]) for i in self.get_coloured_pixels_positions())
        other_pixs = set((i[1], i[0]) for i in other.get_coloured_pixels_positions())
        if len(self_pixs - other_pixs) == len(self_pixs):
            return False
        return True

    def is_object_overlapped(self, other: Object) -> bool:
        """
        Returns whether some visible pixels of this Object are in the same coordinates as some of the pixels of the other
        Object and the pixels of this Object are visible on the Canvas (this Object has a larger canvas_pos.z)
        :param other: The other Object
        :return: True or False
        """
        if self.is_object_superimposed(other) and self.canvas_pos.z > other.canvas_pos.z:
            return True
        return False

    def is_object_underlapped(self, other: Object) -> bool:
        """
        Returns whether some visible pixels of this Object are in the same coordinates as some of the pixels of the other
        Object and the pixels of the other Object are visible on the Canvas (this Object has a smaller canvas_pos.z)
        :param other: The other Object
        :return: True or False
        """
        if self.is_object_superimposed(other) and self.canvas_pos.z < other.canvas_pos.z:
            return True
        return False

    def is_object_touching(self, other: Object) -> bool:
        if self.is_object_superimposed(other):
            return False

        # Check to see if the Objects touch at a corner
        steps = [Dimension2D(0, 1), Dimension2D(0, -1), Dimension2D(1, 0), Dimension2D(-1, 0),
                     Dimension2D(1, 1), Dimension2D(1, -1), Dimension2D(-1, 1), Dimension2D(-1, -1)]

        for step in steps:
            self_temp = copy(self)
            self_temp.translate_by(step)
            if self_temp.is_object_superimposed(other):
                return True

        return False

    def is_object_matching_to_object(self, obj: Object, match_shape_only: bool = False,
                                     transformations: List[str] | None = ('rotate', 'scale', 'flip', 'invert')) -> bool:
        def basic_match(obj_a: Object, obj_b: Object):
            result = False
            if match_shape_only:
                if np.array_equal(obj_a.actual_pixels > 1, obj_b.actual_pixels > 1):
                    result = True
            else:
                if np.array_equal(obj_a.actual_pixels, obj_b.actual_pixels):
                    result = True
            return result

        obj_b = copy(obj)
        if transformations is not None:
            rots = [1, 2, 3] if 'rotate' in transformations else [0]
            scales = [2, 3] if 'scale' in transformations else [1]
            orientations = ObjectTransformations(10).get_all_possible_parameters() if 'flip' in transformations else [None]
            if len(orientations) > 1:
                orientations.remove(None)
            for rot in rots:
                for scale in scales:
                    for orientation in orientations:
                        obj_a = copy(self)
                        if 'rotate' in transformations:
                            obj_a.rotate(rot)
                        if 'scale' in transformations:
                            obj_a.scale(scale)
                        if 'flip' in transformations:
                            obj_a.flip(orientation)
                        if 'invert' in transformations:
                            obj_a.negate_colour()
                        if basic_match(obj_a, obj_b):
                            return True
        else:
            return basic_match(self, obj)

        return False

    def is_object_along_x_to_object(self, obj: Object) -> bool:
        temp_obj = copy(obj)
        temp_obj.translate_to_coordinates(target_point=Point(self.canvas_pos.x, temp_obj.canvas_pos.y))

        if self.is_object_superimposed(temp_obj):
            return True
        else:
            return False

    def is_object_along_y_to_object(self, obj: Object) -> bool:
        temp_obj = copy(obj)
        temp_obj.translate_to_coordinates(target_point=Point(temp_obj.canvas_pos.x, self.canvas_pos.y))

        if self.is_object_superimposed(temp_obj):
            return True
        else:
            return False

    def is_object_along_xy_to_object(self, obj: Object) -> bool:
        dif = (self.canvas_pos - obj.canvas_pos).to_numpy()[:2]
        if dif[0] == dif[1]:
            return True
        else:
            return False

    def is_object_along_xminusy_to_object(self, obj: Object) -> bool:
        dif = (self.canvas_pos - obj.canvas_pos).to_numpy()[:2]
        if dif[0] == - dif[1]:
            return True
        else:
            return False

    def is_object_over_object(self, obj: Object) -> bool:
        temp_obj = copy(obj)
        temp_obj.translate_to_coordinates(Point(self.canvas_pos.x, obj.canvas_pos.y))
        if not self.is_object_superimposed(temp_obj) and self.canvas_pos.y >= temp_obj.canvas_pos.y + temp_obj.dimensions.dy:
            return True
        else:
            return False

    def is_object_under_object(self, obj: Object) -> bool:
        temp_obj = copy(obj)
        temp_obj.translate_to_coordinates(Point(self.canvas_pos.x, obj.canvas_pos.y))
        if not self.is_object_superimposed(temp_obj) and self.canvas_pos.y + self.dimensions.dy <= temp_obj.canvas_pos.y:
            return True
        else:
            return False

    def is_object_left_of_object(self, obj: Object) -> bool:
        temp_obj = copy(obj)
        temp_obj.translate_to_coordinates(Point(obj.canvas_pos.x, self.canvas_pos.y))
        if not self.is_object_superimposed(temp_obj) and self.canvas_pos.x + self.dimensions.dx <= temp_obj.canvas_pos.x:
            return True
        else:
            return False

    def is_object_right_of_object(self, obj: Object) -> bool:
        temp_obj = copy(obj)
        temp_obj.translate_to_coordinates(Point(obj.canvas_pos.x, self.canvas_pos.y))
        if not self.is_object_superimposed(temp_obj) and self.canvas_pos.x >= temp_obj.canvas_pos.x + temp_obj.dimensions.dx:
            return True
        else:
            return False

    def get_coloured_pixels_positions(self, col: int | None = None) -> np.ndarray:
        """
        Get a numpy array with the positions (y,x) of all the coloured pixels of the Object
        :param col: If col is None then get all visible pixels. Otherwise, get only the ones with the col colour
        :return: A numpy array with the coordinates (y,x).
        """
        if col is None:
            result = np.argwhere(self.actual_pixels > 1).astype(int)
        else:
            result = np.argwhere(self.actual_pixels == col).astype(int)
        canv_pos = np.array([self.canvas_pos.to_numpy()[1], self.canvas_pos.to_numpy()[0]]).astype(int)
        result = canv_pos + result
        return result

    def get_background_pixels_positions(self) -> np.ndarray:
        """
        Get a numpy array with the positions (y,x) of all the black pixels of the Object
        :return: A numpy array with the coordinates (y,x).
        """
        return np.argwhere(self.actual_pixels == 1)

    def get_used_colours(self) -> Set:
        """
        Returns the set of all used colours
        :return: The set of used colours
        """
        coloured_pos = self.get_coloured_pixels_positions().astype(int)
        canv_pos = np.array([self.canvas_pos.to_numpy()[1], self.canvas_pos.to_numpy()[0]]).astype(int)
        coloured_pos -= canv_pos
        return set(np.unique(self.actual_pixels[coloured_pos[:, 0], coloured_pos[:, 1]]).astype(int))

    def get_colour_groups(self) -> Dict[int, np.ndarray[List]]:
        """
        Returns a dictionary with keys the different colours of the Object and values the array of positions (y, x)of
        the pixels with the corresponding colours
        :return: The colour -> positions dict
        """
        result = {}
        colours = self.get_used_colours()
        for col in colours:
            positions = self.get_coloured_pixels_positions(col)
            result[int(col)] = positions
        return result

    def get_number_of_pixels_for_each_colour(self) -> np.ndarray[int]:
        result = np.zeros(10)

        for i in self.get_colour_groups():
            result[i] = len(self.get_colour_groups()[i])

        return result

    def get_most_common_colour(self) -> int:
        colours = self.actual_pixels[np.where(self.actual_pixels > 1)]
        if len(colours) == 0:  # A totally empty object
            return 1
        return int(np.median(colours))

    def set_colour_to_most_common(self):
        """
        Sets the colour property of the Object to the most common colour (with the most pixels)
        :return:
        """
        self.colour = self.get_most_common_colour()

    def pick_random_pixels(self, coloured_or_background: str = 'coloured', ratio: int = 10) -> None | np.ndarray:
        """
        Returns the positions (in the self.actual_pixels array) of a random number (ratio percentage) of either
        coloured or background pixels
        :param coloured_or_background: Whether the pixels should come from the coloured group or the background group. 'coloured' or 'background'
        :param ratio: The ratio (percentage) of the picked pixels over the number of the pixels in their group
        :return:
        """
        if coloured_or_background == 'coloured':
            pixels_pos = self.get_coloured_pixels_positions()
            canv_pos = np.array([self.canvas_pos.to_numpy()[1], self.canvas_pos.to_numpy()[0]]).astype(int)
            pixels_pos -= canv_pos
        elif coloured_or_background == 'background':
            pixels_pos = self.get_background_pixels_positions()

        num_of_new_pixels = int((pixels_pos.size // 2) * ratio / 100)
        if num_of_new_pixels < 1:
            num_of_new_pixels = 1

        if len(pixels_pos) > 0:
            t = np.random.choice(np.arange(len(pixels_pos)), num_of_new_pixels)
            return pixels_pos[t]
        else:
            return None

    def detect_holes(self, pixels: np.ndarray | None = None) -> Tuple[np.ndarray, int, int|None]:
        """
        Detects the presence of any holes. A hole is a spatially contiguous set of empty pixels that are fully
        surrounded by coloured pixels.
        :param pixels: The 2d array of pixels in which to detect the presence of holes.
        :return: The array marked with the holes' labels, the number of holes found and the size of each hole (in pixels)
        """
        if pixels is None:
            pixels = copy(self.actual_pixels)

        connected_components, n = ndi.label(np.array(pixels == 1).astype(int))

        for i in np.unique(connected_components):
            comp_coords = np.where(connected_components == i)
            if np.any(comp_coords[0] == 0) or \
                    np.any(comp_coords[1] == 0) or \
                    np.any(comp_coords[0] == pixels.shape[0] - 1) or \
                    np.any(comp_coords[1] == pixels.shape[1] - 1):
                connected_components[comp_coords[0], comp_coords[1]] = 0

        holes, n = ndi.label(connected_components)
        sizes = []
        if n > 0:
            for i in range(n+1)[1:]:
                s = len(np.argwhere(holes == i))
                sizes.append(s)
        else:
            sizes = None

        return holes, n, sizes

    def get_shape_index(self) -> int | None:
        m = self.dimensions.dx
        n = self.dimensions.dy

        if m * n > 20:
            return None
        else:
            size = m * n

            all_possible_designs = np.array(list(itertools.product([0, 1], repeat=size)))
            all_possible_designs = [e.reshape((n, m)) for e in all_possible_designs]

            binary_equivalence = [np.array_equal(e, self.actual_pixels > 1) for e in all_possible_designs]

            return np.where(binary_equivalence)[0][0]

    def get_2x2_shape_index(self) -> int | None:

        if self.dimensions != Dimension2D(2, 2):
            return -1
        else:
            self.get_shape_index()

    def get_3x3_shape_index(self) -> int | None:

        if self.dimensions != Dimension2D(3, 3):
            return -1
        else:
            self.get_shape_index()

    def get_3x3_design(self) -> int:
        all_possible_designs = np.array(list(itertools.product([0, 1], repeat=9)))
        all_possible_designs = [e.reshape((2, 2)) for e in all_possible_designs]

        binary_equivalence = [np.array_equal(e, self.actual_pixels) for e in all_possible_designs]

        return np.where(binary_equivalence)[0][0]

    def match_to_background(self, background_obj: Object, match_shape_only: bool = False, try_unique: bool = True,
                            padding: Surround = Surround(0, 0, 0, 0),
                            transformations: List[str] = ('rotate', 'scale', 'flip', 'invert', 'colour')) \
            -> List[dict[str, int | List[Point]]]:
        """

        :param background_obj: The Object over which to pass this Object to see if any part of it matches with this Object
        :param match_shape_only: Only check out the shape similarity and ignore the colour
        :param try_unique: Does a center surround algorithm trying to increase the accuracy of the match_to_background
        :param padding: Add padding surrounding the object (Surround).
        :param transformations: Allow the type of transformations this Object will undergo to try and match_to_background
        :return: A list of dictionaries describing the translations, rotations, scaling and flips of the best matches.
        """

        positions = []
        max_results = []
        transformation_results = []
        rots = ObjectTransformations(6).get_all_possible_parameters() if 'rotate' in transformations else [None]
        scales = [None, 2, 3] if 'scale' in transformations else [None]
        orientations = ObjectTransformations(10).get_all_possible_parameters() if 'flip' in transformations else [None]
        inversions = [True, None] if 'invert' in transformations else [None]
        colours: List[int | None] = [None] + list(range(2, 11)) if 'colour' in transformations and\
                                                                   len(self.get_used_colours()) == 1 else [None]
        if len(self.get_used_colours()) == 1 and len(colours) > 1:
            colours.remove(self.colour)
        for rot in rots:
            for scale in scales:
                for orientation in orientations:
                    for inversion in inversions:
                        for colour in colours:
                            filter_obj = copy(self)
                            background_obj_cp = copy(background_obj)
                            if rot is not None:
                                filter_obj.rotate(rot)
                            if scale is not None:
                                filter_obj.scale(scale)
                            if orientation is not None:
                                filter_obj.flip(orientation)
                            if inversion is not None:
                                filter_obj.negate_colour()
                            if colour is not None and colour != self.colour:
                                filter_obj.replace_colour(final_colour=colour)
                            position, max_result = Object._match_filter_obj_to_background_obj(background_obj=background_obj_cp,
                                                                                              filter_obj=filter_obj,
                                                                                              match_shape_only=match_shape_only,
                                                                                              try_unique=try_unique,
                                                                                              padding=padding)
                            transformation_results.append([rot, scale, orientation, inversion, colour])
                            positions.append(position)
                            max_results.append(max_result)

        best_results_indices = np.argwhere(max_results == np.amax(max_results))
        best_positions = [positions[i[0]] for i in best_results_indices]
        best_transformations = [transformation_results[i[0]] for i in best_results_indices]
        best_max_results = [max_results[i[0]] for i in best_results_indices]

        result = [{'id': self.id,
                   'rotate': best_transformations[i][0],
                   'scale': best_transformations[i][1],
                   'flip': best_transformations[i][2],
                   'inverse': best_transformations[i][3],
                   'replace_colour': best_transformations[i][4],
                   'translate_to_coordinates':best_positions[i],
                   'result': best_max_results[i]} for i in range(len(best_transformations))]

        return result

    def show(self, symmetries_on=True, show_holes=False, save_as: str | None = None):
        """
        Show a matplotlib.pyplot.imshow image of the actual_pixels array correctly colour transformed
        :param save_as: If a string is given the image is saved to that file.
        :param symmetries_on: Show the symmetries of the object as line
        :param show_holes: If True it marks any holes with white pixels
        :return: Nothing
        """
        xmin = self.bbox.top_left.x - 0.5
        xmax = self.bbox.bottom_right.x + 0.5
        ymin = self.bbox.bottom_right.y - 0.5
        ymax = self.bbox.top_left.y + 0.5
        extent = [xmin, xmax, ymin, ymax]
        pixels_to_show = copy(self.actual_pixels)

        if save_as is not None:
            fig, _ = vis.plot_data(pixels_to_show, extent=extent, thin_lines=False)
            fig.savefig(save_as)
            plt.close(fig)
        else:
            if show_holes:
                pixels_to_show[np.where(self.holes > 0)] = 11
            _, ax = vis.plot_data(pixels_to_show, extent=extent)

            #TODO: DEAL WITH DIAGONAL SYMMETRIES!!!!
            if symmetries_on:
                for sym in self.symmetries:
                    if sym.orientation == Orientation.Up or sym.orientation == Orientation.Down:
                        line_at = sym.origin.x
                        line_min = sym.origin.y - 0.5 if sym.orientation == Orientation.Up else sym.origin.y + 0.5
                        line_max = sym.origin.y + sym.length + 0.5 if sym.orientation == Orientation.Up else \
                            sym.origin.y - sym.length - 0.5
                        plt_lines = ax.vlines
                    else:
                        line_at = sym.origin.y
                        line_min = sym.origin.x - 0.5 if sym.orientation == Orientation.Right else sym.origin.x + 0.5
                        line_max = sym.origin.x + sym.length + 0.5 if sym.orientation == Orientation.Right else \
                            sym.origin.x - sym.length - 0.5
                        plt_lines = ax.hlines

                    plt_lines(line_at, line_min, line_max, linewidth=2)

    # </editor-fold>



