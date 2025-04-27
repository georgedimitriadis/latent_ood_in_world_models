
from __future__ import annotations

import numpy as np
from experiments.data.generation import constants as const
from typing import Union, List
from copy import copy
from enum import Enum
from dataclasses import dataclass

MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Orientation(Enum):
    Up: int = 0
    Up_Right: int = 1
    Right: int = 2
    Down_Right: int = 3
    Down: int = 4
    Down_Left: int = 5
    Left: int = 6
    Up_Left: int = 7

    def rotate(self, affine_matrix: Union[np.ndarray | None] = None, rotation: float = 0):
        assert affine_matrix is None or rotation == 0

        times = 0
        if rotation != 0:
            times = rotation // (np.pi/4)
        if affine_matrix is not None:
            if np.all(affine_matrix[:2, :2].astype(int) == np.array([[0, -1], [1, 0]])):
                times = 2
            if np.all(affine_matrix[:2, :2].astype(int) == np.array([[-1, 0], [0, -1]])):
                times = 4
            if np.all(affine_matrix[:2, :2].astype(int) == np.array([[0, 1], [-1, 0]])):
                times = 6

        value = self.value - times
        if value < 0:
            value = 8 + value

        return Orientation(value=value)

    def __copy__(self):
        return Orientation(self.value)

    def __repr__(self) -> str:
        return f'Orientation(name={self.name}, value={self.value})'

    @staticmethod
    def get_orientation_from_name(name: str) -> Orientation:
        for i in range(len(Orientation)):
            if Orientation(i).name == name:
                return Orientation(i)

    def get_step_towards_orientation(self) -> Dimension2D:
        step = Dimension2D(0, 0)
        if self.name == 'Left':
            step = Dimension2D(-1, 0)
        elif self.name == 'Right':
            step = Dimension2D(1, 0)
        if self.name == 'Up':
            step = Dimension2D(0, 1)
        elif self.name == 'Down':
            step = Dimension2D(0, -1)
        if self.name == 'Up_Left':
            step = Dimension2D(-1, 1)
        elif self.name == 'Up_Right':
            step = Dimension2D(1, 1)
        if self.name == 'Down_Left':
            step = Dimension2D(-1, -1)
        elif self.name == 'Down_Right':
            step = Dimension2D(1, -1)

        return step

    @staticmethod
    def random(probs: List = (1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8)) -> Orientation:
        return Orientation(np.random.choice(range(8), p=probs))


class OrientationZ(Enum):
    Away: int = -1
    Towards: int = 1


class RelativePoint(Enum):
    Top_Left: int = 0
    Top_Center: int = 1
    Top_Right: int = 2
    Middle_Left: int = 3
    Middle_Center: int = 4
    Middle_Right: int = 5
    Bottom_Left: int = 6
    Bottom_Center: int = 7
    Bottom_Right: int = 8

    @staticmethod
    def random() -> RelativePoint:
        return RelativePoint(np.random.randint(0, 9))


@dataclass
class RelativePoints:
    def __init__(self, a_point: Point, which_point: RelativePoint):
        if which_point == RelativePoint.Top_Left: self.Top_Left = a_point
        if which_point == RelativePoint.Top_Center: self.Top_Center = a_point
        if which_point == RelativePoint.Top_Right: self.Top_Right = a_point
        if which_point == RelativePoint.Middle_Left: self.Middle_Left = a_point
        if which_point == RelativePoint.Middle_Center: self.Middle_Center = a_point
        if which_point == RelativePoint.Middle_Right: self.Middle_Right = a_point
        if which_point == RelativePoint.Bottom_Left: self.Bottom_Left = a_point
        if which_point == RelativePoint.Bottom_Center: self.Bottom_Center = a_point
        if which_point == RelativePoint.Bottom_Right: self.Bottom_Right = a_point


@dataclass
class Surround:
    """
    Four integers that define an Up, Down, Left and Right
    """
    Up: int = 0
    Down: int = 0
    Left: int = 0
    Right: int = 0

    def __add__(self, other: Surround | int) -> Surround:
        if type(other) == Surround:
            return Surround(other.Up + self.Up, other.Down + self.Down, other.Left+self.Left, other.Right+self.Right)
        else:
            return Surround(other + self.Up, other + self.Down, other + self.Left, other + self.Right)

    def __sub__(self, other: Surround | int) -> Surround:
        if type(other) == Surround:
            return Surround(self.Up - other.Up, self.Down - other.Down, self.Left - other.Left, self.Right - other.Right)
        else:
            return Surround(self.Up - other, self.Down - other, self.Left - other, self.Right - other)

    def __iadd__(self, other: Surround | int) -> Surround:
        return self + other

    def __isub__(self, other: Surround | int) -> Surround:
        return self - other

    def __copy__(self):
        return Surround(Up=self.Up, Down=self.Down, Left=self.Left, Right=self.Right)

    def __getitem__(self, idx):
        if idx == 0:
            return self.Up
        if idx == 1:
            return self.Down
        if idx == 2:
            return self.Left
        if idx == 3:
            return self.Right

    def __setitem__(self, idx, val):
        if idx == 0:
            self.Up = val
        if idx == 1:
            self.Down = val
        if idx == 2:
            self.Left = val
        if idx == 3:
            self.Right = val

    def to_numpy(self):
        return np.array([self.Up, self.Down, self.Left, self.Right])


@dataclass
class Surround_Percentage:
    """
    Four floats that define percentages for Up, Down, Left and Right
    """
    Up: float = 0
    Down: float = 0
    Left: float = 0
    Right: float = 0

    def __add__(self, other: Surround_Percentage | float) -> Surround_Percentage:
        if type(other) == Surround_Percentage:
            return Surround_Percentage(other.Up + self.Up, other.Down + self.Down, other.Left+self.Left, other.Right+self.Right)
        else:
            return Surround_Percentage(other + self.Up, other + self.Down, other + self.Left, other + self.Right)

    def __sub__(self, other: Surround_Percentage | float) -> Surround_Percentage:
        if type(other) == Surround_Percentage:
            return Surround_Percentage(self.Up - other.Up, self.Down - other.Down, self.Left - other.Left, self.Right - other.Right)
        else:
            return Surround_Percentage(self.Up - other, self.Down - other, self.Left - other, self.Right - other)

    def __iadd__(self, other: Surround_Percentage | float) -> Surround_Percentage:
        return self + other

    def __isub__(self, other: Surround_Percentage | float) -> Surround_Percentage:
        return self - other

    def __copy__(self):
        return Surround_Percentage(Up=self.Up, Down=self.Down, Left=self.Left, Right=self.Right)

    def __getitem__(self, idx):
        if idx == 0:
            return self.Up
        if idx == 1:
            return self.Down
        if idx == 2:
            return self.Left
        if idx == 3:
            return self.Right

    def __setitem__(self, idx, val):
        if idx == 0:
            self.Up = val
        if idx == 1:
            self.Down = val
        if idx == 2:
            self.Left = val
        if idx == 3:
            self.Right = val

    def to_numpy(self):
        return np.array([self.Up, self.Down, self.Left, self.Right])


class Dimension2D:
    def __init__(self, dx: int = 3, dy:  int = 3, array: None | np.ndarray | List = None):
        if array is None:
            self.dx = dx
            self.dy = dy
        else:
            self.dx: int = array[0]
            self.dy: int = array[1]

    def __repr__(self) -> str:
        return f'Dimension:(dX = {self.dx}, dY = {self.dy})'

    def __add__(self, other) -> Dimension2D:
        if type(other) == Dimension2D:
            result = Dimension2D(self.dx + other.dx, self.dy + other.dy)
        if type(other) == Point:
            result = Dimension2D(self.dx + other.x, self.dy + other.y)
        if type(other) == list:
            result = Dimension2D(self.dx + other[0], self.dy + other[1])
        if type(other) == np.ndarray:
            result = Dimension2D(self.dx + other[0], self.dy + other[1])
        if type(other) == int or type(other) == float:
            result = Dimension2D(self.dx + other, self.dy + other)
        return result

    def __sub__(self, other) -> Dimension2D:
        if type(other) == Dimension2D:
            result = Dimension2D(self.dx - other.dx, self.dy - other.dy)
        if type(other) == Point:
            result = Dimension2D(self.dx - other.x, self.dy - other.y)
        if type(other) == list:
            result = Dimension2D(self.dx - other[0], self.dy - other[1])
        if type(other) == np.ndarray:
            result = Dimension2D(self.dx - other[0], self.dy - other[1])
        if type(other) == int or type(other) == float:
            result = Dimension2D(self.dx - other, self.dy - other)
        return result

    def __isub__(self, other) -> Dimension2D:
        return self.__sub__(other)

    def __iadd__(self, other) -> Dimension2D:
        return self.__add__(other)

    def __mul__(self, other: float | int | bool) -> Dimension2D:
        return Dimension2D(self.dx * other, self.dy * other)

    def __matmul__(self, other: Dimension2D) -> Dimension2D:
        return Dimension2D(self.dx * other.dx, self.dy * other.dy)

    def __pow__(self, power, modulo=None):
        return Dimension2D(self.dx ** power, self.dy ** power)

    def __truediv__(self, other: int | float) -> Dimension2D:
        return Dimension2D(self.x / other, self.y / other)

    def __copy__(self):
        return Dimension2D(dx=self.dx, dy=self.dy)

    def __eq__(self, other: Dimension2D) -> bool:
        if self.dx == other.dx and self.dy == other.dy:
            return True
        else:
            return False

    def to_numpy(self):
        return np.array([self.dx, self.dy])

    @staticmethod
    def random(min_dx: int = -32, max_dx: int = 32, min_dy: int = 32, max_dy: int = 32):
        if min_dx >= max_dx:
            print(f'Error in random Point: min_dx = {min_dx}, max_dx = {max_dx}')
        if min_dy >= max_dy:
            print(f'Error in random Point: min_dy = {min_dy}, max_dy = {max_dy}')
        return Dimension2D(np.random.randint(min_dx, max_dx + 1), np.random.randint(min_dy, max_dy + 1))


class Point:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, array: None | List | np.ndarray = None):
        if array is None:
            self.x = x
            self.y = y
            self.z = z  # This is used to define over or under
        else:
            self.x = array[0]
            self.y = array[1]
            try:
                self.z = array[2]
            except:
                self.z = 0

    @staticmethod
    def random(min_x: int = -32, max_x: int = 32, min_y: int = -32, max_y: int = 32, min_z: int = -100, max_z: int = 100):
        return Point(np.random.randint(min_x, max_x + 1), np.random.randint(min_y, max_y + 1),
                     np.random.randint(min_z, max_z + 1))

    def __add__(self, other) -> Point:
        if type(other) == Point:
            result = Point(self.x + other.x, self.y + other.y, self.z + other.z)
        if type(other) == list:
            result = Point(self.x + other[0], self.y + other[1], self.z + other[2])
        if type(other) == np.ndarray:
            result = Point(self.x + other[0], self.y + other[1], self.z + other[2])
        if type(other) == int or type(other) == float:
            result = Point(self.x + other, self.y + other, self.z + other)
        if type(other) == Dimension2D:
            result = Point(self.x + other.dx, self.y + other.dy, self.z)
        return result

    def __sub__(self, other) -> Point:
        if type(other) == Point:
            result = Point(self.x - other.x, self.y - other.y, self.z - other.z)
        if type(other) == list:
            result = Point(self.x - other[0], self.y - other[1], self.z - other[2])
        if type(other) == np.ndarray:
            result = Point(self.x - other[0], self.y - other[1], self.z - other[2])
        if type(other) == int or type(other) == float:
            result = Point(self.x - other, self.y - other, self.z - other)
        return result

    def __mul__(self, other: float | int | bool) -> Point:
        return Point(self.x * other, self.y * other, self.z * other)

    def __matmul__(self, other: Point) -> Point:
        return Point(self.x * other.x, self.y * other.y, self.z * other.z)

    def __truediv__(self, other) -> Point:
        return Point(self.x / other, self.y / other, self.z / other)

    def __repr__(self) -> str:
        return f'Point(x = {self.x}, y = {self.y}, z = {self.z})'

    def __eq__(self, other: Point) -> bool:
        return np.all([self.x == other.x, self.y == other.y, self.z == other.z])

    def __isub__(self, other) -> Point:
        return self.__sub__(other)

    def __iadd__(self, other) -> Point:
        return self.__add__(other)

    def __neg__(self):
        return Point(-self.x, -self.y, -self.z)

    def __len__(self):
        return 3

    def __hash__(self):
        return hash(tuple((self.x, self.y, self.z)))

    def __copy__(self) -> Point:
        return Point(self.x, self.y, self.z)

    def __abs__(self):
        return Point(np.abs(self.x), np.abs(self.y), np.abs(self.z))

    def manhattan_distance(self, other: Point) -> int:
        return np.abs(self.x - other.x) + np.abs(self.y - other.y)

    def euclidean_distance(self, other: Point) -> Vector | None:
        """
        Calculates the Euclidean distance (using the pixels as the smallest unit) on the x,y plane between this Point
        and another Point as long as the two Points lie along one of the 8 directions defined in the Direction class
        and return the Vector that defines this distance.
        :param other: The other Point.
        :return: The Vector specifying the Euclidean distance or None if the two Points do not align along a Direction
        """
        origin = copy(self)
        length = None
        orientation = None
        if self.y == other.y:
            length = np.abs(self.x - other.x)
            orientation = Orientation.Left if self.x > other.x else Orientation.Right
        elif self.x == other.x:
            orientation = Orientation.Up if self.y < other.y else Orientation.Down
            length = np.abs(self.y - other.y)
        elif np.sign(self.x - other.x) == np.sign(self.y - other.y):
            length = np.max([np.abs(self.x - other.x), np.abs(self.y - other.y)])
            if np.abs(self.x - other.x) == np.abs(self.y - other.y):
                orientation = Orientation.Up_Right if self.x < other.x else Orientation.Down_Left
            else:
                orientation = None
        elif np.sign(self.x - other.x) != np.sign(self.y - other.y):
            length = np.max([np.abs(self.x - other.x), np.abs(self.y - other.y)])
            if np.abs(self.x - other.x) == np.abs(self.y - other.y):
                orientation = Orientation.Down_Right if self.x < other.x else Orientation.Up_Left
            else:
                orientation = None

        if length is None:
            return None
        else:
            return Vector(orientation=orientation, length=int(length), origin=origin)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def point_from_numpy(array: np.ndarray):
        z = 0
        if len(array) == 3:
            z = array[2]
        return Point(array[0], array[1], z)

    def transform(self, affine_matrix: np.ndarray | None = None,
                  rotation: float = 0,
                  shear: List | np.ndarray | Point | None = None,
                  translation: List | np.ndarray | Point | Vector | None = None,
                  scale: List | np.ndarray | Point | None = None):

        assert affine_matrix is None or np.all([rotation == 0, shear is None, translation is None, scale is None])

        x = self.x
        y = self.y

        if affine_matrix is not None:
            a0 = affine_matrix[0, 0]
            a1 = affine_matrix[0, 1]
            a2 = affine_matrix[0, 2]
            b0 = affine_matrix[1, 0]
            b1 = affine_matrix[1, 1]
            b2 = affine_matrix[1, 2]

            self.x = a0 * x + a1 * y + a2
            self.y = b0 * x + b1 * y + b2
        else:
            if translation is not None:
                if type(translation) == Point:
                    translation = translation.to_numpy()
                if type(translation) == Vector:
                    temp_trans = []
                    if translation.orientation == Orientation.Up:
                        temp_trans = [0, translation.length]
                    elif translation.orientation == Orientation.Down:
                        temp_trans = [0, - translation.length]
                    if translation.orientation == Orientation.Right:
                        temp_trans = [translation.length, 0]
                    elif translation.orientation == Orientation.Left:
                        temp_trans = [-translation.length, 0]
                    if translation.orientation == Orientation.Up_Right:
                        temp_trans = [translation.length, translation.length]
                    elif translation.orientation == Orientation.Down_Right:
                        temp_trans = [translation.length, - translation.length]
                    if translation.orientation == Orientation.Up_Left:
                        temp_trans = [- translation.length, translation.length]
                    elif translation.orientation == Orientation.Down_Left:
                        temp_trans = [translation.length, - translation.length]
                    translation = temp_trans
                translation_x = int(translation[0])
                translation_y = int(translation[1])
            else:
                translation_x = 0
                translation_y = 0

            if scale is not None:
                if type(scale) == Point:
                    scale = scale.to_numpy()
                scale_x = scale[0] if type(scale) != Point else scale.x
                scale_y = scale[1] if type(scale) != Point else scale.y
            else:
                scale_x = 1
                scale_y = 1

            if shear is not None:
                if type(shear) == Point:
                    shear = shear.to_numpy()
                shear_x = shear[0] if type(shear) != Point else shear.x
                shear_y = shear[1] if type(shear) != Point else shear.y
            else:
                shear_x = 0
                shear_y = 0

            self.x = scale_x * x * (np.cos(rotation) + np.tan(shear_y) * np.sin(rotation)) - \
                     scale_y * y * (np.tan(shear_x) * np.cos(rotation) + np.sin(rotation)) + translation_x

            self.y = scale_x * x * (np.sin(rotation) - np.tan(shear_y) * np.cos(rotation)) - \
                     scale_y * y * (np.tan(shear_x) * np.sin(rotation) - np.cos(rotation)) + translation_y

            #self.x = int(self.x)
            #self.y = int(self.y)

    def copy(self) -> Point:
        return Point(self.x, self.y, self.z)


class Vector:
    def __init__(self, orientation: Orientation | None = Orientation.Up,
                 length: None | float = 0,
                 origin: Point = Point(0, 0, 0)):
        self.orientation = orientation
        self.length = length if length is not None else None
        self.origin = Point(origin.x, origin.y, origin.z)

    def __repr__(self):
        return f'Vector(Orientation: {self.orientation}, Length: {self.length}, Origin Point: {self.origin})'

    def __copy__(self):
        return Vector(orientation=copy(self.orientation), length=self.length, origin=copy(self.origin))

    def __eq__(self, other: Vector) -> bool:
        if self.length == other.length and self.origin == other.origin and self.orientation == other.orientation:
            return True
        else:
            return False

    def __mul__(self, mult: int) -> Vector:
        return Vector(orientation=self.orientation, length=self.length * mult, origin=self.origin)


    # TODO: Need to deal with transformations other than rotation
    def transform(self, affine_matrix: np.ndarray | None = None,
                  rotation: float = 0,
                  shear: List | np.ndarray | Point | None = None,
                  translation: List | np.ndarray | Point | None = None,
                  scale: List | np.ndarray | Point | None = None):
        assert affine_matrix is None or np.all([rotation == 0, shear is None, translation is None, scale is None])

        if rotation != 0:
            self.orientation = self.orientation.rotate(rotation=rotation)
        if affine_matrix is not None:
            self.orientation = self.orientation.rotate(affine_matrix=affine_matrix)

        self.origin.transform(affine_matrix, rotation, shear, translation, scale)

    @staticmethod
    def random(given_origin: Point | None = None,min_length: int = 0, max_length: int = 32,
               orientation_probs: List = (1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8)) -> Vector:
        orientation = Orientation.random(probs=orientation_probs)
        length = np.random.randint(min_length, max_length + 1)
        origin = given_origin if given_origin is not None else Point.random(min_x=0, min_y=0, min_z=0)

        return Vector(orientation=orientation, length=length, origin=origin)


class Bbox:
    def __init__(self, top_left: Point = Point(0, 0), bottom_right: Point = Point(0, 0)):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.center: Point = self._calculate_center()

    def __getattr__(self, center: str) -> Point:
        self.center = self._calculate_center()
        return self.center

    def __copy__(self):
        return Bbox(top_left=self.top_left, bottom_right=self.bottom_right)

    def __repr__(self):
        return f'Bbox(Top Left: {self.top_left}, Bottom Right: {self.bottom_right}, Center: {self.center})'

    def __eq__(self, other: Bbox) -> bool:
        if self.top_left == other.top_left and self.bottom_right == other.bottom_right:
            return True
        else:
            return False

    def _calculate_center(self):
        center = Point(x=(self.bottom_right.x - self.top_left.x) / 2 + self.top_left.x,
                       y=(self.bottom_right.y - self.top_left.y)/2 + self.top_left.y)
        return center

    def transform(self, affine_matrix: np.ndarray | None = None,
                  rotation: float = 0,
                  shear: List | np.ndarray | Point | None = None,
                  translation: List | np.ndarray | Point | None = None,
                  scale: List | np.ndarray | Point | None = None):
        assert affine_matrix is None or np.all([rotation == 0, shear is None, translation is None, scale is None])

        rot = 0
        if rotation == np.pi/2:
            rot = 1
        if affine_matrix is not None and np.all(affine_matrix[:2, :2].astype(int) == np.array([[0, -1], [1, 0]])):
            rot = 1
        if rotation == np.pi:
            rot = 2
        if affine_matrix is not None and np.all(affine_matrix[:2, :2].astype(int) == np.array([[-1, 0], [0, -1]])):
            rot = 2
        if rotation == 3*np.pi/2:
            rot = 3
        if affine_matrix is not None and np.all(affine_matrix[:2, :2].astype(int) == np.array([[0, 1], [-1, 0]])):
            rot = 3

        if rot == 1:
            p1 = copy(self.top_left)
            p1.transform(affine_matrix, rotation, shear, translation, scale)
            p2 = copy(self.bottom_right)
            p2.transform(affine_matrix, rotation, shear, translation, scale)
            self.top_left = Point.point_from_numpy(np.array([p1.x, p2.y, 0]))
            self.bottom_right = Point.point_from_numpy(np.array([p2.x, p1.y, 0]))

        elif rot == 2:
            p = copy(self.top_left)
            self.top_left = Point(self.bottom_right.x - 2*(self.bottom_right.x - self.top_left.x), self.top_left.x)
            self.bottom_right = Point(p.x, p.y - 2 * (p.y - self.bottom_right.y))

        elif rot == 3:
            p1 = copy(self.top_left)
            p1.transform(affine_matrix, rotation, shear, translation, scale)
            p2 = copy(self.bottom_right)
            p2.transform(affine_matrix, rotation, shear, translation, scale)

            self.top_left = Point.point_from_numpy(np.array([p2.x, p1.y, 0]))
            self.bottom_right = Point.point_from_numpy(np.array([p1.x, p2.y, 0]))

        else:
            self.top_left.transform(affine_matrix, rotation, shear, translation, scale)
            self.bottom_right.transform(affine_matrix, rotation, shear, translation, scale)

        self.center = self._calculate_center()

    def area(self) -> int | float:
        return (self.bottom_right.x - self.top_left.x) * (self.top_left.y - self.bottom_right.y)


class Colour:

    Transparent = 0
    Black = 1
    Blue = 2
    Red = 3
    Green = 4
    Yellow = 5
    Gray = 6
    Purple = 7
    Orange = 8
    Azure = 9
    Burgundy = 10
    Hole = 255

    map_int_to_colour = {0: [0, 0, 0, 0],  # Transparent / Mask
                         1: [0, 0, 0, 1],  # Black #000000
                         2: [0, 116 / 255, 217 / 255, 1],  # Blue #0074D9
                         3: [1, 65 / 255, 54 / 255, 1],  # Red #FF4136
                         4: [46 / 255, 204 / 255, 64 / 255, 1],  # Green #2ECC40
                         5: [1, 220 / 255, 0, 1],  # Yellow #FFDC00
                         6: [170 / 255, 170 / 255, 170 / 255, 1],  # Gray #AAAAAA
                         7: [240 / 255, 18 / 255, 190 / 255, 1],  # Purple #F012BE
                         8: [1, 133 / 255, 27 / 255, 1],  # Orange #FF851B
                         9: [127 / 255, 219 / 255, 1, 1],  # Azure #7FDBFF
                         10: [135 / 255, 12 / 255, 37 / 255, 1],  # Burgundy #870C25
                         255: [1, 1, 1, 1]}  # White (used for denoting holes)
    num_of_non_black_colours = 10
    num_of_all_colours = 11
    num_of_entries_in_map = 12
    colour_names = ['Black', 'Blue', 'Red', 'Green', 'Yellow', 'Gray', 'Purple', 'Orange', 'Azure', 'Burgundy']
    helper_names = ['Transparent', 'White']

    map_name_to_int = {'Transparent': 0,
                       'Black': 1,
                       'Blue': 2,
                       'Red': 3,
                       'Green': 4,
                       'Yellow': 5,
                       'Gray': 6,
                       'Purple': 7,
                       'Orange': 8,
                       'Azure': 9,
                       'Burgundy': 10,
                       'White': 255}

    map_int_to_name = {int(v): k for k, v in zip(map_name_to_int.keys(), map_name_to_int.values())}

    def __init__(self, colour_int: int | None = 1, colour_name: str | None = None, random: bool = False):

        self.index = int(colour_int) if colour_int is not None else int(self.map_name_to_int[colour_name])
        if random:
            self.index = int(np.random.randint(2, 11, 1)[0])

        self.name = self.map_int_to_name[self.index]
        self.colour = self.map_int_to_colour[self.index]

    @staticmethod
    def random(not_included: List[Colour] | List[int] | None = None, number: int = 1, replace: bool = True,) \
            -> Colour | List[Colour]:
        """
        Returns a Colour randomly sampled of the Colours (except the not_included) with a flat distribution. For more
        complicated distributions use the DistributionOver_Colours object.
        :param not_included: A List of either Colour or ints (indices of Colours) denoting Colours that should not be considered
        :param number: How many colours to return
        :param replace: With replacement or not
        :return: A Colour if number == 1 otherwise a List of Colours
        """
        if not_included is not None:
            if type(not_included[0]) == Colour:
                not_included = [c.index for c in not_included]
        else:
            not_included = []

        result = np.random.choice(a=np.setdiff1d([Colour.Black, Colour.Blue, Colour.Red, Colour.Green, Colour.Yellow,
                                                  Colour.Gray, Colour.Purple, Colour.Orange, Colour.Azure,
                                                  Colour.Burgundy], not_included), size=number, replace=replace)
        if len(result) == 1:
            return Colour(colour_int=result[0])
        else:
            return [Colour(colour_int=i) for i in result]
