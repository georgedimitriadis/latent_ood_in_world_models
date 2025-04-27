
from copy import copy
from typing import List, Tuple, Any, Type
from itertools import combinations
import numpy as np

from structure.canvas.canvas import Canvas
from structure.geometry.basic_geometry import Point, Vector, Dimension2D, Surround, RelativePoint, Orientation, Colour
from structure.object.primitives import Primitive, Predefined, Random, Parallelogram, Cross, Hole, Pi, InverseCross, \
    Dot, Angle, Diagonal, Steps, Fish, Bolt, Tie, Spiral, Pyramid, Maze


# Math Funcs
def assign(a: int | float | bool | Point | Dimension2D | Primitive | Vector| Surround) -> \
        int | float | bool | Point | Dimension2D | Primitive | Vector| Surround:
    return a


def sum(a: int | float, b: int | float) -> int | float:
    return a + b


def subtract(a: int | float, b: int | float) -> int | float:
    return a - b


def multiply(a: int | float, b: int | float) -> int | float:
    return a * b


def divide(a: int | float, b: int | float) -> int | float:
    return a / b


def divide_to_int(a: int | float, b: int | float) -> int | float:
    return a // b


def modulo(a: int, b: int) -> int:
    return a % b


def sign(a: int) -> int:
    return np.sign(a)


def bigger_than(a: int | float, b: int | float) -> bool:
    return a > b


def bigger_than_or_equal(a: int | float, b: int | float) -> bool:
    return a >= b


def equal(a: int | float | Point | Dimension2D | Vector, b: int | float | Point | Dimension2D | Vector) -> bool:
    return a == b


def not_equal(a: int | float | Point | Dimension2D | Vector, b: int | float | Point | Dimension2D | Vector) -> bool:
    return a != b


def all_binary_combinations(array: List[Any]) -> List[Tuple[Any, Any]]:
    return list(combinations(array, 2))


def select_from_list(array: List, index: int) -> Any:
    return array[index]


def intersect(array_a: Any, array_b: Any) -> List:
    result = []
    array_a = array_a if isinstance(array_a, list) else [array_a]
    array_b = array_a if isinstance(array_b, list) else [array_b]
    for a in array_a:
        for b in array_b:
            if a == b:
                result.append(a)

    return result


def index_of_item_in_list(array: List, value: Any) -> int:
    return array.index(value)


#  Funcs on Structure (Points, Distance2D, Vector, etc)
def make_new_point(x: int, y: int, z:int = 0) -> Point:
    return Point(x, y, z)


def make_new_dimension2d(dx: int, dy: int) -> Dimension2D:
    return Dimension2D(dx, dy)


def make_new_orientation(towards: str | int) -> Orientation:
    if isinstance(towards, str):
        where = {'Up': 0, 'Up_Right': 1, 'Right': 2, 'Down_Right': 3, 'Down': 4, 'Down_Left': 5, 'Left': 6, 'Up_Left': 7}[towards]
    else:
        where = towards
    return Orientation(where)


def tuple_to_point(t: Tuple[int, int]) -> Point:
    return Point(t[0], t[1])


def point_to_tuple(p: Point) -> Tuple[int, int]:
    return p.x, p.y


def furthest_point_to_point(origin: Point, targets: List[Point] | Point) -> Tuple[Vector, int]:
    if type(targets) == Point:
        return origin.euclidean_distance(targets), 0
    result = targets[0]
    index = 0
    for i, t in enumerate(targets):
        if origin.euclidean_distance(result) is None:
            result = t
            index = i
        elif origin.euclidean_distance(t) is not None:
            if origin.euclidean_distance(t).length > origin.euclidean_distance(result).length:
                result = t
                index = i

    return origin.euclidean_distance(result), index


def closest_point_to_point(origin: Point, targets: List[Point] | Point) -> Tuple[Vector, int]:
    if type(targets) == Point:
        return origin.euclidean_distance(targets), 0
    result = targets[0]
    index = 0
    for i, t in enumerate(targets):
        if origin.euclidean_distance(result) is None:
            result = t
            index = i
        elif origin.euclidean_distance(t) is not None:
            if origin.euclidean_distance(t).length < origin.euclidean_distance(result).length:
                result = t
                index = i

    return origin.euclidean_distance(result), index


def sum_points(first: Point, second: Point) -> Point:
    f = copy(first)
    s = copy(second)
    return f + s


def subtract_points(first: Point, second: Point) -> Point:
    f = copy(first)
    s = copy(second)
    return f - s


def multiply_point(point: Point, mult: int | Point) -> Point:
    return copy(point) * mult


def multiply_point_x(point: Point, mult: int | Point) -> Point:
    return Point(point.x * mult, point.y)


def multiply_point_y(point: Point, mult: int | Point) -> Point:
    return Point(point.x, point.y * mult)


def mat_multiply_point(point: Point, other: Point | Dimension2D) -> Point:
    other = copy(other)
    other = Point(other.dx, other.dy) if isinstance(other, Dimension2D) else other
    return copy(point) @ copy(other)


def sum_dimensions(dimension: Dimension2D, other: Dimension2D) -> Dimension2D:
    return dimension + other


def multiply_dimension_x(dimension: Dimension2D, mult: int | float) -> Dimension2D:
    return Dimension2D(int(dimension.dx * mult), dimension.dy)


def multiply_dimension_y(dimension: Dimension2D, mult: int | float) -> Dimension2D:
    return Dimension2D(dimension.dx, int(dimension.dy * mult))


def multiply_dimension(dimension: Dimension2D, mult: int | float) -> Dimension2D:
    return Dimension2D(int(dimension.dx * mult), int(dimension.dy * mult))


def mat_multiply_dimension(dimension: Dimension2D, other: Dimension2D) -> Dimension2D:
    return copy(dimension) @ copy(other)


def modulo_point(point: Point, divisor: int) -> Point:
    return Point(point.x % divisor, point.y % divisor)


def modulo_point_x(point: Point, divisor: int) -> int:
    return point.x % divisor


def modulo_point_y(point: Point, divisor: int) -> int:
    return point.y % divisor


def points_in_line(start_point: Point, end_point: Point, cardinal_only: bool = False) -> Orientation | None:
    if start_point.x == end_point.x:
        if start_point.y > end_point.y:
            return Orientation.Down
        elif end_point.y > start_point.y:
            return Orientation.Up
    elif start_point.y == end_point.y:
        if start_point.x > end_point.x:
            return Orientation.Left
        elif start_point.x < end_point.x:
            return Orientation.Right
    if not cardinal_only:
        if np.abs(start_point.x - end_point.x) == np.abs(start_point.y - end_point.y):
            if start_point.x > end_point.x and start_point.y > end_point.y:
                return Orientation.Down_Left
            elif start_point.x < end_point.x and start_point.y > end_point.y:
                return Orientation.Down_Right
            elif start_point.x < end_point.x and start_point.y < end_point.y:
                return Orientation.Up_Right
            elif start_point.x > end_point.x and start_point.y < end_point.y:
                return Orientation.Up_Left

    return None


def all_points_between_two_points(start_point: Point, end_point: Point, cardinal_only: bool = False) -> List[Point]:
    all_points_in_line = []
    dir = points_in_line(start_point, end_point, cardinal_only)
    if dir is not None:
        if dir == Orientation.Up:
            for i in range(start_point.y, end_point.y + 1):
                all_points_in_line.append(Point(start_point.x, i, start_point.z))
        if dir == Orientation.Down:
            for i in range(start_point.y, end_point.y - 1, -1):
                all_points_in_line.append(Point(start_point.x, i, start_point.z))
        if dir == Orientation.Left:
            for i in range(start_point.x, end_point.x - 1, - 1):
                all_points_in_line.append(Point(i, start_point.y, start_point.z))
        if dir == Orientation.Right:
            for i in range(start_point.x, end_point.x + 1):
                all_points_in_line.append(Point(i, start_point.y, start_point.z))

        if not cardinal_only:
            if dir == Orientation.Up_Left:
                for i in range(0, end_point.y - start_point.y + 1):
                    all_points_in_line.append(Point(start_point.x - i, start_point.y + i))
            if dir == Orientation.Up_Right:
                for i in range(0, end_point.y - start_point.y + 1):
                    all_points_in_line.append(Point(start_point.x + i, start_point.y + i))
            if dir == Orientation.Down_Left:
                for i in range(0, start_point.y - end_point.y + 1):
                    all_points_in_line.append(Point(start_point.x - i, start_point.y - i))
            if dir == Orientation.Down_Right:
                for i in range(0, start_point.y - end_point.y + 1):
                    all_points_in_line.append(Point(start_point.x + i, start_point.y - i))

    return all_points_in_line


def make_new_vector(orientation: Orientation, length: int, origin: Point) -> Vector:
    return Vector(orientation=orientation, length=length, origin=origin)


def get_length_of_vector(v: Vector) -> int:
    return v.length


def get_orientation_of_vector(v: Vector) -> Orientation:
    return v.orientation


def get_origin_of_vector(v: Vector) -> Point:
    return v.origin


def multiply_vector(v: Vector, mult: int) -> Vector:
    return v * mult


#  Funcs on Canvasses
def copy_canvas(canvas: Canvas) -> Canvas:
    return copy(canvas)


def copy_object(obj: Primitive) -> Primitive:
    return copy(obj)


def make_new_canvas_as(canvas: Canvas) -> Canvas:
    if canvas.grid:
        new_canvas = Canvas(as_grid_x_y_tilesize_colour=(canvas.grid_shape[0], canvas.grid_shape[1],
                                                         canvas.size_of_tiles, canvas.grid_colour))
    else:
        new_canvas = Canvas(size=canvas.size)
    return new_canvas


def make_new_canvas(size: Dimension2D) -> Canvas:
    return Canvas(size=size)


def get_canvas_feature_size(canvas: Canvas) -> Dimension2D:
    return canvas.size


def get_canvas_feature_size_x(canvas: Canvas) -> int:
    return canvas.size.dx


def get_canvas_feature_size_y(canvas: Canvas) -> int:
    return canvas.size.dy


def get_canvas_feature_all_object_colours(canvas: Canvas) -> List[int]:
    return canvas.get_used_colours()


def get_canvas_feature_grid_colour(canvas: Canvas) -> int | None:
    if canvas.grid:
        return canvas.grid_colour
    return None


def get_canvas_feature_grid_tile_size(canvas: Canvas) -> int | None:
    if canvas.grid:
        return canvas.size_of_tiles
    return None


def get_colour_common_to_all_objects(canvas: Canvas) -> int | List[int] | None:
    common_colours = list(canvas.objects[0].get_used_colours())
    for o in canvas.objects[1:]:
        common_colours = list(np.intersect1d(common_colours, list(o.get_used_colours())))

    if len(common_colours) == 0:
        return None
    elif len(common_colours) == 1:
        return common_colours[0]

    return common_colours


def add_object_to_canvas(canvas: Canvas, obj: Primitive) -> Canvas:
    new_canvas = copy(canvas)
    new_obj = copy(obj)
    new_canvas.add_new_object(new_obj)
    return new_canvas


def canvas_transform_split_object_by_colour_on_canvas(canvas: Canvas, obj: Primitive) -> Canvas:
    new_canvas = copy(canvas)
    new_canvas.split_object_by_colour(obj)
    return new_canvas


def canvas_transform_and_objects(canvas: Canvas, obj_a: Primitive, obj_b: Primitive, new_colour: Colour,
                                 canvas_pos: Point = Point(0, 0)) -> Primitive:
    canvas = copy(canvas)
    new_object = canvas.and_objects(obj_a, obj_b, new_colour, canvas_pos)

    return new_object


def get_tile_from_canvas_pos(canvas: Canvas, pixel: Point) -> Tuple[int, int] | None:
    for k in canvas.grid_tiles_coordinates:
        point = Point(pixel.x, pixel.y, 0)
        if canvas.grid_tiles_coordinates[k] == point:
            return k
    return None


def get_canvas_pos_from_tile(canvas: Canvas, tile: Tuple[int, int]) -> Point:
    return canvas.grid_tiles_coordinates[tile]


# Funcs to get Primitive features
def is_of_type(obj: Primitive, primitive_type: Any) -> bool:
    return type(obj) == primitive_type


def get_distance_min_between_objects(first: Primitive, second: Primitive) -> Vector | None:
    return first.get_distance_to_object(other=second, dist_type='min')


def get_distance_max_between_objects(first: Primitive, second: Primitive) -> Vector | None:
    return first.get_distance_to_object(other=second, dist_type='max')


def get_distance_origin_to_origin_between_objects(first: Primitive, second: Primitive) -> Vector | None:
    return first.get_distance_to_object(other=second, dist_type='canvas_pos')


def get_distance_touching_between_objects(first: Primitive, second: Primitive) -> Vector | None:
    dist = first.get_distance_to_object(other=second, dist_type='straight_line')
    if dist is not None:
        dist.length -= 1
        return dist
    return None


def get_along_x_distance_between_objects(first: Primitive, second: Primitive) -> Vector:
    orientation = Orientation.Left if first.canvas_pos.x > second.canvas_pos.x else Orientation.Right
    origin = first.canvas_pos
    length = np.abs(second.canvas_pos.x - first.canvas_pos.x).astype(int)
    return Vector(orientation=orientation, length=length, origin=origin)


def get_along_y_distance_between_objects(first: Primitive, second: Primitive) -> Vector:
    orientation = Orientation.Down if first.canvas_pos.y > second.canvas_pos.y else Orientation.Up
    origin = first.canvas_pos
    length = np.abs(second.canvas_pos.y - first.canvas_pos.y).astype(int)
    return Vector(orientation=orientation, length=length, origin=origin)


def get_point_for_match_shape_furthest(background_obj: Primitive, filter_obj: Primitive,
                                       match_shape_only: bool, try_unique: bool = True,
                                       padding: Surround = Surround(0, 0, 0, 0),
                                       transformations: List[str] = ('rotate', 'scale', 'flip', 'invert', 'colour')) -> Point:
    result = filter_obj.match_to_background(background_obj, match_shape_only=match_shape_only, try_unique=try_unique,
                                            padding=padding, transformations=transformations)
    match_positions = []
    for r in result:
        if r['rotate'] is None and r['scale'] is None:
            match_positions.append(r['translate_to_coordinates'])
    match_positions = match_positions[0]
    _, index = furthest_point_to_point(filter_obj.canvas_pos, match_positions)

    return match_positions[index]


def get_point_and_rotation_for_match_shape_furthest(background_obj: Primitive, filter_obj: Primitive,
                                                    match_shape_only: bool, try_unique: bool = True,
                                                    padding: Surround = Surround(0, 0, 0, 0)) -> Tuple[Point, int]:
    result = filter_obj.match_to_background(background_obj, match_shape_only=match_shape_only, try_unique=try_unique, padding=padding)
    match_positions = [result[i]['canvas_pos'][0] for i in range(len(result))]
    rotations = [result[i]['rotation'] for i in range(len(result))]
    _, index = furthest_point_to_point(filter_obj.canvas_pos, match_positions)

    return match_positions[index], rotations[index]


def get_point_for_match_shape_nearest(background_obj: Primitive, filter_obj: Primitive,
                                      match_shape_only: bool, try_unique:bool = True,
                                      padding: Surround = Surround(0, 0, 0, 0)) -> Point:
    result = filter_obj.match_to_background(background_obj, match_shape_only=match_shape_only, try_unique=try_unique, padding=padding)
    rotation = 0
    scale = 1
    match_positions = []
    for r in result:
        if r['rotation'] == rotation and r['scale'] == scale:
            match_positions.append(r['canvas_pos'])
    match_positions = match_positions[0]

    _, index = closest_point_to_point(filter_obj.canvas_pos, match_positions)

    return match_positions[index]


def get_point_and_rotation_for_match_shape_nearest(background_obj: Primitive, filter_obj: Primitive,
                                                   match_shape_only: bool, try_unique:bool = True,
                                                   padding: Surround = Surround(0, 0, 0, 0)) -> \
        tuple[int | list[Point], int | list[Point]]:
    result = filter_obj.match_to_background(background_obj, match_shape_only=match_shape_only, try_unique=try_unique, padding=padding)
    match_positions = [result[i]['canvas_pos'] for i in range(len(result))]
    rotations = [result[i]['rotation'] for i in range(len(result))]
    _, index = closest_point_to_point(filter_obj.canvas_pos, match_positions)

    return match_positions[index], rotations[index]


def get_point_and_rotation_for_best_match_to_objects(object_to_move: Primitive, target_objects: List[Primitive],
                                                     match_shape_only: bool = False) -> Tuple[int, Point]:
    best_match = 0
    best_rotate = 0
    best_translate_to_coordinates = None

    for to in target_objects:
        result = object_to_move.match_to_background(to, match_shape_only=match_shape_only, try_unique=True,
                                                    padding=Surround(0, 0, 0, 0), transformations=['rotate'])
        if result[0]['result'] > best_match:
            best_match = result[0]['result']
            best_rotate = result[0]['rotate']
            best_translate_to_coordinates = result[0]['translate_to_coordinates'][0]

    return best_rotate, best_translate_to_coordinates


def get_random_colours(not_included: List[Colour], number: int = 1, replace: bool = True) -> List[Colour] | Colour:
    return Colour.random(not_included=not_included, number=number, replace=replace)


def get_object_feature_colour_at_position(obj: Primitive, pos: Point) -> int:
    return int(obj.actual_pixels[int(pos.y - obj.canvas_pos.y), int(pos.x - obj.canvas_pos.x)])


def get_object_feature_colour(obj: Primitive) -> int:
    return obj.colour


def get_object_feature_all_colours(obj: Primitive) -> List[int]:
    return list(obj.get_used_colours())


def get_object_feature_size(obj: Primitive) -> Dimension2D:
    return obj.dimensions


def get_object_feature_size_x(obj: Primitive) -> int:
    return obj.dimensions.dx


def get_object_feature_size_y(obj: Primitive) -> int:
    return obj.dimensions.dy


def get_object_feature_canvas_pos(obj: Primitive) -> Point:
    return obj.canvas_pos


def get_object_feature_canvas_pos_x(obj: Primitive) -> int:
    return obj.canvas_pos.x


def get_object_feature_canvas_pos_y(obj: Primitive) -> int:
    return obj.canvas_pos.y


def get_object_feature_coloured_positions(obj: Primitive) -> List[Point]:
    pos = obj.get_coloured_pixels_positions()
    result = []
    for p in pos:
        result.append(Point(p[1], p[0]))

    return result


def get_object_feature_number_of_colours(obj: Primitive) -> int:
    return len(obj.get_coloured_pixels_positions())


def get_object_feature_relative_point_position(obj: Primitive, relative_point: RelativePoint) -> Point:
    return obj.relative_points[relative_point]


def get_object_feature_position_of_colour(obj: Primitive, colour: int) -> Point | List[Point] | None:
    positions = obj.get_coloured_pixels_positions(colour)
    if len(positions) == 0:
        return None
    elif len(positions) == 1:
        return Point(x=positions[0][1], y=positions[0][0], z=obj.canvas_pos.z)
    else:
        point_positions = []
        for p in positions:
            point_positions.append(Point(p[1], p[0], obj.canvas_pos.z))
        return point_positions


def get_object_feature_least_used_colour(obj: Primitive) -> int | List[int]:
    colours = list(obj.get_used_colours())
    amounts_of_colours = []
    for c in colours:
        amounts_of_colours.append(len(obj.get_coloured_pixels_positions(c)))

    least_colours = list(np.array(colours)[np.where(amounts_of_colours == np.min(amounts_of_colours))[0]])

    if len(least_colours) == 1:
        return least_colours[0]

    return least_colours


def get_object_feature_most_used_colour(obj: Primitive) -> int | List[int]:
    colours = list(obj.get_used_colours())
    amounts_of_colours = []
    for c in colours:
        amounts_of_colours.append(len(obj.get_coloured_pixels_positions(c)))

    most_colours = list(np.array(colours)[np.where(amounts_of_colours == np.max(amounts_of_colours))[0]])

    if len(most_colours) == 1:
        return most_colours[0]

    return most_colours


# Funcs to select Primitives
def select_all_objects(canvas: Canvas) -> List[Primitive]:
    return copy(canvas.objects)


def select_object_with_canvas_pos(canvas: Canvas, canvas_pos: Point) -> Primitive | None:
    for o in canvas.objects:
        if o.canvas_pos == canvas_pos:
            return copy(o)
    return None


def select_largest_object_by_area(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='area')[-1]


def select_largest_object_by_height(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='height')[-1]


def select_largest_object_by_width(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='width')[-1]


def select_smallest_object_by_area(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='area')[0]


def select_smallest_object_by_height(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='height')[0]


def select_smallest_object_by_width(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='width')[0]


def select_object_with_the_most_colours(canvas: Canvas) -> Primitive:
    objects = select_all_objects(canvas)
    n = 0
    good_object = None
    for o in objects:
        nn = get_object_feature_number_of_colours(o)
        if nn > n:
            n = nn
            good_object = copy(o)

    return good_object


def select_object_with_the_fewer_colours(canvas: Canvas) -> Primitive:
    objects = select_all_objects(canvas)
    n = 100
    good_object = None
    for o in objects:
        nn = get_object_feature_number_of_colours(o)
        if nn < n:
            n = nn
            good_object = copy(o)

    return good_object


def select_rest_of_the_objects(canvas: Canvas, obj: Primitive | List[Primitive] | None) -> List[Primitive]:
    temp_obj_list = [copy(o) for o in canvas.objects]
    if isinstance(obj, Primitive):
        temp_obj_list.remove(obj)
    if isinstance(obj, List):
        for o in obj:
            temp_obj_list.remove(o)

    return temp_obj_list


def select_all_objects_of_colour(canvas: Canvas, colour: int) -> List[Primitive]:
    new_canvas = copy(canvas)
    return new_canvas.find_objects_of_colour(colour)


def select_only_object_of_colour(canvas: Canvas, colour: int) -> Primitive | None:
    all_objects = select_all_objects_of_colour(canvas, colour=colour)
    if len(all_objects) > 0:
        return all_objects[0]
    else:
        return None


def select_objects_of_type(canvas: Canvas, primitive_type: type[Primitive]) -> List[Primitive]:
    new_canvas = copy(canvas)
    objs_of_type = []
    for obj in new_canvas.objects:
        if isinstance(obj, primitive_type):
            objs_of_type.append(copy(obj))

    return objs_of_type


def group_objects_according_to_colour(canvas: Canvas) -> Tuple[List[int], List[List[Primitive]]]:
    num_of_objects_per_group = []
    objects_in_a_group = []
    colours = get_canvas_feature_all_object_colours(canvas)
    for c in colours:
        objects_in_a_group.append(select_all_objects_of_colour(canvas, c))
        num_of_objects_per_group.append(len(objects_in_a_group[-1]))

    return num_of_objects_per_group, objects_in_a_group


# Funcs to transform Primitives
def object_transform_rotate(obj: Primitive, rotation: int) -> Primitive:
    new_obj = copy(obj)
    new_obj.rotate(times=rotation)
    return new_obj


def object_transform_translate_to_point(obj: Primitive, target_point: Point,
                                        object_point: Point | None = None) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_to_coordinates(target_point=target_point, object_point=object_point)
    return new_obj


def object_transform_change_depth(obj: Primitive, target_depth: int) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_to_coordinates(target_point=Point(obj.canvas_pos.x,
                                                        obj.canvas_pos.y,
                                                        target_depth),
                                     object_point=obj.canvas_pos)
    return new_obj


def object_transform_translate_by_distance(obj: Primitive, distance: Dimension2D) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_by(distance=distance)
    return new_obj


def object_transform_translate_along_direction(obj: Primitive, direction: Vector) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_along(direction=direction)
    return new_obj


def object_transform_translate_relative_point_to_point(obj: Primitive, relative_point: RelativePoint,
                                                       other_point: Point) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_relative_point_to_point(relative_point=relative_point, other_point=other_point)
    return new_obj


def object_transform_translate_to_front_of_all(canvas: Canvas, obj: Primitive) -> Primitive:
    new_obj = copy(obj)
    max_z = 0
    for o in canvas.objects:
        if max_z < o.canvas_pos.z:
            max_z = o.canvas_pos.z

    new_obj.translate_to_coordinates(Point(new_obj.canvas_pos.x, new_obj.canvas_pos.y, max_z + 1))

    return new_obj


def object_transform_translate_to_back_of_all(canvas: Canvas, obj: Primitive) -> Primitive:
    new_obj = copy(obj)
    max_z = 1000
    for o in canvas.objects:
        if max_z < o.canvas_pos.z:
            max_z = o.canvas_pos.z

    new_obj.translate_to_coordinates(Point(new_obj.canvas_pos.x, new_obj.canvas_pos.y, max_z - 1))

    return new_obj


def object_transform_translate_to_front_of_object(obj: Primitive, other: Primitive) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_to_coordinates(Point(new_obj.canvas_pos.x, new_obj.canvas_pos.y, other.canvas_pos.z + 1))

    return new_obj


def object_transform_translate_to_back_of_object(obj: Primitive, other: Primitive) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_to_coordinates(Point(new_obj.canvas_pos.x, new_obj.canvas_pos.y, other.canvas_pos.z - 1))

    return new_obj


def object_transform_mirror(obj: Primitive, axis: Orientation):
    new_obj = copy(obj)
    new_obj.mirror(axis=axis, on_axis=False)
    return new_obj


def object_transform_mirror_on_axis(obj: Primitive, axis: Orientation):
    new_obj = copy(obj)
    new_obj.mirror(axis=axis, on_axis=True)
    return new_obj


def object_transform_flip_only(obj: Primitive, axis: Orientation | Vector):
    new_obj = copy(obj)
    if isinstance(axis, Vector):
        axis = axis.orientation
    new_obj.flip(axis=axis, translate=False)
    return new_obj


def object_transform_flip_and_translate(obj: Primitive, axis: Orientation | Vector):
    new_obj = copy(obj)
    if isinstance(axis, Vector):
        axis = axis.orientation
    new_obj.flip(axis=axis, translate=True)
    return new_obj


def object_transform_new_colour(obj: Primitive, colour: int) -> Primitive:
    new_obj = copy(obj)
    new_obj.set_new_colour(new_colour=colour)
    return new_obj


def object_transform_negate(obj: Primitive) -> Primitive:
    new_obj = copy(obj)
    new_obj.negate_colour()
    return  new_obj


def object_transform_delete_colour(obj: Primitive, colour: int) -> Primitive:
    new_obj = copy(obj)
    new_obj.actual_pixels[np.where(new_obj.actual_pixels == colour)] = 1
    return new_obj


def object_transform_fill_holes(obj: Primitive, colour: int) -> Primitive:
    o = copy(obj)
    o.fill_holes(colour)
    return o


def object_transform_split_object_along_axis(obj: Primitive, cut_orientation: Orientation, percentage: float | None = None,
                                             pixels: int | None = None) -> Tuple[Primitive, Primitive]:
    ob_a, ob_b = obj.split_object_along_axis(cut_orientation=cut_orientation, percentage=percentage, pixels=pixels)
    predef_a = Predefined(actual_pixels=copy(ob_a.actual_pixels))
    predef_a.canvas_pos = ob_a.canvas_pos
    predef_b = Predefined(actual_pixels=copy(ob_b.actual_pixels))
    predef_b.canvas_pos = ob_b.canvas_pos

    return predef_a, predef_b


def object_transform_split_object_in_quarters(obj: Primitive, round_to_include: bool = True) -> Tuple[Primitive,
                                                                                                      Primitive,
                                                                                                      Primitive,
                                                                                                      Primitive]:

    ul_obj, ur_obj, dl_obj, dr_obj = obj.split_object_in_quarters(round_to_include=round_to_include)

    ul = Predefined(actual_pixels=ul_obj.actual_pixels)
    ul.canvas_pos = ul_obj.canvas_pos
    ur = Predefined(actual_pixels=ur_obj.actual_pixels)
    ur.canvas_pos = ur_obj.canvas_pos
    dl = Predefined(actual_pixels=dl_obj.actual_pixels)
    dl.canvas_pos = dl_obj.canvas_pos
    dr = Predefined(actual_pixels=dr_obj.actual_pixels)
    dr.canvas_pos = dr_obj.canvas_pos

    return ul, ur, dl, dr


def object_transform_split_object_by_colour(obj: Primitive) -> List[Primitive]:
    temp_canvas = Canvas(size=obj.dimensions)
    o = copy(obj)
    temp_canvas.add_new_object(o)
    temp_canvas.split_object_by_colour(o)
    objects = [ob for ob in temp_canvas.objects]
    return objects


def object_transform_add_two_objects(obj_a: Primitive, obj_b: Primitive) -> Primitive:
    return copy(obj_a + obj_b)


# Funcs to order objects
def order_objects_according_to_height(objects: List[Primitive], reverse: bool = False) -> List[Primitive]:
    heights = []
    for o in objects:
        heights.append(o.dimensions.dy)
    indices = list(np.argsort(heights)) if not reverse else list(reversed(np.argsort(heights)))

    return list(np.array(objects)[indices])


# Funcs to make Primitives
def make_new_random(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                    canvas_pos: Point = Point(0, 0), colour: None | int = None, occupancy_prob: float = 0.5,
                    required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                    _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Random(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                  occupancy_prob=occupancy_prob, required_dist_to_others=required_dist_to_others, _id=_id,
                  actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_parallelogram(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                           canvas_pos: Point = Point(0, 0), colour: None | int = None,
                           required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                           _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Parallelogram(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                         required_dist_to_others=required_dist_to_others, _id=_id,
                         actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_cross(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                   canvas_pos: Point = Point(0, 0), colour: None | int = None,
                   required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                   _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Cross(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                 required_dist_to_others=required_dist_to_others, _id=_id,
                 actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_hole(size: Dimension2D | np.ndarray | List, thickness: Surround = Surround(1, 1, 1, 1),
                  border_size: Surround = Surround(0, 0, 0, 0), canvas_pos: Point = Point(0, 0),
                  colour: None | int = None, required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                  _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Hole(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour, thickness=thickness,
                required_dist_to_others=required_dist_to_others, _id=_id,
                actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_pi(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                canvas_pos: Point = Point(0, 0), colour: None | int = None,
                required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Pi(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
              required_dist_to_others=required_dist_to_others, _id=_id,
              actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_inverse_cross(height: int, border_size: Surround = Surround(0, 0, 0, 0),
                           canvas_pos: Point = Point(0, 0), colour: None | int = None,
                           required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                           fill_colour: None | int = None, fill_height: None | int = None,
                           _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return InverseCross(height=height, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                        required_dist_to_others=required_dist_to_others, _id=_id,
                        fill_colour=fill_colour, fill_height=fill_height,
                        actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_dot(border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Dot(border_size=border_size, canvas_pos=canvas_pos, colour=colour,
               required_dist_to_others=required_dist_to_others, _id=_id,
               actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_angle(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                   canvas_pos: Point = Point(0, 0), colour: None | int = None,
                   required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                   _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Angle(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                 required_dist_to_others=required_dist_to_others, _id=_id,
                 actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_diagonal(height: int, border_size: Surround = Surround(0, 0, 0, 0),
                      canvas_pos: Point = Point(0, 0), colour: None | int = None,
                      required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                      _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Diagonal(height=height, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                    required_dist_to_others=required_dist_to_others, _id=_id,
                    actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_steps(height: int, depth: int, border_size: Surround = Surround(0, 0, 0, 0),
                   canvas_pos: Point = Point(0, 0), colour: None | int = None,
                   required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                   _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Steps(height=height, depth=depth, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                 required_dist_to_others=required_dist_to_others, _id=_id,
                 actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_fish(border_size: Surround = Surround(0, 0, 0, 0),
                  canvas_pos: Point = Point(0, 0), colour: None | int = None,
                  required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                  _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Fish(border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                required_dist_to_others=required_dist_to_others, _id=_id,
                actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_bolt(_center_on: bool = False, border_size: Surround = Surround(0, 0, 0, 0),
                  canvas_pos: Point = Point(0, 0), colour: None | int = None,
                  required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                  _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Bolt(_center_on=_center_on, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                required_dist_to_others=required_dist_to_others, _id=_id,
                actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_tie(border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Tie(border_size=border_size, canvas_pos=canvas_pos, colour=colour,
               required_dist_to_others=required_dist_to_others, _id=_id,
               actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_spiral(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                    canvas_pos: Point = Point(0, 0), colour: None | int = None, gap: int = 1,
                    required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                    _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Spiral(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                  required_dist_to_others=required_dist_to_others, _id=_id, gap=gap,
                  actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_pyramid(height: int, border_size: Surround = Surround(0, 0, 0, 0),
                     canvas_pos: Point = Point(0, 0), colour: None | int = None,
                     required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                     _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Pyramid(height=height, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                   required_dist_to_others=required_dist_to_others, _id=_id,
                   actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_maze(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                  colour: None | int = None, required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                  _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Maze(size=size, border_size=border_size, colour=colour, required_dist_to_others=required_dist_to_others,
                _id=_id, actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)
