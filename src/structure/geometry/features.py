
from functools import partial

from structure.geometry.basic_geometry import Orientation, Colour
from structure.object.primitives import Primitive, ObjectType


class UnaryLeftFeature(object):
    def __init__(self, func):
        self.func = func

    def __or__(self, other):
        return self.func(other)

    def __call__(self, v):
        return self.func(v)


class UnaryRightFeature(object):
    def __init__(self, func):
        self.func = func

    def __ror__(self, other):
        return self.func(other)

    def __call__(self, v):
        return self.func(v)


class BinaryFeature(object):
    def __init__(self, func):
        self.func = func

    def __or__(self, other):
        return self.func(other)

    def __ror__(self, other):
        return BinaryFeature(partial(self.func, other))

    def __call__(self, v1, v2):
        return self.func(v1, v2)


@UnaryLeftFeature
def dx(o: Primitive) -> int:
    return o.dimensions.dx


@UnaryLeftFeature
def dy(o: Primitive) -> int:
    return o.dimensions.dy


@UnaryLeftFeature
def pos_x(o: Primitive) -> int:
    return o.canvas_pos.x


@UnaryLeftFeature
def pos_y(o: Primitive) -> int:
    return o.canvas_pos.y


@UnaryLeftFeature
def pos_z(o: Primitive) -> int:
    return o.canvas_pos.z


@UnaryLeftFeature
def num_of_colours(o: Primitive) -> int:
    return len(o.get_used_colours())


@UnaryLeftFeature
def num_of_coloured_pixels(o: Primitive) -> int:
    return o.get_number_of_pixels_for_each_colour().sum()


@UnaryLeftFeature
def most_common_colour(o: Primitive) -> Colour:
    return Colour(colour_int=o.get_most_common_colour())


@UnaryRightFeature
def is_of_type(o: Primitive) -> ObjectType:
    return ObjectType(ObjectType.get_int_from_name(o.get_str_type()))


@UnaryLeftFeature
def type_of_primitive(o: Primitive) -> ObjectType:
    return ObjectType(ObjectType.get_int_from_name(o.get_str_type()))


@UnaryLeftFeature
def num_of_holes(o: Primitive) -> int:
    _, n_holes, size_of_holes = o.detect_holes()
    return n_holes


@BinaryFeature
def is_it_of_type(o: Primitive, prim_type: ObjectType | int) -> bool:
    int_type = ObjectType.get_int_from_name(o.get_str_type())
    if (type(prim_type) == int and prim_type == int_type) or \
        (type(prim_type) == ObjectType and ObjectType(ObjectType.get_int_from_name(o.get_str_type()))):
        return True

    return False


@BinaryFeature
def has_colour(o: Primitive, colour: int) -> bool:
    if colour in o.get_used_colours():
        return True
    return False


@BinaryFeature
def has_n_coloured_pixels_of_col(o: Primitive, colour: int) -> int:
    return o.get_number_of_pixels_for_each_colour()[colour]


@BinaryFeature
def is_along_x(o1: Primitive, o2: Primitive) -> bool:
    result = True if o1.is_object_along_x_to_object(o2) else False
    return result


@BinaryFeature
def is_along_y(o1: Primitive, o2: Primitive) -> bool:
    result = True if o1.is_object_along_y_to_object(o2) else False
    return result


@BinaryFeature
def is_along_xy(o1: Primitive, o2: Primitive) -> bool:
    result = True if o1.is_object_along_xy_to_object(o2) else False
    return result


@BinaryFeature
def is_along_xminusy(o1: Primitive, o2: Primitive) -> bool:
    result = True if o1.is_object_along_xminusy_to_object(o2) else False
    return result


@BinaryFeature
def is_over(o1: Primitive, o2: Primitive) -> bool:
    result = True if o1.is_object_over_object(o2) else False
    return result


@BinaryFeature
def is_under(o1: Primitive, o2: Primitive) -> bool:
    result = True if o1.is_object_under_object(o2) else False
    return result


@BinaryFeature
def is_left_of(o1: Primitive, o2: Primitive) -> bool:
    result = True if o1.is_object_left_of_object(o2) else False
    return result


@BinaryFeature
def is_right_of(o1: Primitive, o2: Primitive) -> bool:
    result = True if o1.is_object_right_of_object(o2) else False
    return result


@BinaryFeature
def touches(o1: Primitive, o2: Primitive) -> bool:
    result = True if o1.is_object_touching(o2) else False
    return result


@BinaryFeature
def touches_towards(o1: Primitive, o2: Primitive) -> Orientation | None:
    if o1 | touches | o2:

        if o1 | is_right_of | o2:
            if o1 | is_over | o2:
                return Orientation.Down_Left
            elif o1 | is_under | o2:
                return Orientation.Up_Left
            return Orientation.Left

        elif o1 | is_left_of | o2:
            if o1 | is_over | o2:
                return Orientation.Down_Right
            elif o1 | is_under | o2:
                return Orientation.Up_Right
            return Orientation.Right

        elif o1 | is_over | o2:
            return Orientation.Down
        elif o1 | is_under | o2:
            return Orientation.Up

        return None


@BinaryFeature
def overlaps(o1: Primitive, o2: Primitive) -> bool:
    return o2.is_object_overlapped(o1)


@BinaryFeature
def sublaps(o1: Primitive, o2: Primitive) -> bool:
    return o2.is_object_underlapped(o1)


