
from typing import List, Tuple
import numpy as np

from src.experiments.data.generation.arc_logic_task_generator import ARCLogicTask
from src.experiments.data.generation.random_primitive_object_spawner import RandomPrimitiveInCanvas
from src.structure.canvas.canvas import Canvas
from src.structure.geometry.basic_geometry import Colour, Dimension2D, Orientation, Surround, Point
from src.structure.geometry.probabilities import UniformDistribution, DistributionOver_ObjectTypes
from dsl.functions import dsl_functions as dsl
from src.structure.object.primitives import Primitive

# These are the parameters of the compositionality experiments. Currently, the colour_changes and scales parameters
# are not used.
small_dim = 3
large_dim = 7
sizes = {'small': Dimension2D(small_dim, small_dim), 'large': Dimension2D(large_dim, large_dim)}
types = {'parallelogram': 'parallelogram', 'cross': 'cross', 'angle': 'angle', 'pyramid': 'pyramid'}
colours = {'red': Colour.Red, 'blue': Colour.Blue}
translations = {'up': Orientation.Up, 'left': Orientation.Left}
colour_changes = {'green': Colour.Green, 'yellow': Colour.Yellow}
scales = {'shrinking': -2, 'expanding': 2}
rotations = {'90': 1, '180': 2}


def add_noise_to_coloured_pixels(input_canvas_data: np.ndarray, output_canvas_data: np.ndarray,
                                 colours_to_add_noise: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    This is used to add some noise to some coloured pixels (depending on colour)
    :param input_canvas_data: The array of the input data (canvas x 32 x 32)
    :param output_canvas_data: The array of the output data (canvas x 32 x 32)
    :param colours_to_add_noise: Which colours should the noise be added to
    :return: The tuple of the input and output arrays
    """
    for colour in colours_to_add_noise:
        colour_noise = Colour.Red + np.random.randint(-9, 9) / 100
        input_colour_indices = input_canvas_data == colour
        input_canvas_data[input_colour_indices] = colour_noise
        output_colour_indices = output_canvas_data == colour
        output_canvas_data[output_colour_indices] = colour_noise

    return input_canvas_data, output_canvas_data


def get_neighbouring_object_to_add(initial_obj: Primitive):
    ini_canvas_pos = initial_obj.canvas_pos
    new_canvas_pos = Point(0,0)
    if ini_canvas_pos.x > 5:
        new_canvas_pos.x = np.random.randint(0, ini_canvas_pos.x - 4)
    else:
        new_canvas_pos.x = np.random.randint(ini_canvas_pos.x + initial_obj.size.dx + 1, ini_canvas_pos.x + initial_obj.size.dx + 5)
    if ini_canvas_pos.y > 5:
        new_canvas_pos.y = np.random.randint(0, ini_canvas_pos.y - 4)
    else:
        new_canvas_pos.y = np.random.randint(ini_canvas_pos.y + initial_obj.size.dy + 1, ini_canvas_pos.y + initial_obj.size.dy + 5)

    obj = dsl.make_new_pi(Dimension2D(3, 4), canvas_pos=new_canvas_pos, colour=Colour.Yellow)

    return obj


def generate_compositional_datasets(num_of_outputs: int = 10, distance: int = 0, symmetric_objects: bool = True,
                                    transformation_type: str = 'translate', second_object: bool = False) -> ARCLogicTask:
    """
    This task generates examples using only type, colour and action as Concept Variables
    and object size and object position as a free Variables (so only 5 out of the 6 available Variables).

    For transformation_type = translate and symmetric_objects = True:
    The training Concept Groups (distance = 0) are 1) Square, Red, Moving Up, (000) 2) Cross, Red, Moving Up, (100)
    3) Square, Blue, Moving up (010) and 4) Square, Red, Moving Left (001).
    The test Concept Groups (distance = 1) are 1) Cross, Blue, Moving Up, (110) 2) Crosse, Red, Moving Left, (101)
    and 3) Square, Blue, Moving Left (011) and distance = 2 are (111) Cross, Blue, Moving Left

    For transformation_type = rotate nd symmetric_objects = False:
    The training Concept Groups (distance = 0) are 1) Pyramid, Red, Rotate 90, (000) 2) Angle, Red, Rotate 90, (100)
    3) Pyramid, Blue, Rotate 90 (010) and 4) Pyramid, Red, Rotate 180 (001).
    The test Concept Groups (distance = 1) are 1) Angle, Blue, Rotate 90, (110) 2) Angle, Red, Rotate 180, (101)
    and 3) Pyramid, Blue, Rotate 180 (011) and distance = 2 are (111) Angle, Blue, Rotate 180

    There are other combinations of actions and symmetric_objects possible but the above two are used in the paper

    :param num_of_outputs: Number of outputs.
    :param distance: The distance to the train data. Can be 0, 1 or 2.
    :param symmetric_objects: Use symmetric objects (i.e parallelogram and cross) or not (i.e. pyramid and angle).
    :param transformation_type: This is a string that can be 'translate', 'colour change', 'scale' or 'rotate' and this will define the transformation that the action will perform. translate will move the object 5 pixels up or 5 pixels left. rotate will rotate the object either 90 or 180 degrees. colour change will change the object's colour to either green or yellow and scale will either shrink or expand the object by by a factor of two.
    :param second_object: If True then a yellow Pi is added the canvas somewhere close to the initial object
    :return: The task with the correct input, output canvas pairs and their descriptions
    """

    canvas_size_y_distributions = UniformDistribution(range=(15, 32))
    canvas_size_x_distributions = UniformDistribution(range=(15, 32))
    canvas_sizes = [Dimension2D(canvas_size_y_distributions.sample(), canvas_size_x_distributions.sample()) for _
                    in range(num_of_outputs)]

    def get_transformation_value_str(for_0_or_1: int) -> str:
        transformation_value_str = ''
        if for_0_or_1 == 0:
            if transformation_str == 'translate':
                transformation_value_str = 'up'
            if transformation_str == 'colour change':
                transformation_value_str = colour_changes['green']
            if transformation_str == 'scale':
                transformation_value_str = scales['shrinking']
            if transformation_str == 'rotate':
                transformation_value_str = '90'
        if for_0_or_1 == 1:
            if transformation_str == 'translate':
                transformation_value_str = 'left'
            if transformation_str == 'colour change':
                transformation_value_str = colour_changes['yellow']
            if transformation_str == 'scale':
                transformation_value_str = scales['expanding']
            if transformation_str == 'rotate':
                transformation_value_str = '180'

        return transformation_value_str

    possible_types = ['parallelogram', 'cross'] if symmetric_objects else ['pyramid', 'angle']
    type_str = possible_types[0]
    colour_str = 'red'
    transformation_str = transformation_type
    transformation_value_str = get_transformation_value_str(0)

    bits = []
    if distance == 0:  # Train data: 000, 100, 010, 001
        version = np.random.choice([0, 1, 2, 3], size=1, replace=True, p=[0.25, 0.25, 0.25, 0.25])[0]
        bits = [0, 0, 0]
        if version == 1:
            type_str = possible_types[1]
            bits = [1, 0, 0]
        elif version == 2:
            colour_str = 'blue'
            bits = [0, 1, 0]
        elif version == 3:
            transformation_value_str = get_transformation_value_str(1)
            bits = [0, 0, 1]

    if distance == 1:  # Test data: 110, 101, 011
        version = np.random.choice([0, 1, 2], size=1, replace=True, p=[1/3, 1/3, 1/3])[0]
        if version == 0:
            type_str = possible_types[1]
            colour_str = 'blue'
            bits = [1, 1, 0]
        elif version == 1:
            type_str = possible_types[1]
            transformation_value_str = get_transformation_value_str(1)
            bits = [1, 0, 1]
        elif version == 2:
            colour_str = 'blue'
            transformation_value_str = get_transformation_value_str(1)
            bits = [0, 1, 1]

    if distance == 2:  # Test data: 111
        type_str = possible_types[1]
        colour_str = 'blue'
        transformation_value_str = get_transformation_value_str(1)
        bits = [1, 1, 1]

    task_description = {
        'language': f'Select the {type_str} {colour_str} object then {transformation_str} it {transformation_value_str}',
        'bits': bits,
        'id': bits[2]
    }

    type = types[type_str]
    colour = colours[colour_str]
    if transformation_type == 'translate':
        translation = translations[transformation_value_str]
    if transformation_type == 'rotate':
        rotation = rotations[transformation_value_str]

    def objects_placement_function(in_canvas: Canvas) -> Tuple[Canvas | None, str]:

        size_str = np.random.choice(['small', 'large'], size=1, replace=True, p=[0.5, 0.5])[0]
        size = sizes[size_str]

        probs_of_primitive_types = DistributionOver_ObjectTypes(distribution={type: 1})
        arg_ranges = {'size': [UniformDistribution(range=size.dx), UniformDistribution(range=size.dy)]}

        random_object_gen = RandomPrimitiveInCanvas(debug=False)
        obj = random_object_gen.create_random_primitive(obj_probs=probs_of_primitive_types, arg_ranges=arg_ranges)
        obj.set_new_colour(colour)

        if obj is None:
            return None, ''

        obj = random_object_gen.generate_random_canvas_pos(obj, in_canvas,
                                                           allowed_canvas_limits=Surround(
                                                               in_canvas.size.dy - obj.dimensions.dy - 2,
                                                               -1, 4,
                                                               in_canvas.size.dx - obj.dimensions.dx + 1))
        in_canvas = dsl.add_object_to_canvas(canvas=in_canvas, obj=obj)

        if second_object:
            in_canvas = dsl.add_object_to_canvas(canvas=in_canvas, obj=get_neighbouring_object_to_add(obj))

        description = f'There is a {size_str} {colour_str} {type_str} in position {obj.canvas_pos.x}, {obj.canvas_pos.y}'

        return in_canvas, description

    def logic_function(in_canvas: Canvas) -> Tuple[Canvas | None, str]:
        try:
            out_canvas = dsl.make_new_canvas_as(in_canvas)
            obj = dsl.select_only_object_of_colour(in_canvas, Colour.Blue)
            if obj is None:
                obj = dsl.select_only_object_of_colour(in_canvas, Colour.Red)
            origin = dsl.get_object_feature_canvas_pos(obj)

            if transformation_type == 'translate':
                dir = dsl.make_new_vector(orientation=translation, length=6, origin=origin)
                obj = dsl.object_transform_translate_along_direction(obj, dir)
            if transformation_type == 'rotate':
                obj = dsl.object_transform_rotate(obj, rotation)

            out_canvas = dsl.add_object_to_canvas(out_canvas, obj)

            yellow = dsl.select_only_object_of_colour(in_canvas, Colour.Yellow)
            if yellow is not None:
                out_canvas = dsl.add_object_to_canvas(out_canvas, yellow)

            size_str = 'small' if obj.dimensions.dx == small_dim else 'large'
            description = f'There is a {size_str} {colour_str} {type_str} in position {obj.canvas_pos.x}, {obj.canvas_pos.y}'
            return out_canvas, description
        except:
            return None, ''

    task = ARCLogicTask(objects_placement_function=objects_placement_function,
                        logic_function=logic_function,
                        num_of_outputs=num_of_outputs,
                        canvas_sizes_or_grid_stats=canvas_sizes)

    task.task_description = task_description
    task.task_id = task_description['id']

    return task

