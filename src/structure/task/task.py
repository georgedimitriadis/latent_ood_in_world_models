
from __future__ import annotations

from typing import List, Tuple

from matplotlib import pyplot as plt
import numpy as np
from copy import copy

from structure.canvas.canvas import Canvas
# Do not delete the unused ones! They are actually been used through introspection!!
from structure.object.primitives import Primitive, ObjectType, Random, Parallelogram, Cross, Hole,\
    Pi, InverseCross, Dot, Angle, Diagonal, Steps, Fish, Bolt, Spiral, Tie, Pyramid
from structure.geometry.basic_geometry import Point, Dimension2D, Surround, Colour
from structure.object.transformation import ObjectTransformations
from experiments.data.generation import constants as const


class Task:

    def __init__(self, min_canvas_size_for_background_object: int = 10, prob_of_background_object: float = 0.1,
                 run_generate_canvasses: bool = True, number_of_io_pairs: int | None = None):

        self.min_canvas_size_for_background_object = min_canvas_size_for_background_object
        self.prob_of_background_object = prob_of_background_object
        if number_of_io_pairs is None:
            self.number_of_io_pairs = np.random.randint(2, const.MAX_EXAMPLE_PAIRS)
        else:
            self.number_of_io_pairs = number_of_io_pairs
        self.number_of_canvasses = self.number_of_io_pairs * 2 + 1

        self.input_canvases = []
        self.output_canvases = []
        self.test_input_canvas = None
        self.test_output_canvas = None
        self._canvas_ids = []

        # TODO: Develop the Grd Primitive to be able to also create the Grid Experiment type
        self.experiment_type = np.random.choice(['Object', 'Symmetry', 'Grid'], p=[0.8, 0.2, 0])

        self.canvasses_ids = []
        self.actual_pixel_ids = []
        self.objects = []
        self.temp_objects = []

        if run_generate_canvasses:
            self.generate_canvasses()

        self.task_id = None
        self.input_descriptions = []
        self.output_descriptions = []

        self.task_description = ''

    @property
    def canvas_ids(self):
        c_ids = []
        for i, o in zip(self.input_canvases, self.output_canvases):
            c_ids.append(i.id)
            c_ids.append(o.id)

        c_ids.append(self.test_input_canvas.id)
        if self.test_output_canvas is not None:
            c_ids.append(self.test_output_canvas.id)

        self._canvas_ids = c_ids
        return self._canvas_ids

    @staticmethod
    def get_random_colour(other_colour: int | None = None):
        colour = Colour(random=True).index
        if other_colour is None:
            return colour
        else:
            while colour == other_colour:
                colour = Colour(random=True).index
            return colour

    @staticmethod
    def get_random_position(obj: Primitive, canvas: Canvas) -> Point | None:
        """
        Finds all the positions an object would fit on the canvas (given where the other object are) and
        gives back a random one of these.
        :param obj: The object to position
        :param canvas: The Canvas to position the object in
        :return:
        """
        available_positions = canvas.where_object_fits_on_canvas(obj=obj)
        if len(available_positions) > 0:
            return np.random.choice(available_positions)
        else:
            return None

    def generate_canvasses(self):
        """
        Generate random size canvases and sum background (single colour) pixels to 10% of them if they are bigger than
        10x10. Override if a subclass requires a different way to generate canvases.
        :return:
        """
        min_pad_size = const.MIN_PAD_SIZE if self.experiment_type == 'Object' else const.MIN_PAD_SIZE + 15

        for c in range(self.number_of_io_pairs):
            input_size = Dimension2D(np.random.randint(min_pad_size, const.MAX_PAD_SIZE),
                                     np.random.randint(min_pad_size, const.MAX_PAD_SIZE))
            output_size = Dimension2D(np.random.randint(min_pad_size, const.MAX_PAD_SIZE),
                                      np.random.randint(min_pad_size, const.MAX_PAD_SIZE))

            input_canvas = Canvas(size=input_size, _id=2*c + 1)
            if np.all([input_size.dx > self.min_canvas_size_for_background_object,
                       input_size.dy > self.min_canvas_size_for_background_object]) \
                    and np.random.random() < self.prob_of_background_object:
                background_object = Random(size=input_size, occupancy_prob=np.random.gamma(1, 0.05)+0.1)
                input_canvas.create_background_from_object(background_object)
            self.input_canvases.append(input_canvas)

            output_canvas = Canvas(size=output_size, _id=2*c + 2)
            if np.all([output_size.dx > self.min_canvas_size_for_background_object,
                       output_size.dy > self.min_canvas_size_for_background_object]) \
                    and np.random.random() < self.prob_of_background_object:
                background_object = Random(size=output_size, occupancy_prob=np.random.gamma(1, 0.05)+0.1)
                output_canvas.create_background_from_object(background_object)
            self.output_canvases.append(output_canvas)

        test_canvas_size = Dimension2D(np.random.randint(min_pad_size, const.MAX_PAD_SIZE),
                                       np.random.randint(min_pad_size, const.MAX_PAD_SIZE))

        self.test_input_canvas = Canvas(size=test_canvas_size, _id=self.output_canvases[-1].id + 1)
        if np.all([test_canvas_size.dx > self.min_canvas_size_for_background_object,
                   test_canvas_size.dy > self.min_canvas_size_for_background_object]) \
                and np.random.random() < self.prob_of_background_object:
            background_object = Random(size=test_canvas_size, occupancy_prob=np.random.gamma(1, 0.05) + 0.1)
            self.test_input_canvas.create_background_from_object(background_object)

    def get_canvas_by_id(self, canvas_id: int) -> Canvas | None:
        for i, o in zip(self.input_canvases, self.output_canvases):
            if i.id == canvas_id:
                return i
            if o.id == canvas_id:
                return o

        if self.test_input_canvas.id == canvas_id:
            return self.test_input_canvas
        if self.test_output_canvas is not None and self.test_output_canvas.id == canvas_id:
            return self.test_output_canvas

        return None

    def create_random_object(self, obj_probs: np.ndarray | None = None, max_size_of_obj: Dimension2D = Dimension2D(15, 15),
                             overlap_prob: float = 0.8, far_away_prob: float = 0.1, debug: bool = False) -> Primitive:
        """
        Create a new Primitive chosen randomly.
        :param obj_probs: The probabilities of the type of Primitive
        :param max_size_of_obj: The maximum size of the Primitive
        :param overlap_prob: The probability that the Primitive's required_dist_to_others property will be negative
        allowing it to overlap with other Primitives
        :param far_away_prob: The probability the Primitive's required_dist_to_others property will be large keeping
        the Primitive away from others on the canvas
        :param debug: If True print some stuff
        :return: The Primitive
        """
        obj_type = ObjectType.random(_probabilities=obj_probs)

        if debug: print(obj_type)

        id = self.canvasses_ids[-1] + 1 if len(self.canvasses_ids) > 0 else 0
        actual_pixels_id = self.actual_pixel_ids[-1] + 1 if len(self.actual_pixel_ids) > 0 else 0
        args = {'colour': self.get_random_colour(),
                'border_size': Surround(0, 0, 0, 0),  # For now and then we will see
                'canvas_pos': Point(0, 0, 0),
                '_id': id,
                'actual_pixels_id': actual_pixels_id}
        self.canvasses_ids.append(id)
        self.actual_pixel_ids.append(actual_pixels_id)

        if obj_type.name in ['InverseCross', 'Steps', 'Pyramid', 'Diagonal']:  # These object have height not size
            args['height'] = np.random.randint(2, np.min(max_size_of_obj.to_numpy()))
            if obj_type.name == 'InverseCross' and args['height'] % 2 == 0:  # Inverted Crosses need odd height
                args['height'] += 1

        if obj_type.name == 'InverseCross':
            fill_colour = self.get_random_colour(other_colour=args['colour'])
            args['fill_colour'] = fill_colour
            args['fill_height'] = np.random.randint(1, args['height'] - 2) if args['height'] > 3 else 0
            if args['fill_height'] % 2 == 0:  # Inverted Crosses need odd fill_height
                args['fill_height'] += 1

        if obj_type.name == 'Steps':  # Steps also has depth
            args['depth'] = np.random.randint(1, args['height'])

        if not np.any(np.array(['InverseCross', 'Steps', 'Pyramid', 'Dot', 'Diagonal', 'Fish', 'Bolt', 'Tie'])
                      == obj_type.name):
            size = Dimension2D(np.random.randint(2, max_size_of_obj.dx), np.random.randint(2, max_size_of_obj.dy))
            if np.any(np.array(['Cross', 'InvertedCross']) == obj_type.name):   # Crosses need odd size
                if size.dx % 2 == 0:
                    size.dx += 1
                if size.dy % 2 == 0:
                    size.dy += 1
            args['size'] = size

        if obj_type.name == 'Hole':  # Hole has also thickness
            if args['size'].dx < 4:
                args['size'].dx = 4
            if args['size'].dy < 4:
                args['size'].dy = 4
            up = np.random.randint(1, args['size'].dy - 2)
            down = np.random.randint(1, args['size'].dy - up)
            left = np.random.randint(1, args['size'].dx - 2)
            right = np.random.randint(1, args['size'].dx - left)
            args['thickness'] = Surround(up, down, left, right)

        if debug: print(args)

        object = globals()[obj_type.name](**args)

        # Deal with the allowed minimum distance to other object.
        # Create a standard distance and then check if the object allows for overlap or needs to be far away from
        # others and change the distance to others accordingly
        if 'length' in args:
            min_distance_to_others = Surround(Up=args['length'], Down=0, Left=0, Right=args['length'])
        elif 'height' in args:
            min_distance_to_others = Surround(Up=args['height'], Down=0, Left=0, Right=args['height'])
        elif 'size' in args:
            min_distance_to_others = Surround(Up=args['size'].dy, Down=0, Left=0, Right=args['size'].dx)
        else:
            min_distance_to_others = Surround(Up=1, Down=1, Left=1, Right=1)

        min_distance_to_others += args['border_size']

        allow_overlap = np.random.random() < overlap_prob
        if allow_overlap and np.all((min_distance_to_others - 2).to_numpy() > 1):
            min_distance_to_others -= Surround(Up=np.random.randint(1, min_distance_to_others.Up - 2),
                                               Down=np.random.randint(1, min_distance_to_others.Down - 2),
                                               Left=np.random.randint(1, min_distance_to_others.Left - 2),
                                               Right=np.random.randint(1, min_distance_to_others.Right - 2))
        else:
            far_away = np.random.random() < far_away_prob
            if far_away and np.all((min_distance_to_others - 2).to_numpy() > 1):
                min_distance_to_others += Surround(Up=np.random.randint(1, min_distance_to_others.Up - 2),
                                                   Down=np.random.randint(1, min_distance_to_others.Down - 2),
                                                   Left=np.random.randint(1, min_distance_to_others.Left - 2),
                                                   Right=np.random.randint(1, min_distance_to_others.Right - 2))

        object.required_dist_to_others = min_distance_to_others

        return object

    def generate_objects_from_json_description(self, unique_objects: List):
        for obj_discr in unique_objects:
            obj_type = obj_discr['primitive']

            args = {}
            if obj_type in ['InverseCross', 'Steps', 'Pyramid', 'Diagonal']:
                args['height'] = obj_discr['dimensions'].dy
                if obj_type == 'InverseCross':
                    args['fill_colour'] = obj_discr['fill_colour']
                    args['fill_height'] = obj_discr['fill_height']
            elif obj_type in ['Tie', 'Bolt', 'Fish', 'Dot']:
                pass
            else:
                args['size'] = obj_discr['dimensions']

            if obj_type == 'Bolt':
                args['_center_on'] = obj_discr['center_on']

            try:
                args['colour'] = obj_discr['colour']
            except:
                pass
            args['_id'] = obj_discr['id']
            args['actual_pixels_id'] = obj_discr['actual_pixels_id']

            '''
            for canvas_and_pos in obj_discr['canvases_positions']:
                args['canvas_id'] = canvas_and_pos[0]
                args['canvas_pos'] = canvas_and_pos[1]

                if int(args['canvas_id'] / 2) <= self.number_of_io_pairs - 1:
                    canvas = self.input_canvases[int(args['canvas_id'] / 2)] if args['canvas_id'] % 2 == 0 else \
                        self.output_canvases[int(args['canvas_id'] / 2)]
                elif np.ceil(args['canvas_id'] / 2) == self.number_of_io_pairs:
                    canvas = self.test_input_canvas
                elif np.ceil(args['canvas_id'] / 2) == self.number_of_io_pairs + 1:
                    canvas = self.test_output_canvas
                obj = globals()[obj_type](**args)

                
                for tr in obj_discr['transformations']:
                    transform_name = Transformations(tr[0])
                    tr_args = transform_name.list_to_transformation_args(tr[0], tr[1])
                    transform_method = getattr(obj, transform_name.name)
                    transform_method(**tr_args)
                

                if 'actual_pixels' in obj_discr:
                    obj.actual_pixels = obj_discr['actual_pixels']

                self.object.append(obj)
                canvas.add_new_object(obj)
                '''

            canvas_and_pos = obj_discr['canvas_and_position']
            args['canvas_id'] = canvas_and_pos[0]
            args['canvas_pos'] = canvas_and_pos[1]

            if int(args['canvas_id'] / 2) <= self.number_of_io_pairs - 1:
                canvas = self.input_canvases[int(args['canvas_id'] / 2)] if args['canvas_id'] % 2 == 0 else \
                    self.output_canvases[int(args['canvas_id'] / 2)]
            elif np.ceil(args['canvas_id'] / 2) == self.number_of_io_pairs:
                canvas = self.test_input_canvas
            elif np.ceil(args['canvas_id'] / 2) == self.number_of_io_pairs + 1:
                canvas = self.test_output_canvas
            obj = globals()[obj_type](**args)

            for tr in obj_discr['on_canvas_transformations']:
                transform_name = ObjectTransformations.get_transformation_from_name(tr[0])
                tr_args = transform_name.list_to_transformation_args(tr[0], tr[1])
                transform_method = getattr(obj, transform_name.name)
                transform_method(**tr_args)

            if 'actual_pixels' in obj_discr:
                obj.actual_pixels = obj_discr['actual_pixels']

            self.objects.append(obj)
            canvas.add_new_object(obj)

            for new_obj_trans in obj_discr['in_out_transformations']:
                new_obj = copy(obj)
                for tr in new_obj_trans:
                    transformation = ObjectTransformations.get_transformation_from_name(tr[0])
                    tr_args = transformation.list_to_transformation_args(tr[0], tr[1])
                    transform_method = getattr(new_obj, transformation.name)
                    transform_method(**tr_args)
                if int(new_obj.canvas_id / 2) <= self.number_of_io_pairs - 1:
                    out_canvas = self.output_canvases[int(new_obj.canvas_id / 2)]
                elif np.ceil(new_obj.canvas_id / 2) == self.number_of_io_pairs:
                    out_canvas = self.test_output_canvas
                self.objects.append(new_obj)
                out_canvas.add_new_object(new_obj)

    def do_random_transformations(self, obj: Primitive, debug: bool = False, num_of_transformations: int = 0,
                                  probs_of_transformations: List = tuple([1 / len(ObjectTransformations) * len(ObjectTransformations)])):
        """
        Transform the obj Primitive num_of_transformations times with randomly selected Transformations (except mirror).
        The arguments of each Transformation are also chosen randomly
        :param obj: The Object to transform
        :param debug: If True print the transformations and their arguments
        :param num_of_transformations: How many transformations to do
        :param probs_of_transformations: The probabilities of the 6 possible transformations in the order of the
        Transformations enumeration
        :return: Nothing
        """
        if debug: print(type(obj))

        if debug: print(f'number of transforms = {num_of_transformations}')

        possible_transform_indices = range(len(ObjectTransformations))

        for i in range(num_of_transformations):

            transformation_result_ok = False
            while not transformation_result_ok:
                temp_obj = copy(obj)
                random_transform_index = np.random.choice(possible_transform_indices, p=probs_of_transformations)

                transform_name = ObjectTransformations(random_transform_index)
                if debug: print(f'Transform = {transform_name.name}')

                obj_type = 'Random' if temp_obj.get_str_type() == 'Random' else 'Non-Random'
                args = transform_name.get_random_parameters(obj_type)
                if debug: print(f'Arguments = {args}')

                transform_method = getattr(temp_obj, transform_name.name)

                # Do not allow Cross or InvertedCross to scale by an even factor
                if np.any(np.array(['Cross', 'InvertedCross']) == obj.get_str_type()) and \
                    transform_name == ObjectTransformations.scale and args['factor'] % 2 == 0:
                    break

                transform_method(**args)

                # Make sure the resulting object isn't too big
                if temp_obj.dimensions.dx < const.MAX_PAD_SIZE - 6 and temp_obj.dimensions.dy < const.MAX_PAD_SIZE - 6:
                    transformation_result_ok = True

            transform_method = getattr(obj, transform_name.name)
            transform_method(**args)
            obj.actual_pixels_id = self.actual_pixel_ids[-1] + 1
            self.actual_pixel_ids.append(obj.actual_pixels_id)

    def randomly_position_object_in_all_canvases(self, obj: Primitive):
        """
        It takes an object and places it in random (but allowed) positions in all the Canvases of the Task
        :param obj: The object to position
        :return:
        """
        for input_canvas, output_canvas in zip(self.input_canvases, self.output_canvases):
            in_canvas_pos = self.get_random_position(obj, input_canvas)
            out_canvas_pos = self.get_random_position(obj, output_canvas)
            canvas_pos = None
            c = None
            if in_canvas_pos is not None and out_canvas_pos is not None:
                canvas_pos, c = (in_canvas_pos, input_canvas) if np.random.random() > 0.5 \
                    else (out_canvas_pos, output_canvas)
            elif in_canvas_pos is None:
                canvas_pos = out_canvas_pos
                c = output_canvas
            elif out_canvas_pos is None:
                canvas_pos = in_canvas_pos
                c = input_canvas
            if canvas_pos is not None:
                o = copy(obj)
                o.canvas_pos = canvas_pos
                c.add_new_object(o)
                self.objects.append(o)

        test_canvas_pos = self.get_random_position(obj, self.test_input_canvas)
        if test_canvas_pos is not None:
            o = copy(obj)
            o.canvas_pos = test_canvas_pos
            self.test_input_canvas.add_new_object(o)
            self.objects.append(o)

    def add_object_on_canvasses(self, obj: Primitive, canvas_ids: List[int]):
        self.objects.append(obj)
        for i, o in zip(self.input_canvases, self.output_canvases):
            if i.id in canvas_ids:
                i.add_new_object(obj)
            if o.id in canvas_ids:
                o.add_new_object(obj)

        if self.test_input_canvas.id in canvas_ids:
            self.test_input_canvas.add_new_object(obj)
        if self.test_output_canvas is not None and self.test_output_canvas.id in canvas_ids:
            self.test_output_canvas.add_new_object(obj)

    def create_canvas_arrays_input(self) -> dict:
        result = {'test': [{'input': np.flipud(self.test_input_canvas.full_canvas).tolist()}],
                  'train': []}

        for input_canvas, output_canvas in zip(self.input_canvases, self.output_canvases):
            result['train'].append({'input': np.flipud(input_canvas.full_canvas).tolist(),
                                    'output': np.flipud(output_canvas.full_canvas).tolist()})

        return result

    def get_all_colours(self) -> List[int]:
        pass
        colours = set()
        for i, o in zip(self.input_canvases, self.output_canvases):
            colours.update(set(i.get_used_colours()))
            colours.update(set(o.get_used_colours()))
        colours.update(set(self.test_input_canvas.get_used_colours()))

        return list(colours)

    def json_output_of_all_objects(self, lean: bool = True, only_canvasses: List[int] | None = None) -> Tuple[List, np.ndarray]:
        obj_pix_ids = []
        obj_with_unique_pix_ids = []
        positions_of_same_objects = {}
        actual_pixels_list = []
        unique_objects = []
        for obj in self.objects:
            obj_pix_id = (obj.id, obj.actual_pixels_id)
            if not obj_pix_id in obj_pix_ids:
                unique_objects.append(obj.json_output())
                if not lean:
                    unique_objects[-1]['actual_pixels'] = obj.actual_pixels.tolist()
                actual_pixels_list.append(obj.actual_pixels)
                obj_pix_ids.append(obj_pix_id)
                obj_with_unique_pix_ids.append(obj)
                positions_of_same_objects[obj_pix_id] = [[obj.canvas_id,
                                                         [obj.canvas_pos.x, obj.canvas_pos.y, obj.canvas_pos.z]]]
            else:
                positions_of_same_objects[obj_pix_id].append([obj.canvas_id,
                                                              [obj.canvas_pos.x, obj.canvas_pos.y, obj.canvas_pos.z]])

        for unique_object in unique_objects:
            o_p_id = (unique_object['id'], unique_object['actual_pixels_id'])
            if lean:
                unique_object['canvasses_positions'] = positions_of_same_objects[o_p_id]
            else:
                unique_object['canvasses_positions'] = {'canvas_ids': [],
                                                        'canvas_positions': []}
                for p in positions_of_same_objects[o_p_id]:
                    unique_object['canvasses_positions']['canvas_ids'].append(p[0])
                    unique_object['canvasses_positions']['canvas_positions'].append(p[1])
            unique_object.pop('canvas_pos', None)
            unique_object.pop('dimensions', None)

        actual_pixels_array = np.zeros((const.MAX_PAD_SIZE, const.MAX_PAD_SIZE, len(actual_pixels_list)))
        for i, ap in enumerate(actual_pixels_list):
            actual_pixels_array[:ap.shape[0], :ap.shape[1], i] = ap

        return unique_objects, actual_pixels_array

    def create_example_json(self):
        result = {'Input Canvasses': [], 'Output Canvasses': []}
        for i, o in zip(self.input_canvases, self.output_canvases):
            result['Input Canvasses'].append(i.json_output())
            result['Output Canvasses'].append(o.json_output())

        return result

    def show(self, canvas_index: int | str = 'all', save_as: str | None = None, two_cols: bool = False):
        """
        Shows some (canvas_index is int or 'test') or all (canvas_index = 'all') the Canvases of the Experiment
        :param two_cols: If False it will show the test data in a third column. Otherwise it will show the test data under the train
        :param save_as: If not None then save the figure generated as save_as file (but do not show it).
        :param canvas_index: Which Canvases to show (int, 'test' or 'all')
        :return:
        """

        num_of_figure = None
        if save_as is not None:
            num_of_figure = 1

        thin_lines = True
        if save_as is None:
            thin_lines = False

        if type(canvas_index) == int:
            if canvas_index % 2 == 0:
                canvas = self.input_canvases[canvas_index // 2]
            else:
                canvas = self.output_canvases[canvas_index // 2]
            canvas.show(save_as=save_as)

        elif canvas_index == 'test':
            self.test_input_canvas.show(save_as=save_as)

        elif canvas_index == 'all':
            fig = plt.figure(num=num_of_figure, figsize=(6, 16))
            index = 1
            nrows = self.number_of_io_pairs + 1 if two_cols else self.number_of_io_pairs
            ncoloumns = 2 if two_cols else 3
            if not two_cols:
                for p in range(self.number_of_io_pairs):
                    self.input_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index,
                                                thin_lines=thin_lines)
                    self.output_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index + 1,
                                                 thin_lines=thin_lines)
                    if p == 0:
                        self.test_input_canvas.show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index + 2,
                                                    thin_lines=thin_lines)
                    if p == 1 and self.test_output_canvas is not None:
                        self.test_output_canvas.show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index + 2,
                                                     thin_lines=thin_lines)
                    index += 3
            else:
                for p in range(self.number_of_io_pairs + 1):
                    if p < self.number_of_io_pairs:
                        self.input_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index,
                                                    thin_lines=thin_lines)
                        self.output_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index + 1,
                                                     thin_lines=thin_lines)
                    else:
                        self.test_input_canvas.show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index,
                                                    thin_lines=thin_lines)
                        if self.test_output_canvas is not None:
                            self.test_output_canvas.show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index + 1,
                                                         thin_lines=thin_lines)
                    index += 2
            plt.tight_layout(pad=0.01)

            if save_as is not None:
                fig.savefig(save_as, dpi=500)
                plt.clf()
                #plt.close('all')


