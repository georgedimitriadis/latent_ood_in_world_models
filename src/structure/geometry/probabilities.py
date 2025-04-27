
from copy import copy
import numpy as np
from typing import Dict, List, Tuple

from numpy import isclose

from structure.geometry.basic_geometry import Orientation, Colour


class DiscreteDistribution:
    """
    Defines an abstract class for a discrete distribution over some values.
    """
    def __init__(self, distribution: Dict[str, float], debug_wrong_update_printout: bool = False):
        """
        The initialisation of the abstract Discrete Distribution
        :param distribution: The values to uses for the distribution
        :param debug_wrong_update_printout: If True then every time a probability update becomes impossible there is a print notification
        """
        self.user_defined_probs = []
        self.name_prefix = '_' + self.__class__.__name__ + '__'
        self.debug_wrong_update_printout = debug_wrong_update_printout

        if distribution is None:
            self.set_to_uniform()
        else:
            self.set_to_distribution(distribution)

        assert isclose(self.sum(), 1, rtol=1e-6, atol=0.0), print('No sum to 1 at initialisation.')

    def __len__(self):
        return len([f for f in self.__dict__ if '__' in f])

    def __repr__(self):
        return str(self.get_all_probabilities_as_hashmap())

    def sum(self):
        """
        Calculates the sum over all probabilities
        :return: The sum
        """
        names = [f for f in self.__dict__ if '__' in f]
        result = 0
        for n in names:
            result += self.__dict__[n]
        return result

    def force_to_sum_to_one(self):
        """
        Changes the probabilities that the user hasn't specified so that the sum of the distribution is 1.
        The change happens using a uniform distribution over the non-user specified probabilities.
        :return:
        """
        user_def_values = []
        for tc in copy(self.user_defined_probs):
            user_def_values.append(self.__dict__[self.name_prefix + tc])
        sum_user_def_values = np.sum(user_def_values)
        total_values_to_change = len(self) - len(user_def_values)
        if total_values_to_change > 0:
            uniform_prob = (1 - sum_user_def_values) / total_values_to_change
            for t in self.__dict__:
                if '__' in t and t.split('__')[1] not in self.user_defined_probs:
                    self.__dict__[t] = uniform_prob

        assert isclose(self.sum(), 1, rtol=1e-6, atol=0.0), print('No sum to 1 when redefining the probabilities')

    def set_to_distribution(self, distribution: Dict[str, float]):
        """
        Sets part of the distribution to the one defined by distribution
        :param distribution: A dictionary of names: probability_values
        :return:
        """
        for t in distribution:
            v = distribution[t]
            self.__dict__[self.name_prefix + t] = v
            if t not in self.user_defined_probs:
                self.user_defined_probs.append(t)
        self.force_to_sum_to_one()

    def set_to_uniform(self):
        """
        Sets the whole distribution to a Uniform one
        :return:
        """
        size = len(self)
        self.user_defined_probs = []
        for t in self.__dict__:
            if '__' in t:
                self.__dict__[t] = 1 / size

    def get_all_probabilities_as_hashmap(self) -> Dict[str, float]:
        result = {}
        for k in self.__dict__:
            if '__' in k:
                result[k.split('__')[1]] = self.__dict__[k]
        return result

    def get_all_probabilities_as_list(self) -> List[float]:
        return list(self.get_all_probabilities_as_hashmap().values())

    def get_all_names(self) -> List[str]:
        return list(self.get_all_probabilities_as_hashmap().keys())

    def do_probability_update(self, name: str, new_value: float):
        """
        Update the distribution by changing a specific name to a new value. If it is not possible the change
        doesn't happen. If self.debug_wrong_update_printout is True this also creates a print notification
        :param name: The name of the probability
        :param new_value: The new value
        :return:
        """
        user_def_values = []
        for i in self.user_defined_probs:
            if i != name:
                user_def_values.append(self.__dict__[self.name_prefix + i])

        sum_user_def_values = np.sum(user_def_values) + new_value
        total_values_to_change = len(self) - len(user_def_values) - 1
        uniform_prob = (1 - sum_user_def_values) / total_values_to_change

        if uniform_prob < 0:
            if self.debug_wrong_update_printout:
                old_value = self.__dict__[self.name_prefix + name]
                print(
                    f'Update of {self.__dict__[self.name_prefix + name]} to {new_value} is impossible. '
                    f'Keeping old_value {old_value}')
            return
        else:
            self.__dict__[self.name_prefix + name] = new_value
            if name not in self.user_defined_probs:
                self.user_defined_probs.append(name)
            self.force_to_sum_to_one()

    def remove_from_user_defined(self, name):
        '''
        Removes a name from the user_defined_probs list
        :param name: The name of the probability to remove
        :return:
        '''
        self.user_defined_probs.remove(name)

    def number_of_non_zero(self) -> int:
        result = 0
        hasmap = self.get_all_probabilities_as_hashmap()
        for k in hasmap:
            if hasmap[k] > 0:
                result += 1
        return result

    def sample(self, size: int = 1, replace: bool = True) -> str:
        result = np.random.choice(self.get_all_names(), size=size, replace=replace,
                                p=self.get_all_probabilities_as_list())

        if size == 1:
            return result[0]

        return result


class DistributionOver_ObjectTypes(DiscreteDistribution):
    def __init__(self, distribution: Dict[str, float] | None = None, predefined_to_zero: bool = True):
        """
        Specifies a distribution over the ObjectType types.
        :param distribution: A set of probabilities, one for each type of the ObjectType enum
        :param predefined_to_zero: If True then the predefined type's prob is always set to 0.
        """
        self.__angle: float = 0
        self.__bolt: float = 0
        self.__cross: float = 0
        self.__diagonal: float = 0
        self.__dot: float = 0
        self.__fish: float = 0
        self.__hole: float = 0
        self.__inverseCross: float = 0
        self.__maze: float = 0
        self.__parallelogram: float = 0
        self.__pi: float = 0
        self.__predefined: float = 0
        self.__pyramid: float = 0
        self.__random: float = 0
        self.__steps: float = 0
        self.__spiral: float = 0
        self.__tie: float = 0

        self.predefined_to_zero = predefined_to_zero

        super().__init__(distribution=distribution)

    @property
    def angle(self):
        return self.__angle

    @angle.setter
    def angle(self, prob):
        self.do_probability_update('angle', prob)

    @property
    def bolt(self):
        return self.__bolt

    @bolt.setter
    def bolt(self, prob):
        self.do_probability_update('bolt', prob)

    @property
    def cross(self):
        return self.__cross

    @cross.setter
    def cross(self, prob):
        self.do_probability_update('cross', prob)

    @property
    def diagonal(self):
        return self.__diagonal

    @diagonal.setter
    def diagonal(self, prob):
        self.do_probability_update('diagonal', prob)

    @property
    def dot(self):
        return self.__dot

    @dot.setter
    def dot(self, prob):
        self.do_probability_update('dot', prob)

    @property
    def fish(self):
        return self.__fish

    @fish.setter
    def fish(self, prob):
        self.__fish = prob
        self.do_probability_update('fish', prob)
    @property
    def hole(self):
        return self.__hole

    @hole.setter
    def hole(self, prob):
        self.do_probability_update('hole', prob)

    @property
    def inverseCross(self):
        return self.__inverseCross

    @inverseCross.setter
    def inverseCross(self, prob):
        self.do_probability_update('inverseCross', prob)

    @property
    def maze(self):
        return self.__maze

    @maze.setter
    def maze(self, prob):
        self.do_probability_update('maze', prob)

    @property
    def parallelogram(self):
        return self.__parallelogram

    @parallelogram.setter
    def parallelogram(self, prob):
        self.do_probability_update('parallelogram', prob)

    @property
    def pi(self):
        return self.__pi

    @pi.setter
    def pi(self, prob):
        self.do_probability_update('pi', prob)

    @property
    def predefined(self):
        return self.__predefined

    @predefined.setter
    def predefined(self, prob):
        self.do_probability_update('predefined', prob)

    @property
    def pyramid(self):
        return self.__pyramid

    @pyramid.setter
    def pyramid(self, prob):
        self.do_probability_update('pyramid', prob)

    @property
    def random(self):
        return self.__random

    @random.setter
    def random(self, prob):
        self.do_probability_update('random', prob)

    @property
    def steps(self):
        return self.__steps

    @steps.setter
    def steps(self, prob):
        self.do_probability_update('steps', prob)

    @property
    def spiral(self):
        return self.__spiral

    @spiral.setter
    def spiral(self, prob):
        self.do_probability_update('spiral', prob)

    @property
    def tie(self):
        return self.__tie

    @tie.setter
    def tie(self, prob):
        self.do_probability_update('tie', prob)

    def force_to_sum_to_one(self) -> bool:
        """
        This overwrites the DiscreteDistribution.force_to_sum_to_one because it needs to keep the predefined
        value to 0 if the user has asked this
        :return:
        """
        user_def_values = []
        for tc in copy(self.user_defined_probs):
            user_def_values.append(self.__dict__['_' + self.__class__.__name__ + '__' + tc])
        sum_user_def_values = np.sum(user_def_values)
        total_values_to_change = len(self) - len(user_def_values)
        if self.predefined_to_zero:
            total_values_to_change -= 1
        uniform_prob = (1 - sum_user_def_values) / total_values_to_change
        if uniform_prob < 0:
            return False
        for t in self.__dict__:
            if ('_' + self.__class__.__name__ + '__') in t and t.split('__')[1] not in self.user_defined_probs:
                self.__dict__[t] = uniform_prob
                if 'predefined' in t and self.predefined_to_zero:
                    self.__dict__[t] = 0

        assert isclose(self.sum(), 1, rtol=1e-6, atol=0.0), \
            print(f'Sum = {self.sum()}. No sum to 1 when redefining the probabilities')

        return True

    def set_to_uniform(self):
        """
        This overwrites the DiscreteDistribution.set_to_uniform because the self.__predefined might be required to
        be kept to 0
        :return:
        """
        size = len(self) - 1 if self.predefined_to_zero else len(self)
        self.user_defined_probs = []
        self.__pyramid = 1/size
        self.__maze = 1/size
        self.__tie = 1/size
        self.__bolt = 1/size
        self.__fish = 1/size
        self.__parallelogram = 1/size
        self.__random = 1/size
        self.__hole = 1/size
        self.__cross = 1/size
        self.__diagonal = 1/size
        self.__inverseCross = 1/size
        self.__angle = 1/size
        self.__dot = 1/size
        self.__pi = 1/size
        self.__spiral = 1/size
        self.__steps = 1/size
        self.__predefined = 0 if self.predefined_to_zero else 1/size

    def sample(self, size: int = 1, replace: bool = True) -> str:
        result = np.random.choice(self.get_all_names(), size=size, replace=replace,
                                p=self.get_all_probabilities_as_list())
        result = result[0].capitalize()
        if result == 'Inversecross':
            result = 'InverseCross'
        return result


class DistributionOver_ObjectTransformations(DiscreteDistribution):
    def __init__(self, distribution: Dict[str, float] | None = None):
        """
        Specifies a distribution over the Transformation types.
        :param distribution: A set of probabilities, one for each type of the ObjectType enum
        :param predefined_to_zero: If True then the predefined type's prob is always set to 0.
        """
        self.__translate_to_coordinates: float = 0
        self.__translate_by: float = 0
        self.__translate_along: float = 0
        self.__translate_relative_point_to_point: float = 0
        self.__translate_until_touch: float = 0
        self.__translate_until_fit: float = 0
        self.__rotate: float = 0
        self.__scale: float = 0
        self.__shear: float = 0
        self.__mirror: float = 0
        self.__flip: float = 0
        self.__grow: float = 0
        self.__randomise_colour: float = 0
        self.__randomise_shape: float = 0
        self.__replace_colour: float = 0
        self.__replace_all_colours: float = 0
        self.__delete: float = 0
        self.__fill_holes: float = 0
        self.__fill: float = 0

        super().__init__(distribution=distribution)

    @property
    def translate_to_coordinates(self):
        return self.__translate_to_coordinates

    @translate_to_coordinates.setter
    def translate_to_coordinates(self, prob):
        self.do_probability_update('translate_to_coordinates', prob)

    @property
    def translate_by(self):
        return self.__translate_by

    @translate_by.setter
    def translate_by(self, prob):
        self.do_probability_update('translate_by', prob)

    @property
    def translate_along(self):
        return self.__translate_along

    @translate_along.setter
    def translate_along(self, prob):
        self.do_probability_update('translate_along', prob)

    @property
    def translate_relative_point_to_point(self):
        return self.__translate_relative_point_to_point

    @translate_relative_point_to_point.setter
    def translate_relative_point_to_point(self, prob):
        self.do_probability_update('translate_relative_point_to_point', prob)

    @property
    def translate_until_touch(self):
        return self.__translate_until_touch

    @translate_until_touch.setter
    def translate_until_touch(self, prob):
        self.do_probability_update('translate_until_touch', prob)

    @property
    def translate_until_fit(self):
        return self.__translate_until_fit

    @translate_until_fit.setter
    def translate_until_fit(self, prob):
        self.do_probability_update('translate_until_fit', prob)

    @property
    def rotate(self):
        return self.__rotate

    @rotate.setter
    def rotate(self, prob):
        self.do_probability_update('rotate', prob)

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, prob):
        self.do_probability_update('scale', prob)

    @property
    def shear(self):
        return self.__shear

    @shear.setter
    def shear(self, prob):
        self.do_probability_update('shear', prob)

    @property
    def mirror(self):
        return self.__mirror

    @mirror.setter
    def mirror(self, prob):
        self.do_probability_update('mirror', prob)

    @property
    def flip(self):
        return self.__flip

    @flip.setter
    def flip(self, prob):
        self.do_probability_update('flip', prob)

    @property
    def grow(self):
        return self.__grow

    @grow.setter
    def grow(self, prob):
        self.do_probability_update('grow', prob)

    @property
    def randomise_colour(self):
        return self.__randomise_colour

    @randomise_colour.setter
    def randomise_colour(self, prob):
        self.do_probability_update('randomise_colour', prob)

    @property
    def randomise_shape(self):
        return self.__randomise_shape

    @randomise_shape.setter
    def randomise_shape(self, prob):
        self.do_probability_update('randomise_shape', prob)

    @property
    def replace_colour(self):
        return self.__replace_colour

    @replace_colour.setter
    def replace_colour(self, prob):
        self.do_probability_update('replace_colour', prob)

    @property
    def replace_all_colours(self):
        return self.__replace_all_colours

    @replace_all_colours.setter
    def replace_all_colours(self, prob):
        self.do_probability_update('replace_all_colours', prob)

    @property
    def delete(self):
        return self.__delete

    @delete.setter
    def delete(self, prob):
        self.do_probability_update('delete', prob)

    @property
    def fill_holes(self):
        return self.__fill_holes

    @fill_holes.setter
    def fill_holes(self, prob):
        self.do_probability_update('fill_holes', prob)

    @property
    def fill(self):
        return self.__fill

    @fill.setter
    def fill(self, prob):
        self.do_probability_update('fill', prob)


class DistributionOver_Orientation(DiscreteDistribution):
    def __init__(self, distribution: Dict[str, float] | None = None):
        """
        Specifies a distribution over the Transformation types.
        :param distribution: A set of probabilities, one for each type of the ObjectType enum
        :param predefined_to_zero: If True then the predefined type's prob is always set to 0.
        """
        self.__up: float = 0
        self.__down: float = 0
        self.__left: float = 0
        self.__right: float = 0
        self.__up_right: float = 0
        self.__up_left: float = 0
        self.__down_right: float = 0
        self.__down_left: float = 0

        super().__init__(distribution=distribution)

    @property
    def up(self):
        return self.__up

    @up.setter
    def up(self, prob):
        self.do_probability_update('up', prob)

    @property
    def down(self):
        return self.__down

    @down.setter
    def down(self, prob):
        self.do_probability_update('down', prob)

    @property
    def left(self):
        return self.__left

    @left.setter
    def left(self, prob):
        self.do_probability_update('left', prob)

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, prob):
        self.do_probability_update('right', prob)

    @property
    def up_left(self):
        return self.__up_left

    @up_left.setter
    def up_left(self, prob):
        self.do_probability_update('up_left', prob)

    @property
    def up_right(self):
        return self.__up_right

    @up.setter
    def up_right(self, prob):
        self.do_probability_update('up_right', prob)

    @property
    def down_left(self):
        return self.__down_left

    @down_left.setter
    def down_left(self, prob):
        self.do_probability_update('down_left', prob)

    @property
    def down_right(self):
        return self.__down_right

    @down_right.setter
    def down_right(self, prob):
        self.do_probability_update('down_right', prob)

    def sample(self, size: int = 1, replace: bool = True) -> Orientation:
        result = np.random.choice(self.get_all_names(), size=size, replace=replace,
                                  p=self.get_all_probabilities_as_list())
        parts = result[0].split('_')
        name = parts[0].capitalize() if len(parts) == 1 else parts[0].capitalize() + '_' + parts[1].capitalize()

        return Orientation.get_orientation_from_name(name)


class DistributionOver_Colours(DiscreteDistribution):
    def __init__(self, distribution: Dict[str, float] | None = None):
        """
        Specifies a distribution over the Colours.
        :param distribution: A set of probabilities, one for each colour name of the Colour object. The str key should be the name of a colour with all small letters.
        """

        self.__black: float = 0
        self.__blue: float = 0
        self.__red: float = 0
        self.__green: float = 0
        self.__yellow: float = 0
        self.__gray: float = 0
        self.__purple: float = 0
        self.__orange: float = 0
        self.__azure: float = 0
        self.__burgundy: float = 0

        super().__init__(distribution=distribution)

    @property
    def black(self):
        return self.__black

    @black.setter
    def black(self, prob):
        self.do_probability_update('black', prob)

    @property
    def blue(self):
        return self.__blue

    @blue.setter
    def blue(self, prob):
        self.do_probability_update('blue', prob)

    @property
    def red(self):
        return self.__red

    @red.setter
    def red(self, prob):
        self.do_probability_update('red', prob)

    @property
    def green(self):
        return self.__green

    @green.setter
    def green(self, prob):
        self.do_probability_update('green', prob)

    @property
    def yellow(self):
        return self.__yellow

    @yellow.setter
    def yellow(self, prob):
        self.do_probability_update('yellow', prob)

    @property
    def gray(self):
        return self.__gray

    @gray.setter
    def gray(self, prob):
        self.do_probability_update('gray', prob)

    @property
    def purple(self):
        return self.__purple

    @purple.setter
    def purple(self, prob):
        self.do_probability_update('purple', prob)

    @property
    def orange(self):
        return self.__orange

    @orange.setter
    def orange(self, prob):
        self.do_probability_update('orange', prob)

    @property
    def azure(self):
        return self.__azure

    @azure.setter
    def azure(self, prob):
        self.do_probability_update('azure', prob)

    @property
    def burgundy(self):
        return self.__burgundy

    @burgundy.setter
    def burgundy(self, prob):
        self.do_probability_update('burgundy', prob)

    def sample(self, size: int = 1, replace: bool = True) -> Colour | List[Colour]:
        result = np.random.choice(self.get_all_names(), size=size, replace=replace,
                                  p=self.get_all_probabilities_as_list())

        result = [Colour(Colour.map_name_to_int[c.capitalize()]) for c in result]

        if len(result) == 1:
            return result[0]

        return result


class UniformDistribution:

    def __init__(self, range: Tuple[float|int, float|int] | float | int, step: int = 1):
        """
        A Uniform Distribution over floats, or ints
        :param range: The range of the distribution. If it is a Tuple then the two numbers are the min and max (so the interval is [min, max] for ints and [min, max) for floats. If it is a single float or int then the distribution is practically that number.
        :param step: The step over which to define an integer range. This has the effect of sampling as follows: np.random.choise(np.arange(min, max, step)) if the range is a Tuple of ints and no effect if it is a Tuple of floats.
        """
        self.range = range
        self.step = step

        if type(self.range) != tuple or self.range[0] == self.range[1]:
            self.min = self.max = self.range if type(self.range) != tuple else self.range[0]
            self.range = self.min
        else:
            self.min = self.range[0]
            self.max = self.range[1]
            assert type(self.min) == type(self.max), print(f'The types of both range elements {self.min}, {self.max} must be the same')

    def sample(self) -> int | float:
        if type(self.range) != tuple:
            return self.range

        if type(self.range[0]) == int:
            result = np.random.randint(self.min, self.max + 1)
            while (result - self.min) % self.step != 0:
                result = np.random.randint(self.min, self.max + 1)
            return result

        if type(self.range[1]) == float:
            return self.min + np.random.rand() * (self.max - self.min)

    def sample_n(self, n: int = 10) -> np.ndarray[int] | np.ndarray[float]:
        if type(self.range) != tuple:
            return np.array([self.range] * n)

        if type(self.range[0]) == int:
            if self.step == 1:
                return np.random.randint(self.min, self.max, size=n)
            else:
                possible_values = np.arange(self.min, self.max + 1, self.step)
                return np.random.choice(possible_values, n)

        if type(self.range[1]) == float:
            return self.min + np.random.rand(n) * (self.max - self.min)


