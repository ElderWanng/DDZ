import random

import numpy as np


class ReservoirBuffer(object):
    ''' Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    '''

    def __init__(self, reservoir_buffer_capacity):
        ''' Initialize the buffer.
        '''
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        ''' Potentially adds `element` to the reservoir buffer.

        Args:
            element (object): data to be added to the reservoir buffer.
        '''
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        ''' Returns `num_samples` uniformly sampled from the buffer.

        Args:
            num_samples (int): The number of samples to draw.

        Returns:
            An iterable over `num_samples` random elements of the buffer.

        Raises:
            ValueError: If there are less than `num_samples` elements in the buffer
        '''
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                    num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        ''' Clear the buffer
        '''
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)
