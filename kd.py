from typing import Optional, List, Tuple

import numpy as np


class KDTree:

    def __init__(self, data: Optional[np.ndarray], axis: int = 0, min_dims: np.ndarray = None,
                 max_dims: np.ndarray = None):
        if data is None and (min_dims is None or max_dims is None):
            raise ValueError('Cannot create Tree without bounding boxes!')

        if min_dims is None:
            min_dims = np.zeros((data.shape[1]))
        if max_dims is None:
            max_dims = np.ones((data.shape[1]))

        self._min_dims = min_dims.copy()
        self._max_dims = max_dims.copy()
        self._axis = axis

        if data is not None and len(data) != 0:
            self._axis = axis % data.shape[1]

            self._data = data[np.argsort(data[:, self._axis].copy())].copy()
            # integer division
            median = len(self._data) // 2

            # updating bounding box
            new_min = self._min_dims.copy()
            new_min[self._axis] = self._data[median, self._axis].copy()

            new_max = self._max_dims.copy()
            new_max[self._axis] = self._data[median, self._axis].copy()
            self._root = self._data[median, :].copy()
            self._annotation = f'Split on axis {self._axis}, <{self._root[self._axis]}>'

            # single elem
            if len(self._data) == 1:
                self._root = self._data[median, :]
                self._split = new_min, new_max
                self._left = KDTree(None, self._axis + 1, min_dims.copy(), new_max.copy())
                self._right = KDTree(None, self._axis + 1, new_min.copy(), max_dims.copy())
            else:
                self._root = self._data[median, :].copy()

                self._split = new_min, new_max

                self._left = KDTree(self._data[:median, :], self._axis + 1, min_dims.copy(),
                                    new_max.copy())

                right = self._data[median + 1:, :]
                self._right = KDTree(right if len(right) != 0 else None, self._axis + 1,
                                     new_min.copy(), max_dims.copy())
        else:
            # completely empty tree
            self._data = None
            self._root = None
            self._left = None
            self._right = None

    def boxes(self, data: np.ndarray):
        for target, min, max in zip(data, self._min_dims, self._max_dims):
            if not (min <= target <= max):
                return False
        return True

    def _add(self, data: np.ndarray) -> None:
        # go to left node
        if self.left and self.left.boxes(data):
            self.left.add(data)
        # go to right node
        elif self.right and self.right.boxes(data):
            self.right.add(data)
        # empty node
        else:
            self._data = data.copy()

            self._axis = self._axis % len(data)

            # updating bounding box
            new_min = self._min_dims.copy()
            new_min[self._axis] = data[self._axis]

            new_max = self._max_dims.copy()
            new_max[self._axis] = data[self._axis]
            self._root = self._data.copy()
            self._annotation = f'Split on axis {self._axis}, <{self._root[self._axis]}>'

            self._left = KDTree(None, self._axis + 1, self._min_dims.copy(), new_max.copy())
            self._right = KDTree(None, self._axis + 1, new_min.copy(), self._max_dims.copy())
            self._split = new_min, new_max

    def add(self, data: np.ndarray) -> 'KDTree':
        if self.boxes(data):
            self._add(data)
            return self
        else:
            return KDTree(np.asarray(self.data + [data]), self._axis)

    def _delete(self, data: np.ndarray) -> None:
        if self.left and np.array_equal(self.left._root, data):
            elems = self.left._data_without(data)
            self._left = KDTree(np.asarray(elems), self._axis + 1, self._max_dims.copy(),
                                self._max_dims.copy())
        elif self.right and np.array_equal(self.right._root, data):
            elems = self.right._data_without(data)
            self._right = KDTree(np.asarray(elems), self._axis + 1, self._max_dims.copy(),
                                 self._max_dims.copy())
        else:
            # go to left node
            if self.left and self.left.boxes(data):
                self.left._delete(data)
            # go to right node
            else:
                self.right._delete(data)

    def delete(self, data: np.ndarray) -> 'KDTree':
        # tree does not contain data
        if not self.contains(data):
            return self
        # delete root
        elif np.array_equal(self._root, data):
            elems = self._data_without(data)
            return KDTree(np.asarray(elems), self._axis)
        # delete some other node
        else:
            self._delete(data)
            return self

    def contains(self, data: np.ndarray) -> bool:
        elems = self.data
        for elem in elems:
            if np.array_equal(elem, data):
                return True
        return False

    def _data_without(self, data: np.ndarray):
        elems = self.data
        for i, elem in enumerate(elems):
            if np.array_equal(elem, data):
                del elems[i]
                return elems
        return elems

    @property
    def left(self) -> Optional['KDTree']:
        if hasattr(self, '_left'):
            return self._left

    @property
    def right(self) -> Optional['KDTree']:
        if hasattr(self, '_right'):
            return self._right

    @property
    def data(self) -> List[np.ndarray]:
        data = [self._root] if self._root is not None else []
        data += self.left.data if self.left is not None else []
        data += self.right.data if self.right is not None else []

        return data

    @property
    def split(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if hasattr(self, '_split'):
            return self._split

    @property
    def bounding_boxes(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        splits = [self.split] if self.split else []
        splits += self.left.bounding_boxes if self.left else []
        splits += self.right.bounding_boxes if self.right else []

        return splits

    @property
    def annotations(self) -> List[str]:
        ann = [self._annotation] if hasattr(self, '_annotation') else []
        ann += self.left.annotations if self.left else []
        ann += self.right.annotations if self.right else []

        return ann

    def __str__(self):
        return f'KDTree of {self._min_dims, self._max_dims}'
