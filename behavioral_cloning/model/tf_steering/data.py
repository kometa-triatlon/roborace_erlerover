import numpy as np
import h5py


class BatchGenerator:
    def __init__(self, filepath, batch_size=1, shuffle=False, infinite=False):
        with h5py.File(filepath) as f:
            self.__data = np.asarray(f['data'])
            self.__label = np.asarray(f['label'])

        self.__count = self.__data.shape[0]
        self.__indices = np.arange(self.__count)
        if shuffle:
            np.random.shuffle(self.__indices)

        self.__index = 0
        self.__batch_size = batch_size
        self.__infinite = infinite


    def __iter__(self):
        return self


    def next(self):
        if self.__index >= self.__count and not self.__infinite:
            raise StopIteration()

        index = self.__index % self.__count
        batch_size = np.min([self.__batch_size, self.__count - index])
        batch_data = np.zeros((batch_size, self.__data.shape[2], self.__data.shape[3], self.__data.shape[1]), dtype=np.float32)
        batch_label = np.zeros((batch_size,), dtype=np.float32)
        for i in np.arange(index, index+batch_size):
            batch_data[i - index, :, :, :] = self.__data[self.__indices[i], :, :, :].transpose([1, 2, 0])
            batch_label[i - index] = self.__label[self.__indices[i]]

        self.__index += batch_size
        return batch_data, batch_label

    def reset(self):
        self.__index = 0
