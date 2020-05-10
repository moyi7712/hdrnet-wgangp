import tensorflow as tf
from Config import Config
import numpy as np
import json, os


class PipeLine(object):
    def __init__(self, config, batch_size=None):
        self.config = config
        self.batch_size = batch_size if batch_size else self.config.batch_size
        with open(self.config.filelist, 'r') as f:
            self.filelist = json.load(f)

    def _ImRead(self, file_path, flage=None):
        assert flage in ['jpg', 'png']
        if flage == 'png':
            # file_path = tf.strings.regex_replace(file_path, 'jpg', 'png')
            image = tf.image.convert_image_dtype(tf.image.decode_png(tf.io.read_file(file_path),
                                                                     dtype=tf.uint16), tf.dtypes.float32)
        else:
            # file_path = tf.strings.regex_replace(file_path, 'png', 'jpg')
            image = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(file_path)
                                                                      ), tf.dtypes.float32)
        return image

    def _ReSize(self, image, size, flage=None):
        assert flage in ['longest', 'both']
        shape = tf.shape(image)

        if flage == 'longest':
            shape = tf.cast(512 * shape[:-1] / tf.reduce_max(shape), dtype=tf.int32)

            padding = tf.concat(
                [tf.constant([[0], [0], [0]]), tf.reshape(tf.constant([size, size, 3]) - shape, shape=(3, 1))],
                axis=1)
        if flage == 'both':
            shape = [size, size, 3]
            padding = tf.concat(
                [tf.constant([[0], [0], [0]]), tf.reshape(tf.constant([size, size, 3]) - shape, shape=(3, 1))],
                axis=1)
        image = tf.image.resize(image, size=shape[:-1], antialias=True, method=getattr(tf.image.ResizeMethod, self.config.resize_method))
        image = tf.pad(image, paddings=padding, mode='SYMMETRIC')
        return image

    def _PreProcess(self, filename_input, filename_label):
        image_input = self._ReSize(self._ImRead(filename_input, flage=self.config.image_type), self.config.image_size,
                                   flage=self.config.resize_mode)

        image_label = self._ReSize(self._ImRead(filename_label, flage=self.config.image_type), self.config.image_size,
                                   flage=self.config.resize_mode)

        if self.config.is_rot:
            k = np.random.randint(1, 8)
            image_input = tf.image.rot90(image_input, k)
            image_label = tf.image.rot90(image_label, k)
        return image_input, image_label

    def _build_dataset(self, inputs_list, lables_list):
        dataset = tf.data.Dataset.from_tensor_slices((inputs_list, lables_list))
        dataset = dataset.repeat(2)
        dataset = dataset.map(self._PreProcess, num_parallel_calls=self.config.num_parallel_calls)
        dataset = dataset.batch(self.batch_size,drop_remainder=True) \
            .prefetch(buffer_size=self.config.buffer_size) \
            .shuffle(self.config.shuffle)
        return dataset

    def DatasetTest(self):
        inputs_list = [os.path.join(self.config.input_path, temp) for temp in self.filelist['test']]
        labels_list = [os.path.join(self.config.label_path, temp) for temp in self.filelist['test']]
        return self._build_dataset(inputs_list, labels_list)

    def DatasetTrain(self):
        inputs_list = [os.path.join(self.config.input_path, temp) for temp in self.filelist['input']]
        if self.config.isPair:
            labels_list = [os.path.join(self.config.label_path, temp) for temp in self.filelist['input']]
        else:
            labels_list = [os.path.join(self.config.label_path, temp) for temp in self.filelist['label']]
        return self._build_dataset(inputs_list, labels_list)
