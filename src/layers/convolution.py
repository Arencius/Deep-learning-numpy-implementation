import numpy as np
from src.layers.layer import Layer


class Conv2DLayer(Layer):
    def __init__(self,
                 input_size: tuple,
                 output_filters: int,
                 filter_size: int = 3,
                 stride: int = 1,
                 padding: bool = False):
        """
        Convolution layer.
        :param image_size:
        :param output_filters:
        :param filter_size:
        :param stride:
        :param padding:
        """
        super().__init__()
        if not len(input_size) == 3:
            raise ValueError(f'Input should be 3-dimentional: h,w,c. Got: {input_size}')

        self.height, self.width, self.channels = input_size
        self.output_filters = output_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.filters = self._initialize_weights()
        self.biases = np.zeros(shape=(self.output_filters,))

    @property
    def parameters(self):
        return np.prod(self.filters.shape) + self.output_filters

    @property
    def output_shape(self):
        output_height = (self.height - self.filter_size + int(self.padding) * 2) / self.stride + 1
        output_width = (self.width - self.filter_size + int(self.padding) * 2) / self.stride + 1

        return int(output_height), int(output_width), self.output_filters

    def _initialize_weights(self):
        return np.random.normal(size=(self.filter_size, self.filter_size, self.channels, self.output_filters)) * 0.01

    def pad_image(self, image):
        padded = np.pad(image, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')
        return padded

    def _get_submatrices(self, images):
        batch_size = images.shape[0]
        shape = (batch_size, self.output_shape[0], self.output_shape[1],
                 self.filter_size, self.filter_size, self.channels)

        strides = (images.strides[0], images.strides[1] * self.stride, images.strides[2] * self.stride,
                   images.strides[1], images.strides[2], images.strides[3])

        sub_matrices = np.lib.stride_tricks.as_strided(images,
                                                       shape=shape,
                                                       strides=strides)
        return sub_matrices

    def forward(self, x):
        if self.padding:
            x = self.pad_image(x)

        submatrices = self._get_submatrices(x)
        filters = self.filters[:, :, None, :, :]
        convolved_img = np.tensordot(submatrices, filters, axes=((3, 4, 5), (0, 1, 3))) + self.biases

        return np.squeeze(convolved_img, axis=-2)

    def backward(self, output_gradients):
        pass
