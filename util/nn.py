from typing import NamedTuple, Tuple, Callable, Optional, Union, List

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.bijectors as tfb
import numpy as np
from tensorflow_probability.python import bijectors as tfb


class ModelArchitecture(NamedTuple):
    hidden_units: Optional[Tuple[int, ...]] = None
    activation: Optional[str] = None
    output_dim: Optional[Tuple[int, ...]] = None
    input_dim: Optional[Tuple[int, ...]] = None
    name: Optional[str] = None
    batch_norm: bool = False
    filters: Optional[Tuple[int, ...]] = None
    kernel_size: Optional[Union[Tuple[int, ...], int]] = None
    strides: Optional[Union[Tuple[int, ...], int]] = None
    padding: Union[Tuple[str, ...], str] = None
    raw_last: bool = True
    transpose: bool = False
    max_pooling: bool = False
    avg_pooling: bool = False

    @property
    def is_cnn(self):
        return self.filters is not None

    def invert(self, input_dim: Optional[Tuple[int, ...]]):
        assert not self.transpose
        model_arch = self._asdict()
        if self.name is not None:
            model_arch['name'] = 'inv_' + self.name
        if input_dim is None:
            input_dim = self.input_dim
        if self.is_cnn:
            model_arch['filters'] = tuple(reversed(self.filters[:-1])) + (self.input_dim[-1],)
            model_arch['kernel_size'] = tuple(reversed(self.kernel_size))
            model_arch['strides'] = tuple(reversed(self.strides))
            model_arch['padding'] = tuple(reversed(self.padding))
        else:
            model_arch["hidden_units"] = tuple(reversed(self.hidden_units))
        model_arch['output_dim'] = self.input_dim
        model_arch['input_dim'] = self.output_dim
        model_arch['transpose'] = True
        model_arch['max_pooling'] = self.max_pooling
        model_arch['avg_pooling'] = self.avg_pooling
        model_arch = ModelArchitecture(**model_arch)
        return model_arch

    def short_dict(self):
        return {k: v for k, v in self._asdict().items() if v is not None}


def generate_sequential_model(architecture: ModelArchitecture):
    m = tfk.Sequential(name=architecture.name)
    for i, units in enumerate(architecture.hidden_units):
        m.add(
            tfkl.Dense(
                units,
                activation=get_activation_fn(architecture.activation),
                name="{}_layer{:d}".format(architecture.name, i) if architecture.name is not None else None,
            ))
        if architecture.batch_norm:
            m.add(tfkl.BatchNormalization())
    return m


def get_model(
        model_arch: ModelArchitecture,
        invert: bool = False,
        output_dim: Optional[Tuple[int, ...]] = None,
        input_dim: Optional[Tuple[int, ...]] = None,
        as_model: bool = False,
):
    if model_arch.is_cnn:
        model_arch = model_arch._replace(hidden_units=None)
    if as_model:
        if invert:
            if model_arch.output_dim is not None:
                assert output_dim is None
                output_dim = model_arch.output_dim
            if output_dim is None:
                # dirty output dim inference
                _net = get_model(model_arch, as_model=True)
                output_dim = _net.outputs[0].shape[1:]
                del _net
            input_ = tfk.Input(output_dim)
            model_arch = model_arch.invert(output_dim)
        else:
            if model_arch.input_dim is not None:
                assert input_dim is None
                input_dim = model_arch.input_dim
            assert input_dim is not None, "input_dim should be provided"
            input_ = tfk.Input(input_dim)
        if model_arch.is_cnn:
            output = _conv_network(
                input_=input_,
                input_shape=input_.shape[1:],
                output_shape=model_arch.output_dim,
                **model_arch.short_dict(),)
        else:
            output = _fc_network(input_=input_, **model_arch.short_dict())
        model = tfk.Model(inputs=input_, outputs=output)
        return model
    if model_arch.is_cnn:
        if invert:
            layer = Deconvolutional(model_arch=model_arch, output_shape=output_dim)
        else:
            layer = Convolutional(model_arch)
    else:
        if invert:
            layer = TransposeFullyConnected(model_arch=model_arch, output_shape=output_dim)
        else:
            layer = FullyConnected(model_arch=model_arch)
    return layer


def get_activation_fn(activation: str):
    if callable(activation):
        return activation
    if hasattr(tf.nn, activation):
        return getattr(tf.nn, activation)
    elif hasattr(tfb, activation):
        return getattr(tfb, activation)()
    else:
        # custom activations

        def leaky_dSiLU(x):
            """Derivative of SiLU activation function."""
            cut_off = -3.
            sigmoid = tf.nn.sigmoid(x)
            one_minus_sigmoid = tf.where(x < cut_off, tf.exp(x), 1. - sigmoid)
            return sigmoid * (1 + x * one_minus_sigmoid) + 0.1 * tf.nn.leaky_relu(x)

        other_activations = {
            'dsilu': dSiLU,
            'smooth_elu': lambda x: tf.nn.softplus(2. * x + 2.) / 2. - 1.,
            'SmoothELU': tfb.Chain([tfb.Shift(-1.), tfb.Scale(.5), tfb.Softplus(), tfb.Shift(2.), tfb.Scale(2.)])
        }
        return other_activations.get(
            activation,
            ValueError("activation {} unknown".format(activation)))


def _pass_in_layers(layers: List[tfkl.Layer], input_):
    output = input_
    for layer in layers:
        output = layer(output)
    return output


def _fc_network_layers(hidden_units: Tuple[int, ...],
                       # F: general activation functions can now be provided by hand
                       # the set of available activation function is now larger (see get_activation_fn)
                       activation: Union[Callable, str],
                       output_dim: Optional[Tuple[int, ...]],
                       batch_norm: bool,
                       raw_last: bool,
                       **kwargs,
                       ):
    layers = [tfkl.Flatten()]
    assert output_dim is not None, "output_dim should be provided"
    units = tuple(list(hidden_units) + [np.prod(output_dim)])
    for i, unit in enumerate(units):
        apply_ = not raw_last and (i + 1 == len(units))
        layers.append(tfkl.Dense(unit))
        if apply_ and batch_norm:
            layers.append(tfkl.BatchNormalization())
        if apply_ and activation:
            if callable(activation):
                layers.append(tfkl.Activation(activation))
            else:
                layers.append(tfkl.Activation(get_activation_fn(activation)))
    if output_dim is not None:
        layers.append(tfkl.Reshape(output_dim))
    return layers


def _fc_network(input_: tfkl.Input,
                hidden_units: Tuple[int, ...],
                activation: str,
                output_dim: Optional[Tuple[int, ...]],
                batch_norm: bool,
                raw_last: bool,
                **kwargs,
                ):
    layers = _fc_network_layers(hidden_units, activation, output_dim, batch_norm, raw_last)
    return _pass_in_layers(layers, input_)


class FullyConnected(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            **kwargs
    ):
        self.model_arch = model_arch
        d = model_arch.short_dict()
        name = d.pop('name', None)
        super().__init__(name=name, **kwargs)
        self._layers = _fc_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)


class TransposeFullyConnected(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            output_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ):
        self.model_arch = model_arch
        self.invert_model_arch = model_arch.invert(output_shape)
        d = self.invert_model_arch.short_dict()
        name = d.pop('name')
        super().__init__(name=name, **kwargs)
        self._layers: List[tfkl.Layer] = _fc_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)


def _cnn_network_layers(
        filters: Tuple[int, ...],
        kernel_size: Tuple[Union[Tuple[int, ...], int], ...],
        strides: Tuple[Union[Tuple[int, ...], int], ...],
        padding: Tuple[str, ...],
        activation: Union[Callable, str],
        batch_norm: bool,
        raw_last: bool,
        transpose: bool = False,
        input_shape: Optional[Tuple[int, ...]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        max_pooling: bool = False,
        avg_pooling: bool = False,
        **kwargs,
):
    layers = []
    tf_layer = tfkl.Conv2D if not transpose else tfkl.Conv2DTranspose
    elements = [filters, kernel_size, strides, padding]
    # check that the number of elements is the same for all components
    n = len(elements[0])
    for element in elements[1:]:
        assert len(element) == n, "the number of filters, kernel_size, strides, padding should be the same"
    shapes = [[], []]
    if input_shape is not None:
        shapes[0].append(input_shape[0])
        shapes[1].append(input_shape[1])
    for i, (filters_, kernel_size_, stride_, padding_) in enumerate(zip(*elements)):
        apply_ = (i + 1 != len(filters)) or not raw_last
        conv_layer = tf_layer(
            filters=filters_,
            kernel_size=kernel_size_,
            strides=stride_,
            padding=padding_, )
        if input_shape is not None and transpose:
            rows, cols = shapes[0][-1], shapes[1][-1]
            if padding_ == 'valid':
                padding = [0, 0]
            else:
                p_left = np.floor((rows - conv_layer.strides[0]) / 2)
                p_right = rows - conv_layer.strides[0] - p_left
                p_top = np.floor((cols - conv_layer.strides[1]) / 2)
                p_bottom = cols - conv_layer.strides[1] - p_left
                padding = [p_left + p_right, p_top + p_bottom]
            new_rows = (rows - 1) * conv_layer.strides[0] + conv_layer.kernel_size[0] - 2 * padding[0]
            new_cols = (cols - 1) * conv_layer.strides[1] + conv_layer.kernel_size[1] - 2 * padding[1]
            if i + 1 == len(filters):
                output_padding = [0, 0]
                if output_shape[0] != new_rows:
                    output_padding[0] = output_shape[0] - new_rows
                if output_shape[1] != new_cols:
                    output_padding[1] = output_shape[1] - new_cols
                if output_padding != [0, 0]:
                    new_cols += output_padding[0]
                    new_rows += output_padding[1]
                    del conv_layer
                    conv_layer = tf_layer(
                        filters=filters_,
                        kernel_size=kernel_size_,
                        strides=stride_,
                        padding=padding_,
                        output_padding=output_padding)
            shapes[0].append(new_rows)
            shapes[1].append(new_cols)
        layers.append(conv_layer)
        if apply_ and activation:
            if callable(activation):
                layers.append(tfkl.Activation(activation))
            else:
                layers.append(tfkl.Activation(get_activation_fn(activation)))
        if apply_ and batch_norm:
            layers.append(tfkl.BatchNormalization())
        if max_pooling:
            layers.append(tfkl.MaxPooling2D())
        if avg_pooling:
            layers.append(tfkl.AveragePooling2D())
    return layers


def _conv_network(input_: tfkl.Input,
                  filters: Tuple[int, ...],
                  kernel_size: Tuple[Union[Tuple[int, ...], int], ...],
                  strides: Tuple[Union[Tuple[int, ...], int], ...],
                  padding: Tuple[str, ...],
                  activation: str,
                  batch_norm: bool,
                  raw_last: bool,
                  transpose: bool = False,
                  input_shape: Optional[Tuple[int, ...]] = None,
                  output_shape: Optional[Tuple[int, ...]] = None,
                  max_pooling: bool = False,
                  avg_pooling: bool = False,
                  **kwargs,
                  ):
    layers = _cnn_network_layers(
        filters, kernel_size, strides, padding, activation, batch_norm, raw_last, transpose, input_shape, output_shape,
        max_pooling, avg_pooling)
    return _pass_in_layers(layers, input_)


class Convolutional(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            **kwargs,
    ):
        self.model_arch = model_arch
        d = model_arch.short_dict()
        name = d.pop('name', None)
        super().__init__(name=name, **kwargs)
        # kernel_size, strides, padding = [
        #     tf.nest.flatten((param if param is not None else default))
        #     for param, default in zip((kernel_size, strides, padding), (3, 1, 'valid'))
        # ]
        self._layers: List[tfkl.Layer] = _cnn_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)


class Deconvolutional(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            output_shape: Optional[Tuple[int, ...]] = None,
            **kwargs,
    ):
        self.model_arch = model_arch
        self.invert_model_arch = model_arch.invert(output_shape)
        d = self.invert_model_arch.short_dict()
        name = d.pop('name')
        super().__init__(name=name, **kwargs)
        self._layers: List[tfkl.Layer] = _cnn_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)


def _get_elem(
        list_of_models: Union[List[Optional[tfk.Model]], Optional[tfk.Model]],
        i: int,
        default=tfk.Sequential([tfkl.Lambda(tf.identity)])
):
    list_of_models = tf.nest.flatten(list_of_models)
    model = list_of_models[min(i, len(list_of_models) - 1)]
    if model is None:
        return default
    else:
        return model


def dSiLU(x):
    """Derivative of SiLU activation function."""
    cut_off = 9.
    sigmoid = tf.nn.sigmoid(x)
    one_minus_sigmoid = tf.where(
        tf.abs(x) > cut_off,
        tf.exp(-tf.math.sign(x) * x),
        1. - sigmoid)
    return sigmoid * (1 + x * one_minus_sigmoid)


class StableSigmoid(tfb.Sigmoid):
    """
    A stable version of the sigmoid function that avoids numerical instabilities.
    """
    def __init__(self, epsilon: float = 1e-6):
        super(StableSigmoid, self).__init__()
        self._sigmoid = tfb.Sigmoid()
        self._epsilon = epsilon
        self._low = self._sigmoid.inverse(self._epsilon)
        self._high = self._sigmoid.inverse(1. - self._epsilon)

    def _forward(self, logits):
        clipped_logits = tf.clip_by_value(logits, self._low , self._high)
        logits = logits + tf.stop_gradient(clipped_logits - logits)

        # apply straight through gradients for damped logits
        res = self._sigmoid(logits)
        res = tf.where(
            res  <= 1.1 * self._epsilon,
            res - tf.stop_gradient(res),
            res)
        res = tf.where(
            res >= 1. - 1.1 * self._epsilon,
            res + tf.stop_gradient(1. - res),
            res)

        return res