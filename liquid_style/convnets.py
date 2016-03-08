from collections import OrderedDict

from plato.core import symbolic, create_shared_variable
from plato.interfaces.helpers import get_named_activation_function
import theano.tensor as tt
from theano.tensor.signal.pool import pool_2d

__author__ = 'peter'


@symbolic
class ConvLayer(object):

    def __init__(self, w, b, force_shared_parameters = True):
        """
        w is the kernel, an ndarray of shape (n_in, n_out, n_y, n_x)
        b is the bias, an ndarray of shape (n_out, )
        force_shared_parameters: Set to true if you want to make the parameters shared variables.  If False, the
            parameters will be
        """
        self.w = create_shared_variable(w) if force_shared_parameters else tt.constant(w)
        self.b = create_shared_variable(b) if force_shared_parameters else tt.constant(b)

    def __call__(self, x):
        """
        param x: A (n_samples, n_rows, n_cols) image/feature tensor
        return: A (n_samples, n_rows-w_rows+1, n_cols-w_cols+1) tensor
        """
        return tt.nnet.conv2d(input=x, filters=self.w) + self.b[:, None, None]

    @property
    def parameters(self):
        return [self.w, self.b]

@symbolic
class Nonlinearity(object):

    def __init__(self, activation):
        """
        activation:  a name for the activation function. {'relu', 'sig', 'tanh', ...}
        """
        self.activation = get_named_activation_function(activation)

    def __call__(self, x):
        return self.activation(x)

@symbolic
class Pooler(object):

    def __init__(self, region, stride = None, mode = 'max'):
        """
        :param region: Size of the pooling region e.g. (2, 2)
        :param stride: Size of the stride e.g. (2, 2) (defaults to match pooling region size for no overlap)
        """
        assert len(region) == 2, 'Region must consist of two integers.  Got: %s' % (region, )
        if stride is None:
            stride = region
        assert len(region) == 2, 'Stride must consist of two integers.  Got: %s' % (region, )
        self.region = region
        self.stride = stride
        self.mode = mode

    def __call__(self, x):
        return pool_2d(x, ds = self.region, st = self.stride, mode = self.mode)


@symbolic
class ConvNet(object):

    def __init__(self, layers):
        self.n_layers = len(layers)
        if isinstance(layers, (list, tuple)):
            layers = OrderedDict(zip(enumerate(layers)))
        else:
            assert isinstance(layers, OrderedDict), "Layers must be presented as a list, tuple, or OrderedDict"
        self.layers = layers

    def __call__(self, x):
        return self.get_named_layer_activations(x).values()[-1]

    def get_named_layer_activations(self, x):
        """
        :returns: An OrderedDict<layer_name/index, activation>
            If you instantiated the convnet with an OrderedDict, the keys will correspond to the keys for the layers.
            Otherwise, they will correspond to the index which identifies the order of the layer.
        """
        named_activations = OrderedDict()
        for name, layer in self.layers.iteritems():
            x = layer(x)
            named_activations[name] = x
        return named_activations
