from collections import OrderedDict

from general.should_be_builtins import memoize
from liquid_style.convnets import ConvLayer, Nonlinearity, Pooler, ConvNet
from fileman.file_getter import get_file
from scipy.io import loadmat
import numpy as np

__author__ = 'peter'


type_matches = lambda collection, klass: np.array([isinstance(x, type) for x in list], dtype = np.bool)

find_nth_match = lambda bool_arr, n: np.nonzeros


@memoize
def get_vgg_net(up_to_layer = None, force_shared_parameters = True, pooling_mode = 'max'):
    """
    :param up_to_layer: The layer to stop at.  Or a list of layers, in which case the network will go to the highest.
        Layers are identified by their string names:
        ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5', 'fc6', 'relu6', 'fc7', 'relu7',
        'fc8', 'prob']
    :param force_shared_parameters: Create net with shared paremeters.
    :return: A ConvNet object representing the VGG network.
    """

    filename = get_file(
        relative_name = 'data/vgg-19.mat',
        url = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat',
        )
    network_params = loadmat(filename)

    def struct_to_layer(struct):
        layer_type = struct[1][0]
        layer_name = str(struct[0][0])
        assert isinstance(layer_type, basestring)
        if layer_type == 'conv':
            w_orig = struct[2][0, 0]  # (n_rows, n_cols, n_in_maps, n_out_maps)
            w = w_orig.T.swapaxes(2, 3)  # (n_out_maps, n_in_maps, n_rows, n_cols)  (Theano conventions)
            b = struct[2][0, 1][:, 0]
            layer = ConvLayer(w, b, force_shared_parameters=force_shared_parameters)
        elif layer_type in ('relu', 'softmax'):
            layer = Nonlinearity(layer_type)
        elif layer_type == 'pool':
            layer = Pooler(region=tuple(struct[3][0].astype(int)), stride=tuple(struct[4][0].astype(int)), mode = pooling_mode)
        else:
            raise Exception("Don't know about this '%s' layer type." % layer_type)
        return layer_name, layer

    print 'Loading VGG Net...'
    network_layers = OrderedDict(struct_to_layer(network_params['layers'][0, i][0, 0]) for i in xrange(network_params['layers'].shape[1]))
    if up_to_layer is not None:
        if isinstance(up_to_layer, (list, tuple)):
            up_to_layer = network_layers.keys()[max(network_layers.keys().index(layer_name) for layer_name in up_to_layer)]
        layer_names = [network_params['layers'][0, i][0, 0][0][0] for i in xrange(network_params['layers'].shape[1])]
        network_layers = OrderedDict((k, network_layers[k]) for k in layer_names[:layer_names.index(up_to_layer)+1])
    print 'Done.'
    return ConvNet(network_layers)
