from liquid_style.art_gallery import get_image
from liquid_style.convnets import ConvLayer
from liquid_style.images2gif import OnlineGifWriter
from liquid_style.pretrained_networks import get_vgg_net
from fileman.experiment_record import ExperimentLibrary, Experiment, get_current_experiment_id
from fileman.local_dir import get_local_path
from general.numpy_helpers import get_rng
from general.should_be_builtins import bad_value
from plato.core import symbolic, create_shared_variable
from plato.tools.optimization.optimizers import GradientDescent, get_named_optimizer
from plotting.db_plotting import dbplot
from theano.gof.graph import Variable
import theano.tensor as tt
import numpy as np
from matplotlib import pyplot as plt

__author__ = 'peter'

"""
Here, we try a variant on the Algorithm presented in:
A Neural Algorithm of Artistic Style
by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
http://arxiv.org/pdf/1508.06576v2.pdf

"""


@symbolic
def feature_corr(feature_representation):
    """
    Find the correlation within a feature map, pooled over pixels.
    See equation 3 from the paper.

    :param feature_representation: A shape (n_samples,n_maps,size_y,size_x) feature representation
    :return: A shape (n_samples, n_maps, n_maps) matrix of correlations.
    """
    flattened_rep = feature_representation.flatten(3)
    return (flattened_rep[:, None, :, :] * flattened_rep[:, :, None, :]).sum(axis=3)


@symbolic
def layer_style_loss(evolving_features, style_features):
    """
    Find the style-loss given the features of the content and style images for a given layer.
    See equation 4 from the paper.
    :param evolving_features: (n_samples, n_maple, size_y, size_x) feature rep derived from the content image
    :param style_features:  (n_samples, n_maple, size_y, size_x) feature rep derived from the style image
    :return: A scalar representing the loss due to the difference in content-image
        autocorrelations and style-image autocorrelations.
    """
    content_correlations = feature_corr(evolving_features)
    style_correlations = feature_corr(style_features)
    n = evolving_features.shape[1]
    m = evolving_features.shape[2] * evolving_features.shape[3]
    style_loss = ((content_correlations - style_correlations) ** 2).sum() / (4 * n ** 2 * m ** 2)
    return style_loss


@symbolic
def layer_content_loss(evolving_features, content_features):
    """
    :param evolving_features: (n_samples, n_maple, size_y, size_x) feature rep derived from the content image
    :param content_features:  (n_samples, n_maple, size_y, size_x) feature rep derived from the style image
    :return: A scalar representing the loss due to the difference in content-image autocorrelations and style-image autocorrelations.
    """
    return ((evolving_features[-1] - content_features[-1]) ** 2).sum() / 2


@symbolic
def get_total_loss(evolving_input, content_input, style_input, alpha=0.001, beta=1, weighting_scheme='uniform',
        style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], content_layer='conv4_2',
        force_shared_parameters = False, pooling_mode = 'max'):
    net = get_vgg_net(up_to_layer=style_layers+[content_layer], force_shared_parameters=force_shared_parameters, pooling_mode = pooling_mode)
    weights = np.ones(len(style_layers)) / len(style_layers) if weighting_scheme == 'uniform' else \
        bad_value(weighting_scheme)
    assert len(weights) == len(style_layers)
    evolving_features = net.get_named_layer_activations(evolving_input)
    content_features = net.get_named_layer_activations(content_input)  # List of (n_samples,n_maps,n_rows,n_cols) feature maps
    style_features = net.get_named_layer_activations(style_input)  # List of (n_samples,n_maps,n_rows,n_cols) feature maps
    print 'Got Features'
    content_loss = layer_content_loss(evolving_features[content_layer], content_features[content_layer])
    style_loss = sum([w * layer_style_loss(evolving_features[sl], style_features[sl]) for w, sl in zip(weights, style_layers)])
    print 'Initial Content Loss: %s' % (content_loss.ival,)
    print 'Initial Style Loss: %s' % (style_loss.ival,)
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss


@symbolic
def evolve_image(content_image, style_image, eta=0.01, init_mag=0.01, rng=None, optimizer='sgd', **loss_args):

    print "Start Symbolic"
    rng = get_rng(rng)

    evolving_input = create_shared_variable(im2feat(rng.normal(scale=init_mag, size=content_image.ishape)))
    loss = get_total_loss(
        evolving_input=evolving_input,
        content_input=im2feat(content_image),
        style_input=im2feat(style_image),
        **loss_args
    )
    optimizer = get_named_optimizer(name=optimizer, learning_rate=eta) if isinstance(optimizer, str) else optimizer
    optimizer(cost=loss, parameters=[evolving_input])
    evolving_image = feat2im(evolving_input)
    print "Stop Symbolic"
    return evolving_image

# @symbolic
# def produce_style_sequence(content_image, style_image, epsilon = 0.01, weighting_scheme = 'uniform', sylishness = 1):
#


def im2feat(im): return im.dimshuffle('x', 2, 0, 1) if isinstance(im, Variable) else np.rollaxis(im, 2, 0)[None, :, :, :]

def feat2im(feat): return feat.dimshuffle(0, 2, 3, 1)[0, :, :, :] if isinstance(feat, Variable) else np.rollaxis(feat, 0,2)[0, :, :, :]

def normalize(arr): return (arr - np.mean(arr)) / np.std(arr)


def map_to_original_cmap(float_im, mean, std):
    new_im = np.minimum(np.maximum(((float_im*std)+mean), 0), 255).astype(np.uint8)
    return new_im


def plot_liquid_style(content_image_name, style_image_name, n_steps=100, size=72, **kwargs):
    fig = plt.figure()
    raw_content_image = get_image(content_image_name, size=(size, None))
    raw_style_image = get_image(style_image_name, size=(size, None))
    content_colour_scale = np.mean(raw_content_image), np.std(raw_content_image)
    # content_image = normalize(raw_content_image).astype('float32')  # (size_y, size_x, 3) ndarray
    # style_image = normalize(raw_style_image).astype('float32')  # (size_y, size_x, 3) ndarray

    dbplot(raw_content_image, "Content", figure=fig)
    dbplot(raw_style_image, "Style", figure=fig)

    f = evolve_image.compile(
        fixed_args=dict(
            content_image=normalize(raw_content_image).astype(np.float32),
            style_image=normalize(raw_style_image).astype(np.float32),
            **kwargs),
        add_test_values = False
        )

    file_loc = get_local_path('output/{0}-{1}-{2}.gif'.format(get_current_experiment_id(), content_image_name, style_image_name), make_local_dir=True)
    with OnlineGifWriter(file_loc, fps=30) as gw:
        for i in xrange(n_steps):
            print 'Step %s' % (i,)
            im = f()
            print 'Done'
            raw_im = map_to_original_cmap(im, *content_colour_scale)
            gw.write(np.concatenate([raw_content_image, raw_im, raw_style_image], axis=1))
            # fig.savefig('%s-%s-%s' % (content_image_name, style_image_name, i))
            dbplot(im, 'Evolved Image', figure=fig)
    fig.savefig(get_local_path('output/{0}-{1}-{2}.png'.format(get_current_experiment_id(), content_image_name, style_image_name)))


ExperimentLibrary.shallow_art = Experiment(
    description="A shallow version - just works on the lowest layer.",
    function=lambda alpha=0.00001, eta=1., n_steps=300, **kwargs: plot_liquid_style(
        content_image_name='lenna',
        style_image_name='starry_night',
        content_layer='conv2_2',
        style_layers = ['conv1_1', 'conv2_1'],
        eta=eta,
        alpha = alpha,
        n_steps=n_steps,
        **kwargs
    ),
    versions = dict(
        low_res = dict(size=72),
        high_res = dict(size=256),
        high_style = dict(alpha = 0.000001, size = 256),
        higher_style = dict(alpha = 1e-8, eta=100, size = 256, n_steps=600),
        higher_style_adamax = dict(alpha = 1e-8, eta=1, size = 256, n_steps=600, optimizer = 'adam'),
        extreme_style = dict(alpha = 1e-9, eta=100, size = 256, n_steps=600),
        extreme_style_adamax = dict(alpha = 1e-9, eta=1e-1, size = 256, n_steps=600, optimizer = 'adamax'),
        extreme_style_adamax_small = dict(alpha = 1e-8, eta=1e-1, size = 128, n_steps=600, optimizer = 'adamax'),
        extreme_style_small = dict(alpha = 1e-9, eta=10, size = 128, n_steps=600),
        extreme_style_momentum_small = dict(alpha = 1e-9, size = 128, n_steps=600, optimizer = GradientDescent(eta=1, momentum=0.9)),
        extreme_style_more_momentum_small = dict(alpha = 1e-8, size = 128, n_steps=600, optimizer = GradientDescent(eta=1., momentum=0.99)),

        # extreme_style_hmc_small = dict(alpha = 1e-8, size = 128, n_steps=600, optimizer = HMCPartial(step_size=100, alpha = 0.9, temperature=10)),
        ),
    current_version='low_res',
    conclusion="""
        low_res & high_res: Stay very close to original image in form, appear to mainly change colour content.
        adamax_opt:
        high_style: Still sticks very closely to lenna
        higher_style: We begin to see a nice comprimize, with kind of Van-Gogh-esque squiggles.
        higher_style_adamax: Surprisingly different.  Lenna is recreated accurately, but in the colour scheme of starry night.
        extreme_style: Lots of pretty swirls and textures but no sign of Lenna.
        extreme_style_adamax: Lenna slowly emerges out of what initially appears to be Van-Gogh-esque squiggles.

    """
)

ExperimentLibrary.figure_2 = Experiment(
    description="Replicate Figure 2 From the paper",
    function=lambda style_image_name, alpha: plot_liquid_style(
        content_image_name='nektarfront',
        style_image_name=style_image_name,
        content_layer='conv4_2',
        style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        eta=.1,
        alpha=alpha,
        n_steps=600,
        size=256,
        force_shared_parameters = False,
        optimizer = 'sgd'
        # content_style_ratio = 1e-3,
    ),
    versions = {
        'B': dict(alpha = 1e-3, style_image_name='the_shipwreck_of_the_minotaur'),
        'C': dict(alpha = 1e-3, style_image_name='starry_night'),
        'D': dict(alpha = 1e-3, style_image_name='scream'),
        'E': dict(alpha = 1e-4, style_image_name='femme_nue_assise'),
        'F': dict(alpha = 1e-4, style_image_name='composition_vii'),
    },
    current_version = 'B',
    conclusion="""
        Hmm... still having trouble replicating results
    """
)


ExperimentLibrary.deeper = Experiment(
    description="fd.",
    function=lambda: plot_liquid_style(
        content_image_name='lenna',
        style_image_name='starry_night',
        top_layer='conv4_1',
        eta=0.001,
        alpha=0.01,
        n_steps=1000,
        size=256,
        force_shared_parameters = False,
        optimizer = 'sgd'
        # content_style_ratio = 1e-3,
    ),
    conclusion="""
        Lorem Ipsum
    """
)


ExperimentLibrary.deeeeper = Experiment(
    description="Blah Blah Blah",
    function=lambda: plot_liquid_style(
        content_image_name='manchester_newyear',
        style_image_name='starry_night',
        top_layer='conv5_1',
        eta=0.001,
        alpha=0.001,
        n_steps=1000,
        size=256,
        force_shared_parameters = False,
        optimizer = 'sgd'
        # content_style_ratio = 1e-3,
    ),
    conclusion="""
        Lorem Ipsum
    """
)

ExperimentLibrary.shallow_goodres = Experiment(
    description="Blah Blah Blah",
    function=lambda: plot_liquid_style(
        content_image_name='manchester_newyear',
        style_image_name='limbo',
        top_layer='conv5_1',
        eta=0.001,
        alpha=0.001,
        n_steps=1000,
        size=512,
        force_shared_parameters = False,
        optimizer = 'sgd'
        # content_style_ratio = 1e-3,
    ),
    conclusion="""
        Lorem Ipsum
    """
)

if __name__ == '__main__':
    exp = ExperimentLibrary.shallow_art
    exp.run()
