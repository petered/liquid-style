import numpy as np
import theano
from fileman.experiment_record import ExperimentLibrary, Experiment, get_current_experiment_id
from fileman.local_dir import get_local_path
from general.numpy_helpers import get_rng
from general.should_be_builtins import bad_value
from liquid_style.art_gallery import get_image
from liquid_style.hmc import HamiltonianMonteCarlo
from liquid_style.images2gif import OnlineGifWriter
from liquid_style.liquid_style_helpers import layer_style_loss, layer_content_loss
from liquid_style.pretrained_networks import get_vgg_net
from matplotlib import pyplot as plt
from plato.core import symbolic, create_shared_variable
from plato.tools.optimization.optimizers import GradientDescent, get_named_optimizer
from plotting.db_plotting import dbplot
from theano.gof.graph import Variable

__author__ = 'peter'

"""
Here, we try a variant on the Algorithm presented in:
A Neural Algorithm of Artistic Style
by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
http://arxiv.org/pdf/1508.06576v2.pdf

"""


@symbolic
def get_total_loss(evolving_input, content_input, style_input, content_weight=0.001, style_weight=1, weighting_scheme='uniform',
                   style_layers = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'), content_layer='conv4_2',
                   force_shared_parameters = True, pooling_mode = 'max'):
    """
    :param evolving_input: (n_samples, n_colours, size_y, size_x) shared variable representing the image to evolve.
    :param content_input: (n_samples, n_colours, size_y, size_x) array representing the content image
    :param style_input: (n_samples, n_colours, style_size_y, style_size_x) array representing the style image
    :param content_weight: The weight to put on the style.
    :param style_weight: The weight to put on content (just leave it at 1, since it's sort of redundant with alpha & learning rate)
    :param weighting_scheme: The relative weight of each style layer when computing the style loss
    :param style_layers: The Layers of the vggnet to use for style loss
    :param content_layer: The layer to use for content loss
    :param force_shared_parameters: If True, prevents constant folding (which causes slower compilation, and bug in
        theano versions below 0.8.0
    :param pooling_mode: Which type of pooling to use {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}

    :return: A vector of per-sample total losses
    """
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
    total_losses = content_weight * content_loss + style_weight * style_loss
    return total_losses

#
# def get_image_loss_function(**kwargs):
#     """
#     :param kwargs: See doc for get_total_loss, above.
#     :return: A function of the form: loss = f(evolving_input, content_input, style_input)
#     """
#     def image_loss(evolving_input, content_input, style_input):
#         return get_total_loss(evolving_input, content_input, style_input, **kwargs)
#     return image_loss
#
#
# # @symbolic
# # def evolve_image(content_image, style_image, eta=0.01, init_mag=0.01, rng=None, optimizer='sgd', temperature = 1e-6, **loss_args):
# #
# #     rng = get_rng(rng)
# #     evolving_input = create_shared_variable(im2feat(rng.normal(scale=init_mag, size=content_image.ishape)))
# #
# #     if optimizer == 'hmc':
# #         sampler = HamiltonianMonteCarlo(
# #             initial_state=evolving_input,
# #             alpha = 0.95,
# #             energy_fcn= lambda state: (1./temperature)*get_total_loss(
# #                 evolving_input=state,
# #                 content_input=im2feat(content_image),
# #                 style_input=im2feat(style_image),
# #                 **loss_args
# #                 )
# #             )
# #         sampler.update_state()
# #     else:
# #         loss = get_total_loss(
# #             evolving_input=evolving_input,
# #             content_input=im2feat(content_image),
# #             style_input=im2feat(style_image),
# #             **loss_args
# #             ).sum()
# #         optimizer = get_named_optimizer(name=optimizer, learning_rate=eta) if isinstance(optimizer, str) else optimizer
# #         optimizer(cost=loss, parameters=[evolving_input])
# #     evolving_image = feat2im(evolving_input)  # Note.. we output the variable before it's updated.
# #     return evolving_image


@symbolic
def optimize_image(content_image, style_image, image_loss_function, optimizer = GradientDescent(0.01), init_mag = 0.01,
        start_from_content = False, rng = None):
    """
    :param content_image:
    :param style_image:
    :param image_loss_function:  A function that takes
    :param optimizer: Optimizer to use (IGradientOptimizer object)
    :param init_mag: Initial magnitude of evolving image
    :return: evolving_image: A variable representing the evolving image.


    """
    rng = get_rng(rng)
    content_input = im2feat(content_image)
    evolving_input = create_shared_variable(content_input.ival if start_from_content else rng.normal(scale=init_mag, size=content_input.ishape))
    loss = image_loss_function(
        evolving_input=evolving_input,
        content_input=content_input,
        style_input=im2feat(style_image),
        ).sum()
    optimizer(cost=loss, parameters=[evolving_input])
    evolving_image = feat2im(evolving_input)  # Note.. we output the variable before it's updated.
    return evolving_image


@symbolic
def hmc_sample_image(content_image, style_image, image_loss_function, init_mag = 0.01, init_step_size = 0.01,
        temperature = 1e-6, rng = None):

    rng = get_rng(rng)
    evolving_input = create_shared_variable(im2feat(rng.normal(scale=init_mag, size=content_image.ishape)))
    sampler = HamiltonianMonteCarlo(
        initial_state=evolving_input,
        alpha = 0.95,
        initial_stepsize=init_step_size,
        energy_fcn= lambda state: (1./temperature)*image_loss_function(
            evolving_input=state,
            content_input=im2feat(content_image),
            style_input=im2feat(style_image),
            )
        )
    sampler.update_state()
    evolving_image = feat2im(evolving_input)  # Note.. we output the variable before it's updated.
    return evolving_image


def im2feat(im):
    """
    :param im: A (size_y, size_x, 3) array representing a RGB image on a [0, 255] scale
    :returns: A (1, 3, size_y, size_x) array representing the BGR image that's ready to feed into VGGNet

    """
    centered_bgr_im = im[:, :, ::-1] - np.array([103.939, 116.779, 123.68])
    feature_map_im = centered_bgr_im.dimshuffle('x', 2, 0, 1) if isinstance(centered_bgr_im, Variable) else np.rollaxis(centered_bgr_im, 2, 0)[None, :, :, :]
    return feature_map_im.astype(theano.config.floatX)


def feat2im(feat):
    """
    :param feat: A (1, 3, size_y, size_x) array representing the BGR image that's ready to feed into VGGNet
    :returns: A (size_y, size_x, 3) array representing a RGB image.
    """
    bgr_im = (feat.dimshuffle(0, 2, 3, 1) if isinstance(feat, Variable) else np.rollaxis(feat, 0, 2))[0, :, :, :]
    decentered_rgb_im = (bgr_im + np.array([103.939, 116.779, 123.68]))[:, :, ::-1]
    return decentered_rgb_im


def float2uint(im):
    return im.clip(0, 255).astype('uint8')


# def preprocess(im)


# def normalize(arr):
#     return (arr - np.mean(arr)) #/ np.std(arr)


def map_to_original_cmap(float_im, mean, std):
    new_im = np.minimum(np.maximum(((float_im*std)+mean), 0), 255).astype(np.uint8)
    return new_im


def plot_liquid_style(content_image_name, style_image_name, evolving_image_fcn, style_scale = 1, n_steps=100, size=72,):
    """
    Create a live plot, and write a GIF, of the evolving image.

    :param content_image_name: Name of the content image (see arg_gallery)
    :param style_image_name: Name of the style image (see arg_gallery)
    :param evolving_image_fcn: A function of the format - (content_image, style_image, image_loss_function) -> evolving_image
    :param n_steps:
    :param size:
    """
    fig = plt.figure()
    content_image = get_image(content_image_name, size=(size, None))
    style_image = get_image(style_image_name, size=(int(np.ceil(size*style_scale)), None))
    # content_colour_scale = np.mean(raw_content_image), np.std(raw_content_image)
    # content_image = normalize(raw_content_image).astype('float32')  # (size_y, size_x, 3) ndarray
    # style_image = normalize(raw_style_image).astype('float32')  # (size_y, size_x, 3) ndarray

    dbplot(content_image, "Content", figure=fig)
    dbplot(style_image, "Style", figure=fig)

    f = evolving_image_fcn.compile(
        fixed_args=dict(content_image=content_image, style_image=style_image),
        add_test_values = False
        )

    file_loc = get_local_path('output/{0}-{1}-{2}.gif'.format(get_current_experiment_id(), content_image_name, style_image_name), make_local_dir=True)
    with OnlineGifWriter(file_loc, fps=30) as gw:
        for i in xrange(n_steps):
            print 'Step %s' % (i,)
            evolving_im = f()
            print 'Done'
            # raw_im = map_to_original_cmap(im, *content_colour_scale)
            gw.write(np.concatenate([content_image, float2uint(evolving_im), style_image], axis=1))
            # fig.savefig('%s-%s-%s' % (content_image_name, style_image_name, i))
            dbplot(evolving_im, 'Evolving Image', figure=fig)
    fig.savefig(get_local_path('output/{0}-{1}-{2}.png'.format(get_current_experiment_id(), content_image_name, style_image_name)))


ExperimentLibrary.shallow_art = Experiment(
    description="A shallow version - just works on the lowest layer.",
    function=lambda alpha=1e-5, n_steps=300, optimizer = 'sgd', eta = 0.01, size = 128: plot_liquid_style(
        content_image_name='lenna',
        style_image_name='starry_night',
        size = size,
        n_steps=n_steps,
        evolving_image_fcn= optimize_image.partial(
            image_loss_function = get_total_loss.partial(
                style_layers = ['conv1_1', 'conv2_1'],
                content_layer='conv2_2',
                content_weight= alpha,
                ),
            optimizer = get_named_optimizer(optimizer, eta),
            ),
        ),
    versions = dict(
        low_res = dict(size=72),
        high_res = dict(size=256),
        high_style = dict(alpha = 1e-5, size = 256, eta = 1e-1),
        high_style_adamax = dict(alpha = 1e-5, size = 256, eta = 1e0, optimizer = 'adamax'),
        higher_style = dict(alpha = 1e-8, eta=100, size = 256, n_steps=600),

        higher_style_adamax = dict(alpha = 1e-8, eta=1, size = 256, n_steps=600, optimizer = 'adamax'),
        extreme_style = dict(alpha = 1e-9, eta=100, size = 256, n_steps=600),
        extreme_style_adamax = dict(alpha = 1e-9, eta=1e0, size = 256, n_steps=600, optimizer = 'adamax'),
        extreme_style_adamax_small = dict(alpha = 1e-8, eta=1e-1, size = 128, n_steps=600, optimizer = 'adamax'),
        extreme_style_small = dict(alpha = 1e-9, eta=10, size = 128, n_steps=600),
        extreme_style_momentum_small = dict(alpha = 1e-9, size = 128, n_steps=600, optimizer = GradientDescent(eta=1, momentum=0.9)),
        extreme_style_more_momentum_small = dict(alpha = 1e-8, size = 128, n_steps=600, optimizer = GradientDescent(eta=1., momentum=0.99)),

        extreme_style_hmc_small = dict(alpha = 1e-8, size = 128, n_steps=600, optimizer = 'hmc'),
        ),
    current_version='high_style_adamax',
    conclusion="""
        low_res & high_res: Stay very close to original image in form, appear to mainly change colour content.
        adamax_opt:
        high_style: Still sticks very closely to lenna
        higher_style: We begin to see a nice comprimize, with kind of Van-Gogh-esque squiggles.  But the squiggles seem
            to fade over time, and just a blue-coloured Lenna remains.
        higher_style_adamax: Surprisingly different.  Lenna is recreated accurately, but in the colour scheme of starry night.
            The magnitude of the evolving image is much higher than that of the content image.
        extreme_style: Lots of pretty swirls and textures but no sign of Lenna.
        extreme_style_adamax: Lenna slowly emerges out of what initially appears to be Van-Gogh-esque squiggles.
    """
)

ExperimentLibrary.hmc_experiments = Experiment(
    description="Lets get that dreamy HMC sampling going",

    function=lambda alpha=0.00001, eta=1., n_steps=300, content_layer='conv2_2', style_layers = ['conv1_1', 'conv2_1'], **kwargs: plot_liquid_style(
        content_image_name='lenna',
        style_image_name='starry_night',
        content_layer='conv2_2',
        style_layers = ['conv1_1', 'conv2_1'],
        eta=eta,
        optimizer = 'hmc',
        alpha = alpha,
        n_steps=n_steps,
        **kwargs
    ),
    versions=dict(
        shallow_small = dict(alpha = 1e-8, size = 128, n_steps=600, temperature = 1e-6, ),
        deep = dict(content_layer='conv4_2', style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                alpha = 1e-8, size = 256, n_steps=600, temperature = 1e-6,)
    ),
    current_version = 'shallow_small'
    )


ExperimentLibrary.figure_2 = Experiment(
    description="Replicate Figure 2 From the paper",
    # function=lambda style_image_name, alpha: plot_liquid_style(
    #     content_image_name='nektarfront',
    #     style_image_name=style_image_name,
    #     content_layer='conv4_2',
    #     style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
    #     eta=.1,
    #     alpha=alpha,
    #     n_steps=600,
    #     size=256,
    #     force_shared_parameters = False,
    #     optimizer = 'sgd'
    #     # content_style_ratio = 1e-3,
    # ),
    function=lambda style_image_name, alpha, n_steps = 600, size = 512, optimizer = 'adam', eta=1e0: plot_liquid_style(
        content_image_name='nektarfront',
        style_image_name=style_image_name,
        size = size,
        n_steps=n_steps,
        evolving_image_fcn= optimize_image.partial(
            image_loss_function = get_total_loss.partial(
                style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                content_layer='conv4_2',
                # content_weight= alpha,
                content_weight= 5e0,
                style_weight = 1e3,
                ),
            optimizer = get_named_optimizer(optimizer, eta),
            ),
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
    exp = ExperimentLibrary.figure_2
    exp.run()
