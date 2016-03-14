from plato.core import symbolic
from theano import tensor as tt


@symbolic
def batch_gram_matrix(x, mode = 'middle'):
    """
    Compute the Gram-Matix for each of a batch of collections of vectors.

    :param x: A matrix of shape (n_samples, n_vecs, n_dims)
    :param mode: Controls the trade-off between "sequential and memory saving" and "parallel but memory intensive".
        'seq': Means do the dot-products sequentially.
        'mem': Means go crazy with memory, but very parallelizable
        'middle': A middle ground between the other two options
    :return: A batch of Gram-Matrices (n_samples, n_vecs, n_vecs)
    """
    assert mode in ('mem', 'seq', 'middle')
    if mode == 'mem':
        return (x[:, None, :, :] * x[:, :, None, :]).sum(axis=3)
    elif mode == 'middle-2':
        @symbolic
        def mmul(v, m):
            """
            :param v: A (n_samples, n_dims) tensor
            :param m: A (n_samples, n_vecs, n_dims) tensor
            :returns: A (n_samples, n_vecs) tensor
            """
            return tt.batched_dot(m, v)
        gram_matrix = mmul.scan(sequences = [x.dimshuffle(1, 0, 2)], non_sequences = [x]) # (n_vecs, n_samples, n_vecs)
        return gram_matrix.dimshuffle(1, 0, 2)  # (n_samples, n_vecs, n_vecs)
    elif mode == 'middle':
        @symbolic
        def do_a_row(i):
            """
            :param v: A (n_samples, n_dims) tensor
            :param m: A (n_samples, n_vecs, n_dims) tensor
            :returns: A (n_samples, n_vecs) tensor
            """
            return tt.batched_dot(x, x[:, i, :])

        # gram_matrix = mmul.scan(sequences = [x.dimshuffle(1, 0, 2)], non_sequences = [x]) # (n_vecs, n_samples, n_vecs)
        gram_matrix = do_a_row.scan(sequences = [tt.arange(x.shape[1])]) # (n_vecs, n_samples, n_vecs)
        return gram_matrix.dimshuffle(1, 0, 2)  # (n_samples, n_vecs, n_vecs)


    elif mode == 'seq':
        raise NotImplementedError("Have not yet implemented the batch gram matrix in the full sequential form.")


@symbolic
def feature_corr(feature_representation, mode = 'middle'):
    """
    Find the correlation within a feature map, pooled over pixels.
    See equation 3 from the paper.

    :param feature_representation: A shape (n_samples,n_maps,size_y,size_x) feature representation
    :return: A shape (n_samples, n_maps, n_maps) matrix of correlations.
    """
    flattened_rep = feature_representation.flatten(3)  # (n_samples, n_maps, n_pixels)
    return batch_gram_matrix(flattened_rep, mode = mode)
    # tt.batched_tensordot()
    # tt.batched_dot
    # return (flattened_rep[:, None, :, :] * flattened_rep[:, :, None, :]).sum(axis=3)


@symbolic
def layer_style_loss(evolving_features, style_features):
    """
    Find the style-loss given the features of the content and style images for a given layer.
    See equation 4 from the paper.
    :param evolving_features: (n_samples, n_maple, size_y, size_x) feature rep derived from the content image
    :param style_features:  (n_samples, n_maple, size_y, size_x) feature rep derived from the style image
    :return: A vector of per-sample losses representing the loss due to the difference in content-image
        autocorrelations and style-image autocorrelations.
    """

    content_correlations = feature_corr(evolving_features)
    style_correlations = feature_corr(style_features)
    n = evolving_features.shape[1]
    m = evolving_features.shape[2] * evolving_features.shape[3]
    style_losses = ((content_correlations - style_correlations) ** 2).flatten(2).sum(axis=1) / (4 * n ** 2 * m ** 2)
    return style_losses


@symbolic
def layer_content_loss(evolving_features, content_features):
    """
    :param evolving_features: (n_samples, n_maple, size_y, size_x) feature rep derived from the content image
    :param content_features:  (n_samples, n_maple, size_y, size_x) feature rep derived from the style image
    :return: A scalar representing the loss due to the difference in content-image autocorrelations and style-image autocorrelations.
    """
    return ((evolving_features - content_features) ** 2).flatten(2).sum(axis=1) / 2