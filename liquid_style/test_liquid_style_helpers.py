import numpy as np
from liquid_style.liquid_style_helpers import batch_gram_matrix


def test_batched_gram_matrix():

    data = np.random.randn(3, 4, 5)

    expected_result = np.einsum('ijk,imk->ijm', data, data)

    f1 = batch_gram_matrix.partial(mode = 'mem').compile()
    assert np.allclose(expected_result, f1(data))

    f2 = batch_gram_matrix.partial(mode = 'middle').compile()
    assert np.allclose(expected_result, f2(data))


if __name__ == '__main__':

    test_batched_gram_matrix()
