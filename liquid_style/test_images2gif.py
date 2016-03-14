from liquid_style.images2gif import GifWriter, writeGif, OnlineGifWriter, checkImages
from fileman.file_getter import get_file
from fileman.local_dir import get_local_path

__author__ = 'peter'
import numpy as np

def test_images2gif():

    ix = np.arange(50)[:, None, None]
    base_im = np.zeros((50, 60, 3), dtype = int)

    # images = (np.random.RandomState(3).rand(10, 150, 190, 3) * 256).astype(int)
    images = [base_im + (ix+5*i)%100 for i in xrange(100)]
    file_path = get_local_path('tests/_test_gif_file.gif', True)

    # Works, but not online
    # writeGif(
    #     filename = file_path,
    #     images = images,
    #     )

    with OnlineGifWriter(filename=file_path) as gw:
        for im in images:
            gw.write(im)


if __name__ == '__main__':
    test_images2gif()
