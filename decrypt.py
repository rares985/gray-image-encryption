"""
Decryption module
----------------

Functions: undiffuse, decrypt

"""
from copy import deepcopy
import numpy as np
from scipy import sparse
from utils import blocks, merge, MASTER_KEY, indices, shift, diffusion_characteristics
from utils import row_transform, column_transform, substitution_characteristics

def undiffuse(diffused, xs_rnd, ys_rnd):
    """
    Traverses the matrix in a zig-zag pattern, starting
    at index (xr, yr) and places the elements sequentially
    by columns.

    Parameters
    ---------
    block: Block to be undiffused
    xr: starting x
    yr: starting y

    Returns
    -------
    undiffused block
    """
    indexes = indices(diffused.shape[0])
    raveled_array = np.roll(
        np.transpose(diffused).ravel(),
        (shift(diffused.shape[0], diffused.shape[0])[(xs_rnd, ys_rnd)] - 1),
        axis=0
    )
    diagonals = np.array([raveled_array[indexes[i]: indexes[i + 1]]
                          for i in range(len(indexes) - 1)])
    for i, diagonal in enumerate(diagonals):
        diagonals[i] = diagonal[::2 * (i % 2) -1]
    return np.transpose(
        np.flipud(
            sparse.diags(diagonals, range(1-diffused.shape[0], diffused.shape[0])).toarray()))


def decrypt(img, key=MASTER_KEY, permute=False, rounds=8):
    """
    Performs decryption of image.

    Parameters
    ---------

    img: The image to be decrypted
    key: The key which was used for encryption
    clip_each_round: Whether the image is trimmed back to original
    size after each round of diffussion or at the end.abs
    rounds: The number of diffusion rounds applied on the image.

    Returns
    ------
    The decrypted image

    """
    height, width = img.shape
    decrypted_img = deepcopy(img)

    # First perform substitution
    if permute:
        bss = substitution_characteristics(key)
        for  rnd in range(rounds - 1, -1 -1):
            blk_list, shape = blocks(decrypted_img, bss[rnd])
            substituted_blocks = []
            for blk in blk_list:
                substituted_blocks.append(row_transform(column_transform(blk)))
            decrypted_img = merge(substituted_blocks, shape)[:height, :width]
    else:
        pass

    # Then perform difussion
    bss, xrs, yrs = diffusion_characteristics(key)
    for rnd in range(rounds - 1, -1, -1):
        blk_list, shape = blocks(decrypted_img, bss[rnd])
        undiffused_blocks = []
        for blk in blk_list:
            undiffused_blocks.append(undiffuse(blk, xrs[rnd], yrs[rnd]))
        decrypted_img = merge(undiffused_blocks, shape)[:height, :width]


    return decrypted_img
