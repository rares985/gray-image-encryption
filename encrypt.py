"""
Encryption module
-----------------

Functions: encrypt, diffuse

"""

from copy import deepcopy
import numpy as np
from utils import shift, diffusion_characteristics, blocks, merge, MASTER_KEY, substitution_characteristics
from utils import row_transform, column_transform

def diffuse(block, xs_rnd, ys_rnd):
    """
    Traverses the matrix in a zig-zag pattern, starting
    at index (xr, yr) and places the elements sequentially
    by columns.

    Parameters
    ---------
    block: Block to be diffused
    xs_rnd: starting x
    ys_rnd: starting y

    Returns
    -------
    Diffused block
    """
    return np.transpose(
        np.split(
            np.roll(
                np.concatenate([
                    np.diagonal(block[::-1, :], k)[::(2 * (k % 2) - 1)]
                    for k in range(1-block.shape[0], block.shape[1])
                ]),
                -(shift(block.shape[0], block.shape[1])[(xs_rnd, ys_rnd)] - 1),
                axis=0
            ),
            block.shape[1]
        )
    )

def encrypt(img, key=MASTER_KEY, permute=False, rounds=8):
    """
    Performs encryption of image.

    Parameters
    ---------

    img: The image to be encrypted
    key: The key which will be used for encryption
    clip_each_round: Whether the image is trimmed back to original
    size after each round of diffussion or at the end.abs
    rounds: The number of diffusion rounds applied on the image.

    Returns
    ------
    The encrypted image

    """
    height, width = img.shape
    encrypted_image = deepcopy(img)
    bss, xrs, yrs = diffusion_characteristics(key)

    # First perform diffusion
    for rnd in range(0, rounds):
        blk_list, shape = blocks(encrypted_image, bss[rnd])
        diffused_blocks = []
        for blk in blk_list:
            diffused_blocks.append(diffuse(blk, xrs[rnd], yrs[rnd]))
        encrypted_image = merge(diffused_blocks, shape)[:height, :width]

    # Then perform substitution
    if permute:
        bss = substitution_characteristics(key)
        for rnd in range(0, rounds):
            blk_list, shape = blocks(encrypted_image, bss[rnd])
            substituted_blocks = []
            for blk in blk_list:
                substituted_blocks.append(column_transform(row_transform(blk)))
            encrypted_image = merge(substituted_blocks, shape)[:height, :width]

    else:
        pass

    return encrypted_image