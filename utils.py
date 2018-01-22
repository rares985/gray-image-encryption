"""
Utilities module; provides various functions
used by encryption/decryption.

Functions
---------
hex2dec, matrix_diagonals, indices
generate_key, shift, diffusion_characteristics
blocks, merge
"""
from random import choice
import numpy as np

DEFAULT_KEY_SIZE = 32
MASTER_KEY = '80b33216c772547c5b0b34dc6adf55d9'

HEX_DIGIT_MAP = {str(i) : i for i in range(0, 10)}
HEX_DIGIT_MAP.update({dig : ord(dig) - ord('a') + 10 for dig in 'abcdef'})

def hex2dec(digit):
    """
    Converts a hexadecimal digit to its representation
    in base 10.

    Parameters
    ----------

    digit: The hex digit to be converted.

    Returns
    -------

    Decimal representation of the digit.
    """
    return HEX_DIGIT_MAP[digit]

def matrix_diagonals(arr):
    """
    Computes a matrix's diagonals.

    Parameters
    ----------

    arr: The matrix whose diagonals we want

    Returns
    -------
    A list of the matrix's diagonals
    """
    return [np.diagonal(arr[::-1], k) for k in range(1-arr.shape[0], arr.shape[0])]

def indices(size):
    """
    Computes start and end indexes for each
    diagonal in the zig-zag array.

    Parameters
    ----------
    n: Size of the array

    Returns
    -------
    List containing indexes at which the zig-zag
    array should be split.
    """
    diagonals = matrix_diagonals(np.random.random((size, size)))
    return np.insert(
        np.cumsum(
            np.array(
                list(map(np.size, diagonals))
            )
        ), 0, 0)

def generate_key(key_size=DEFAULT_KEY_SIZE):
    """
    Generates a random hexstring key of specified
    size.

    Parameters
    ----------

    key_size: The size (in bytes) of the key that
    we want to be generated.

    Returns
    -------

    String of hex digits, representing the key.

    """
    return "".join([choice(list(HEX_DIGIT_MAP.keys())) for _ in range(key_size)])

def shift(height, width):
    """
    Computes the index of elements in a zig-zag array.
    """
    def comparator(pos):
        """
        Comparator to order indexes for zig-zag matrix
        """
        x, y = pos
        return (x + y, -y if (x + y) % 2 else y)

    ordered_indexes = sorted(((x, y) for x in range(width) for y in range(height)),
                             key=comparator)
    ret = {index: n for n, index in enumerate(ordered_indexes)}
    return ret

def diffusion_characteristics(key):
    """
    Computes the diffusion process characteristics,
    based on the specified key.

    Parameters
    ----------

    key: The key from which characteristics will be generated.

    Returns
    -------

    Tuple containing three lists: block sizes, starts on x,
    starts on y.

    """
    
    bss = []
    xrs = []
    yrs = []

    for rnd in range(1, 9):
        x = 0
        y = 0
        b = 0

        for p in range(1, 5):
            index = 4 * (rnd - 1) + p
            b += hex2dec(key[index - 1])

        for p in range(1, 4):
            index = 4 * (rnd - 1) + p
            x += hex2dec(key[index - 1])

        for p in range(2, 5):
            index = 4 * (rnd - 1) + p
            y += hex2dec(key[index - 1])
        # Y is from [4r-2: 4r]
        # B is from [4r-3: 4r]

        bss.append(b)
        xrs.append(x)
        yrs.append(y)

    return bss, xrs, yrs

def substitution_characteristics(key):
    """
    Computes substitution process characteristics,
    based on the provided key.

    Parameters
    ----------
    key: The key to be used as seed.

    Returns
    -------
    List containing the block sizes for each round.
    """

    bss = []
    for rnd in range(1, 9):
        b = 0
        for p in range(1, 4):
            index = 4 * (8 - rnd) + p
            b += hex2dec(key[index-1])
        bss.append(b)

    return bss

def blocks(matrix, block_size):
    """
    Splits a 2D array into square blocks.
    Also zero-pads the matrix to the nearest
    greater multiple of block_size on both axes.

    Parameters
    ----------

    matrix: The 2D array to be split
    block_size: Size of blocks

    Returns
    -------
    Tuple containing an 1D array containing the blocks and the
    matrix size after padding(in blocks)
    """
    hpad = 0
    vpad = 0
    height, width = matrix.shape

    # Compute padding size on both axis
    if height % block_size != 0:
        vpad = block_size - height % block_size
    if width % block_size != 0:
        hpad = block_size - width % block_size

    # Pad the matrix
    out = np.pad(matrix, ((0, vpad), (0, hpad)), 'constant', constant_values=0)

    # Split matrix and return
    return (np.array([out[i: i+block_size, j: j+block_size]
                      for i in range(0, out.shape[0] - block_size + 1, block_size)
                      for j in range(0, out.shape[1] - block_size + 1, block_size)]),
            (out.shape[0] // block_size, out.shape[1] // block_size))

def merge(blks, shape):
    """
    Merges blocks according to specified shape.

    Parameters
    ----------

    blks: List containing the blocks to be merged
    shape: The shape of the padded matrix(in blocks)

    Returns
    -------

    Merged 2D array.
    """
    return np.vstack(
        tuple(
            [np.hstack(tuple(blks[i:i + shape[0]])) for i in range(0, len(blks), shape[0])]
        )
    )

def row_transform(block):
    """
    Performs row transformation on a block. That is, sutbracts
    a(i, j) = max(row(i)) - a(i, j)
    """
    for i in range(block.shape[0]):
        max = np.amax(block[i, :])
        block[i, :][block[i, :] < max] = max - block[i, :][block[i, :] < max]

    return block

def column_transform(block):
    """
    Performs column transformation on a block. That is, subtracts
    a(i,j) = max(col(j)) - a(i, j)
    """
    for i in range(block.shape[0]):
        max = np.amax(block[:, i])
        block[:, i][block[:, i] < max] = max - block[:, i][block[:, i] < max]

    return block
