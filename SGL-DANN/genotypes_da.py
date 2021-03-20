from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal reduce')

NORMAL_PRIMITIVES = [
    'none',
    # 'max_pool_2x2',
    # 'avg_pool_2x2',
    'skip_connect',
    'conv_3x3',
    'conv_5x5',
    # 'dil_conv_3x3',
    # 'dil_conv_5x5'
]

REDUCE_PRIMITIVES = [
    'max_pool_2x2',
    'avg_pool_2x2',
]
