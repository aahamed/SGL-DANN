import torch

ROOT_STATS_DIR = './experiment_data'
NUM_CLASSES = 10
NUM_LEARNERS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# options are Adam and Sgd
OPTIM = 'Adam'

# Use fixed architecture for DANN ( only for debugging )
ARCH_FIXED = False
# Feature Extractor architecture type. Options are:
# fixed, darts, darts-simple
ARCH_TYPE = 'darts-simple'

def IS_ARCH_FIXED():
    return True if ARCH_TYPE == 'fixed' else False

# Put your other constants here.
