import torch
import numpy as np
import random

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
# Number of epochs during architecture evaluation
NUM_EVAL_EPOCHS = 10

def IS_ARCH_FIXED():
    return True if ARCH_TYPE == 'fixed' else False

# set seed
manual_seed = random.randint(1, 10000)
# manual_seed = 1768
print( 'seed:', manual_seed )
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)

# Put your other constants here.
