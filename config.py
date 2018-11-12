import argparse
import sys
import os

BATCH_SIZE = 32
BN_WEIGHT_DECAY = 0.9997
CKPT = -1
DATA_DIRECTORY = './data'
DATA_NAME = 'omniglot'
DISTANCE = 'cosine' # 'euclidean'
GAMMA = 0.1
EXP_NAME = 'char'
INPUT_SIZE = 28
LAMBDA = 0.001
LEARNING_RATE = 0.0001
LOG_DIR = './log'
MOMENTUM = 0.9
NUM_EPOCHS = 100
NUM_GROUPS = 3
NUM_WAYS = 5
NUM_SHOTS = 1
NUM_TARGET_EXAMPLES = 20
NUM_D_ITERS = 1
NUM_G_ITERS = 1
POWER = 0.9
RANDOM_SEED = 1992
RESTORE_FROM = None
SNAPSHOT_DIR = './saved_models'
SOURCE = 'omniglot'
TARGET = 'emnist_20'
SPLIT_NAME = 'train'
WEIGHT_DECAY = 1e-4

parser = argparse.ArgumentParser(description="DAOSL")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Number of images sent to the network in one step.")
parser.add_argument("--bn-weight-decay", type=float, default=BN_WEIGHT_DECAY,
                    help="Regularisation parameter for batch norm.")
parser.add_argument("--ckpt", type=int, default=CKPT, 
                    help="Checkpoint to restore.")
parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                    help="Path to the directory containing the data.")
parser.add_argument("--data-name", type=str, default=DATA_NAME,
                    help="Data name.")
parser.add_argument("--distance", type=str, default=DISTANCE,
                    help="Distance.")
parser.add_argument("--exp-name", type=str, default=EXP_NAME,
                    help="Name of the experiment.")
parser.add_argument("--freeze-bn", action="store_true",
                    help="Whether to freeze batch norm params.")
parser.add_argument("--gamma", type=float, default=GAMMA)
parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                    help="height and width of images.")
parser.add_argument("--la", type=float, default=LAMBDA)
parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                    help="Base learning rate for training with polynomial decay.")
parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                    help="Path to the directory containing the log.")
parser.add_argument("--momentum", type=float, default=MOMENTUM,
                    help="Momentum component of the optimiser.")
parser.add_argument("--not-restore-last", action="store_true",
                    help="Whether to not restore last (FC) layers.")
parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                    help="Number of epochs.")
parser.add_argument("--num-groups", type=int, default=NUM_GROUPS)
parser.add_argument("--num-shots", type=int, default=NUM_SHOTS,
                    help="Number of shots.")
parser.add_argument("--num-target-examples", type=int, 
                    default=NUM_TARGET_EXAMPLES,
                    help="Number of exampels for each class of the target domain.")
parser.add_argument("--num-ways", type=int, default=NUM_WAYS,
                    help="Number of ways.")
parser.add_argument("--num-d-iters", type=int, default=NUM_D_ITERS,
                    help="Number of iterations for disc.")
parser.add_argument("--num-g-iters", type=int, default=NUM_G_ITERS,
                    help="Number of iterations for gen.")
parser.add_argument("--power", type=float, default=POWER,
                    help="Decay parameter to compute the learning rate.")
parser.add_argument("--random-mirror", action="store_true",
                    help="Whether to randomly mirror the inputs during the training.")
parser.add_argument("--random-scale", action="store_true",
                    help="Whether to randomly scale the inputs during the \
                    training.")
parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                    help="Random seed to have reproducible results.")
parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                    help="Where to save snapshots of the model.")
parser.add_argument("--source", type=str, default=SOURCE,
                    help="Name of the dataset.")
parser.add_argument("--split-name", type=str, default=SPLIT_NAME,
                    help="Split name.")
parser.add_argument("--target", type=str, default=TARGET,
                    help="Name of the dataset.")
parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                    help="Regularisation parameter for L2-loss.")

args = parser.parse_args()