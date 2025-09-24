import torch

DATA_DIR = './data'
MODEL_SAVE_PATH = './best_model_efficientnet_finetuned.pth'

IMAGE_MODEL_NAME = 'tf_efficientnet_b0'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 25

IMAGE_LR = 1e-4
HEAD_LR = 1e-3

EMBEDDING_DIM = 128

SEED = 42