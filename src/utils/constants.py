import os

DATA_DIR = '.data'
MODELS_DIR = '.models'

RES_TRAIN_BELIEF_DATASET_PATH = os.path.join(DATA_DIR, 'res_train_belief_dataset.npz')
RES_VAL_BELIEF_DATASET_PATH = os.path.join(DATA_DIR, 'res_val_belief_dataset.npz')
SPY_TRAIN_BELIEF_DATASET_PATH = os.path.join(DATA_DIR, 'spy_train_belief_dataset.npz')
SPY_VAL_BELIEF_DATASET_PATH = os.path.join(DATA_DIR, 'spy_val_belief_dataset.npz')

RES_BELIEF_MODEL_PATH = os.path.join(MODELS_DIR, 'res_belief_16_30_10_v1.pt')
SPY_BELIEF_MODEL_PATH = os.path.join(MODELS_DIR, 'spy_belief_16_30_10_v1.pt')