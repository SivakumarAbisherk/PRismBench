import torch.optim as optim

SEED = 1

EPOCHES = 50

BATCH_SIZE = 32

OPTIMIZER = {
    "adam": optim.Adam,
    "adamw": optim.AdamW
}

LABEL_THRESHOLD = 0.4

PARAM_GRID = {
    "lr":[1e-3, 5e-4],
    "weight_decay": [1e-4, 1e-3],
    "hidden_dims":[
        (256, 128, 64), 
        (128, 64, 32)
        ],
    "drop_out":[
        (0.1, 0.1, 0.2), 
        # (0.1, 0.3, 0.5)
        ],
}

EARLY_STOPPING = 10