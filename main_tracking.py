from src.utils_tracking import (
    HeadPoseDataset, split_and_load, check_GPU_availability, fix_seed
)
from src.simple_tracking_model import SimpleHeadPoseModel
from src.train_tracking import train

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW


# Define hyperparameters
BATCH_SIZE = 4

LR = 1e-2
STEP_SIZE = 10
GAMMA = 0.8
EPOCHS = 4


if __name__ == "__main__":
    # Fix seed so the results can be reproduced later
    fix_seed(0)

    device = check_GPU_availability()

    head_pose_dataset = HeadPoseDataset()
    loader_train, loader_test = split_and_load(head_pose_dataset, batch_size=BATCH_SIZE)

    model = SimpleHeadPoseModel()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    train(model, loader_train, optimizer, loss_fn, scheduler, EPOCHS, device)





