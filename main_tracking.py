from src.utils_tracking import (
    HeadPoseDataset, split_and_load, check_GPU_availability, fix_seed, count_parameters
)
from src.simple_tracking_model import SimpleHeadPoseModel
from src.train_tracking import train, check_accuracy
from src.model import FSANet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW


# Define hyperparameters
BATCH_SIZE = 8

LR = 1e-3
STEP_SIZE = 15
GAMMA = 0.8
EPOCHS = 10


if __name__ == "__main__":
    # Fix seed so the results can be reproduced later
    fix_seed(827)

    # model = FSANet(var=False)
    # print(count_parameters(model))
    # raise

    device = check_GPU_availability()

    head_pose_dataset = HeadPoseDataset(maxlen=9*10, normalise=True)
    loader_train, loader_test = split_and_load(head_pose_dataset, test_ratio=0.1, batch_size=BATCH_SIZE)

    model = SimpleHeadPoseModel(hidden_layer=1024, dropout=0.15)
    print("Number of trainable parameters in the model: ", count_parameters(model))
    
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    train(model, loader_train, optimizer, loss_fn, scheduler, EPOCHS, device, loader_test)

    acc = check_accuracy(model, loader_test, device)





