import torch
from torch.utils.data import TensorDataset, Subset, random_split, DataLoader
from torch._utils import _accumulate

import pandas as pd
import numpy as np
import random
from os import listdir
from os.path import isfile, join
from pathlib import Path


ROOT_PATH = str(Path(__file__).absolute().parent.parent)
HEAD_POSE = ["nod", "shake", "other"]
EULER_ANGLES = ["yaw", "pitch", "roll"]

class HeadPoseDataset(TensorDataset):
    """
    Custom torch.utils.data.Dataset for head pose data
    """
    def __init__(self, file_list=None, normalise=True):
        """
        Read head pose data and create a custom tensor dataset. The dataset contains a tuple of
        two tensors of shape:
            :input: (sample_size, frame_length, euler_angle)
            :pose: (sample_size) (label)
        where sample_size is the number of samples in the dataset, frame_length is the number of
        data points (timestamp) each sample contains (how long the data has been collected for), 
        and euler_angle is the number of euler angles (=3) collected in the data (i.e. features)

        :param file_list: list of files to use to load the data. If not given use all files
            in collected_data directory
        :param normalise: boolean, default to True; normalise the data by subtracting the mean
            It does not normalise the variance as this can be useful for the model
        """
        super().__init__()

        # Load all files inside collected data unless specified otherwise
        if not file_list:
            file_list = [
                f for f in listdir(f"{ROOT_PATH}/src/collected_data") if (
                    isfile(join(f"{ROOT_PATH}/src/collected_data", f))
                )
            ]
        
        data = {angle: [] for angle in EULER_ANGLES}
        data["pose"] = []

        self.label_mapping = {pose: i for i, pose in enumerate(HEAD_POSE)}

        for pose in HEAD_POSE:
            pose_list = [f for f in file_list if pose in f]
            for f in pose_list:
                df = pd.read_csv(join(f"{ROOT_PATH}/src/collected_data", f))
                for angle in EULER_ANGLES:
                    angle_data = df[angle].values
                    if normalise:
                        angle_data = angle_data / np.mean(angle_data)
                    data[angle].append(angle_data)
                data["pose"].append(self.label_mapping[f.split("_")[0]])

        self.tensors = (
            torch.tensor([data["yaw"], data["pitch"], data["roll"]]).transpose(0, 1).transpose(1, 2),
            torch.tensor(data["pose"]),
        )

def fix_seed(seed=0):
    """
    Fix seed for torch and numpy to produce consistent results
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed_all(seed)

def split_and_load(dataset, test_ratio=0.2, batch_size=16):
    """
    Splits the dataset into train and test with given test_ratio. 
    It then creates dataloaders for each with batch_size.
    """
    N_total = len(dataset)
    N_test = int(N_total * 0.2)
    lengths = [N_total - N_test, N_test]
    dataset_train, dataset_test = random_split(dataset, lengths)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size)

    return loader_train, loader_test

def check_GPU_availability(cuda_n=0):
    """
    Checks if GPU is available

    :return: avilable device name
    """
    if torch.cuda.is_available(): # Check we're using GPU
        torch.backends.cudnn.deterministic = True
        device = 'cuda:%s' %(cuda_n)
    else:
        device = 'cpu'

    return device


if __name__ == "__main__":
    fix_seed(0)
    head_pose_dataset = HeadPoseDataset()
    loader_train, loader_test = split_and_load(head_pose_dataset)