import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class SimpleHeadPoseModel(nn.Module):
    """
    Simple head pose detection model using LSTM
    """
    def __init__(self, input_size=3, label_size=3, hidden_layer=10, dropout=0.1):
        """
        :param input_size: number of features in the data
        :param label_size: number of labels
        :param hidden_layer: number of neurons in linear layers
        :param softmax: whether to softmax the output or not
        """
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_layer, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_layer, 2 * hidden_layer)
        self.fc2 = nn.Linear(2 * hidden_layer, 2 * hidden_layer)
        self.fc3 = nn.Linear(2 * hidden_layer, label_size)

        self.dropout = nn.Dropout(dropout)
        self.softmax = F.softmax

    def forward(self, inputs, softmax=False):
        """
        :param inputs: shape (batch_size, seq_length, input_size)
        """
        lstm_out, _ = self.lstm(inputs.float()) # -> (batch_size, sequence_length, num_directions*hidden_size)
        pooled = F.max_pool1d(lstm_out.transpose(1, 2), lstm_out.shape[1]).squeeze(2)  # -> (batch_size, num_directions*hidden_size)
        out = self.dropout(pooled)
        out = F.relu(self.fc1(out))
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        if softmax:
            out = self.softmax(out)

        return out