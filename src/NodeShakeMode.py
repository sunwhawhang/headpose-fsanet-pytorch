"""
original copyright: github.com/saraheee/LerntiaControl
"""
from FileReader import ConfigFileReader
from myhmm import MyHmm
import os
from collections import deque
import numpy as np

numf = 20  # frame number for nod and shake detection
nod_diff_eps = 150  # shift in weighted difference values of head nods
shake_diff_eps = 200  # shift in weighted difference values of head shakes
nod_relative_diff_eps = 0.2  # shift in weighted difference values of head nods
shake_relative_diff_eps = 0.1  # shift in weighted difference values of head shakes
nothing_num = 5  # number of times nothing is detected before a nod or shake is recognized


def set_config_parameters():
    """
    Set config parameters for the 2-gestures mode.

    :return: none
    """
    global numf, nod_diff_eps, shake_diff_eps
    f = ConfigFileReader()

    value = f.read_int('nod_shake_mode', 'numf')
    numf = value if value != -1 else numf

    value = f.read_int('nod_shake_mode', 'nod_diff_eps')
    nod_diff_eps = value if value != -1 else nod_diff_eps

    value = f.read_int('nod_shake_mode', 'shake_diff_eps')
    shake_diff_eps = value if value != -1 else shake_diff_eps


class NodShakeMode:
    """
    A 2-gestures mode for performing key events through detecting head nods and head shakes.

    """

    def __init__(self, prev_data, data):
        """
        The constructor that sets the initialization parameters for the 2-gestures mode.

        :param prev_data:  the data of previous frames
        :param data: the data of the active frame
        """
        self.prev_data = prev_data
        self.data = data
        self.nod_detected = False
        self.shake_detected = False
        self.nothing = nothing_num
        # set_config_parameters()

    def set_data(self, prev_data, data):
        """
        Set data for the analysis of head nods and head shakes.

        :param prev_data: the data of previous frames
        :param data: the data of the active frame
        :return: none
        """
        self.prev_data = prev_data
        self.data = data

    def apply(self):
        """
        The application method that detects head nods and shakes. If a nod or a shake is
        detected, a key event is performed.

        :return: none
        """
        if self.prev_data and self.data:
            last_frames = self.prev_data[-numf:]

            # weighted difference values
            x_differences = []
            y_differences = []
            x_differences_relative = []
            y_differences_relative = []

            for d in last_frames:
                x_differences.append(abs(self.data.x_middle - d.x_middle))
                y_differences.append(abs(self.data.y_middle - d.y_middle))
                x_differences_relative.append(abs(self.data.x_middle_relative - d.x_middle_relative))
                y_differences_relative.append(abs(self.data.y_middle_relative - d.y_middle_relative))

            self.shake_detected = (
                sum(x_differences) > sum(y_differences) + abs(shake_diff_eps) and
                sum(x_differences_relative) > sum(y_differences_relative) + abs(shake_relative_diff_eps) and
                sum(x_differences) < 500
            )
            self.nod_detected = (
                sum(y_differences) > sum(x_differences) + abs(nod_diff_eps) and
                sum(y_differences_relative) > sum(x_differences_relative) + abs(nod_relative_diff_eps) and
                sum(y_differences) < 500
            ) 

        if self.nothing >= nothing_num and self.nod_detected:
            print("nod detected!", sum(y_differences), sum(y_differences_relative))
            self.nothing = 0
            return "Nod"

        if self.nothing >= nothing_num and self.shake_detected:
            print("shake detected!", sum(x_differences), sum(x_differences_relative))
            self.nothing = 0
            return "Shake"

        else:
            if self.prev_data and self.data:
                print("     neither!", sum(x_differences), sum(y_differences))
            self.nothing = self.nothing + 1
            return ""


class NodShakeHMM(object):
    """
    """

    alpha_ud = 4  # up/down threshold
    alpha_lr = 10  # left/right threshold
    beta = 3  # stationary threshold

    def __init__(self, maxlen=15):
        """
        Initialise HMM models for nod and shake and prepare an empty seq
        """
        self.data_list = deque(maxlen=3)

        models_dir = ''
        nod_file = "nod.json" # this is the model file name - you can create one yourself and set it in this variable
        shake_file = "shake.json"

        self.hmm_nod = MyHmm(os.path.join(models_dir, nod_file))
        self.hmm_shake = MyHmm(os.path.join(models_dir, shake_file))

        self.seq = deque(maxlen=maxlen)

    def add_data(self, new_data):
        self._determine_new_observable(new_data)

    def _determine_new_observable(self, new_data):
        x_differences = []
        y_differences = []
        x_differences_relative = []
        y_differences_relative = []

        for d in self.data_list:
            x_differences.append(abs(new_data.x_middle - d.x_middle))
            y_differences.append(abs(new_data.y_middle - d.y_middle))
            x_differences_relative.append(abs(new_data.x_middle_relative - d.x_middle_relative))
            y_differences_relative.append(abs(new_data.y_middle_relative - d.y_middle_relative))

        x_diff = np.mean(x_differences)
        y_diff = np.mean(y_differences)
        x_diff_relative = np.mean(x_differences_relative)
        y_diff_relative = np.mean(y_differences_relative)

        print("Y and X diff: ", y_diff, x_diff)

        if y_diff < self.beta and x_diff < self.beta:
            observable = 'stationary'
        elif y_diff > x_diff + self.alpha_ud:
            observable = 'up' if new_data.y_middle > self.data_list[-1].y_middle else 'down'
        elif x_diff > y_diff + self.alpha_lr:
            observable = 'left' if new_data.x_middle > self.data_list[-1].x_middle else 'right'
        else:
            observable = 'stationary'
        self._update_seq(observable)
        self.data_list.append(new_data)

    def _update_seq(self, observable):
        self.seq.append(observable)

    def determine_pose(self,):
        if len(self.seq) != self.seq.maxlen:  # Not enough data
            return ''
        p_nod = self.hmm_nod.forward(self.seq)
        p_shake = self.hmm_shake.forward(self.seq)

        print("     HMM Forward: ", p_shake, p_shake)
        print("     Sequence: ", self.seq)

        # seq = ['stationary', 'stationary', 'stationary', 'up', 'up', 'stationary', 'down', 'stationary', 'stationary', 'up', 'stationary', 'stationary', 'stationary', 'stationary', 'stationary']
        # print(self.hmm_nod.forward(seq))
        # raise

        if not p_nod and not p_shake:
            return ''
        elif p_nod > p_shake:
            return 'Nod'
        elif p_shake > p_nod:
            return 'Shake'