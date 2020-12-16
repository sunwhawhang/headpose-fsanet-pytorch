"""
original copyright: github.com/saraheee/LerntiaControl
"""
from FileReader import ConfigFileReader

numf = 20  # frame number for nod and shake detection
nod_diff_eps = 150  # shift in weighted difference values of head nods
shake_diff_eps = 200  # shift in weighted difference values of head shakes
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

            for d in last_frames:
                x_differences.append(abs(self.data.x_middle - d.x_middle))
                y_differences.append(abs(self.data.y_middle - d.y_middle))

            self.shake_detected = sum(x_differences) > sum(y_differences) + abs(shake_diff_eps)
            self.nod_detected = sum(y_differences) > sum(x_differences) + abs(nod_diff_eps)

        if self.nothing >= nothing_num and self.nod_detected:
            print("nod detected!")
            return "Nod"

        if self.nothing >= nothing_num and self.shake_detected:
            print("shake detected!")
            return "Shake"

        else:
            return ""
