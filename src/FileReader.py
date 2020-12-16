import configparser
import re

config_path = r'../../config.yaml'
css_path = "../ui/style.css"


def set_style_sheet(widget):
    """
     Set style sheet for a widget.

     :param widget: the widget to be styled
     :return: none
     """
    with open(css_path, "r") as fh:
        widget.setStyleSheet(fh.read())
    fh.close()


class ConfigFileReader(object):
    """
    A class for retrieving values from config files.

    """

    def __init__(self):
        """
        The constructor that sets the initialization parameters for the file reader.

        """
        self.config_parser = configparser.RawConfigParser()
        self.config_parser.read(config_path)

    def read_int(self, section, var):
        """
        Read int value from config file.

        :param section: the section that contains the entry
        :param var: the variable that holds the value
        :return: the absolute value of the int value read, or `-1` if no value is found
        """
        value = self.get_value(section, var)
        return abs(int(value)) if value else -1

    def read_float(self, section, var):
        """
        Read float value from config file.

        :param section: the section that contains the entry
        :param var: the variable that holds the value
        :return: the absolute value of the float value read, or `-1` if no value is found
        """
        value = self.get_value(section, var)
        return abs(float(value)) if value else -1

    def read_bool(self, section, var):
        """
        Read bool value from config file.

        :param section: the section that contains the entry
        :param var: the variable that holds the value
        :return: the bool value read, or `-1` if no value is found
        """
        value = self.get_value(section, var)
        return abs(bool(int(value))) if value else -1

    def get_value(self, section, var):
        """
        Get a value from the config file.

        :param section: the section that contains the entry
        :param var: the variable that holds the value
        :return: the retrieved value
        """
        value = self.config_parser.get(section, var)
        value = re.search(r"[-+]?\d*\.\d+|\d+", value).group()
        return value
