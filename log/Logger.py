import os
from pathlib import Path


class Logger:
    """ Class that implements simple logging facilities (with multi-process support if needed) """

    def __init__(self, filepath, mode, debug):
        """
        Initializes the logger
        :param filepath: the path to the file where to write
        :param mode: can be 'w' or 'a'
        """
        self.debug = debug
        self.filepath = Path(filepath)
        if not os.path.exists(self.filepath.parent):
            os.makedirs(self.filepath.parent)

        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode

    def log(self, str):
        """ Logs a string to the file associated to self.filepath """
        if self.debug:
            print(str)

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)
