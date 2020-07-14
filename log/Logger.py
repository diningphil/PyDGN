import os
from pathlib import Path

class Logger:
    """ Class that implements simple logging facilities (with multi-process support if needed) """

    def __init__(self, filepath, mode):
        """
        Initializes the logger
        :param filepath: the path to the file where to write
        :param mode: can be 'w' or 'a'
        """
        self.filepath = Path(filepath)
        if not os.path.exists(self.filepath.parent):
            os.makedirs(self.filepath.parent)

        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode

    def log(self, str):
        """ Logs a string to the file associated to self.filepath """
        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)
