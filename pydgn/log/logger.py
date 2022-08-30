import os
from pathlib import Path


class Logger:
    r"""
    Class that implements simple logging facilities

    Args:
        filepath (str): the path to the file where to write
        mode (str):  can be 'w' (write) or 'a' (append)
        debug (bool): whether to print con screen (``True``)or to
            actually log on file (``False``)

    """

    def __init__(self, filepath, mode, debug):
        self.debug = debug
        self.filepath = Path(filepath)
        if not os.path.exists(self.filepath.parent):
            os.makedirs(self.filepath.parent)

        if mode not in ["w", "a"]:
            assert False, "Mode must be one of w, r or a"
        else:
            self.mode = mode

    def log(self, content):
        r"""
        Logs a string to the file associated to ``filepath``

        Args:
            content (str): content to log
        """
        if self.debug:
            print(content)
        try:
            with open(self.filepath, self.mode) as f:
                f.write(content + "\n")
        except Exception as e:
            print(e)
