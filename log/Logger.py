class Logger:
    """ Class that implements simple logging facilities (with multi-process support if needed) """

    def __init__(self, filepath, mode, lock=None):
        """
        Initializes the logger
        :param filepath: the path to the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str):
        """ Logs a string to the file associated to self.filepath """
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()


