import datetime
import math
import os
import random
from typing import Tuple, Callable

import tqdm

from pydgn.experiment.util import s2c
from pydgn.static import *


def return_class_and_args(config: dict, key: str, return_class_name: bool=False) -> Tuple[Callable[..., object], dict]:
    r"""
    Returns the class and arguments associated to a specific key in the configuration file.

    Args:
        config (dict): the configuration dictionary
        key (str): a string representing a particular class in the configuration dictionary
        return_class_name (bool): if ``True``, returns the class name as a string rather than the class object

    Returns:
        a tuple (class, dict of arguments), or (None, None) if the key is not present in the config dictionary
    """
    if key not in config or config[key] is None:
        return None, None
    elif isinstance(config[key], str):
        return s2c(config[key]), {}
    elif isinstance(config[key], dict):
        return s2c(config[key]['class_name']) if not return_class_name else config[key]['class_name'],\
               config[key]['args'] if 'args' in config[key] else {}
    else:
        raise NotImplementedError('Parameter has not been formatted properly')


def clear_screen():
    """
    Clears the CLI interface.
    """
    try:
        os.system('clear')
    except Exception as e:
        try:
            os.system('cls')
        except Exception:
            pass


class ProgressManager:
    r"""
    Class that is responsible for drawing progress bars.

    Args:
        outer_folds (int): number of external folds for model assessment
        inner_folds (int): number of internal folds for model selection
        no_configs (int): number of possible configurations in model selection
        final_runs (int): number of final runs per outer fold once the best model has been selected
        show (bool): whether to show the progress bar or not. Default is ``True``
    """
    # Possible vars of ``bar_format``:
    #       * ``l_bar, bar, r_bar``,
    #       * ``n, n_fmt, total, total_fmt``,
    #       * ``percentage, elapsed, elapsed_s``,
    #       * ``ncols, nrows, desc, unit``,
    #       * ``rate, rate_fmt, rate_noinv``,
    #       * ``rate_noinv_fmt, rate_inv, rate_inv_fmt``,
    #       * ``postfix, unit_divisor, remaining, remaining_s``
    def __init__(self, outer_folds, inner_folds, no_configs, final_runs, show=True):
        self.ncols = 100
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.no_configs = no_configs
        self.final_runs = final_runs
        self.pbars = []
        self.show = show

        if not self.show:
            return

        clear_screen()
        self.show_header()
        for i in range(self.outer_folds):
            for j in range(self.inner_folds):
                self.pbars.append(self._init_selection_pbar(i, j))

        for i in range(self.outer_folds):
            self.pbars.append(self._init_assessment_pbar(i))

        self.show_footer()

        self.times = [{} for _ in range(len(self.pbars))]

    def _init_selection_pbar(self, i, j):
        position = i * self.inner_folds + j
        pbar = tqdm.tqdm(total=self.no_configs, ncols=self.ncols, ascii=True,
                         position=position, unit="config",
                         bar_format=' {desc} {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}{postfix}')
        pbar.set_description(f'Out_{i + 1}/Inn_{j + 1}')
        mean = str(datetime.timedelta(seconds=0))
        pbar.set_postfix_str(f'(1 cfg every {mean})')
        return pbar

    def _init_assessment_pbar(self, i):
        position = self.outer_folds * self.inner_folds + i
        pbar = tqdm.tqdm(total=self.final_runs, ncols=self.ncols, ascii=True,
                         position=position, unit="config",
                         bar_format=' {desc} {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}{postfix}')
        pbar.set_description(f'Final run {i + 1}')
        mean = str(datetime.timedelta(seconds=0))
        pbar.set_postfix_str(f'(1 run every {mean})')
        return pbar

    def show_header(self):
        """
        Prints the header of the progress bar
        """
        '''
        \033[F --> move cursor to the beginning of the previous line
        \033[A --> move cursor up one line
        \033[<N>A --> move cursor up N lines
        '''
        print(
            f'\033[F\033[A{"*" * ((self.ncols - 21) // 2 + 1)} Experiment Progress {"*" * ((self.ncols - 21) // 2)}\n')

    def show_footer(self):
        pass  # need to work how how to print after tqdm

    def refresh(self):
        """
        Refreshes the progress bar
        """

        self.show_header()
        for i, pbar in enumerate(self.pbars):

            # When resuming, do not consider completed exp. (delta approx. < 1)
            completion_times = [delta for k, (delta, completed) in self.times[i].items() if completed and delta > 1]

            if len(completion_times) > 0:
                min_seconds = min(completion_times)
                max_seconds = max(completion_times)
                mean_seconds = sum(completion_times) / len(completion_times)
            else:
                min_seconds = 0
                max_seconds = 0
                mean_seconds = 0

            mean_time = str(datetime.timedelta(seconds=mean_seconds)).split('.')[0]
            min_time = str(datetime.timedelta(seconds=min_seconds)).split('.')[0]
            max_time = str(datetime.timedelta(seconds=max_seconds)).split('.')[0]
            pbar.set_postfix_str(f'min:{min_time}|avg:{mean_time}|max:{max_time}')

            pbar.refresh()
        self.show_footer()

    def update_state(self, msg: dict):
        """
        Updates the state of the progress bar (different from showing it on screen, see :func:`refresh`) once a message
        is received

        Args:
            msg (dict): message with updates to be parsed
        """
        if not self.show:
            return

        try:
            type = msg.get('type')

            if type == END_CONFIG:
                outer_fold = msg.get(OUTER_FOLD)
                inner_fold = msg.get(INNER_FOLD)
                config_id = msg.get(CONFIG_ID)
                position = outer_fold * self.inner_folds + inner_fold
                elapsed = msg.get(ELAPSED)
                configs_times = self.times[position]
                # Compute delta t for a specific config
                configs_times[config_id] = (elapsed, True)  # (time.time() - configs_times[config_id][0], True)
                # Update progress bar
                self.pbars[position].update()
                self.refresh()
            elif type == END_FINAL_RUN:
                outer_fold = msg.get(OUTER_FOLD)
                run_id = msg.get(RUN_ID)
                position = self.outer_folds * self.inner_folds + outer_fold
                elapsed = msg.get(ELAPSED)
                configs_times = self.times[position]
                # Compute delta t for a specific config
                configs_times[run_id] = (elapsed, True)  # (time.time() - configs_times[run_id][0], True)
                # Update progress bar
                self.pbars[position].update()
                self.refresh()
            else:
                raise Exception(f"Cannot parse type of message {type}, fix this.")

        except Exception as e:
            print(e)
            return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for pbar in self.pbars:
            pbar.close()


'''
Various options for random search model selection
'''
choice = lambda *args: random.choice(args)
uniform = lambda *args: random.uniform(*args)
normal = lambda *args: random.normalvariate(*args)
randint = lambda *args: random.randint(*args)

def loguniform(*args):
    """
    Performs a log-uniform random selection.

    Args:
        *args: a tuple of (log min, log max, [base]) to use. Base 10 is used if the third argument is not available.

    Returns:
        a randomly chosen value
    """
    log_min, log_max, *base = args
    base = base[0] if len(base) > 0 else 10

    log_min = math.log(log_min) / math.log(base)
    log_max = math.log(log_max) / math.log(base)

    return base ** (random.uniform(log_min, log_max))
