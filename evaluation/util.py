import datetime
import math
import os
import random

import tqdm

from pydgn.static import *


def set_gpus(num_gpus):
    try:
        import gpustat
    except ImportError as e:
        print("gpustat module is not installed. No GPU allocated.")

    try:
        selected = []

        stats = gpustat.GPUStatCollection.new_query()

        for i in range(num_gpus):

            ids_mem = [res for res in map(lambda gpu: (int(gpu.entry['index']),
                                                       float(gpu.entry['memory.used']) / \
                                                       float(gpu.entry['memory.total'])),
                                          stats) if str(res[0]) not in selected]

            if len(ids_mem) == 0:
                # No more gpus available
                break

            best = min(ids_mem, key=lambda x: x[1])
            bestGPU, bestMem = best[0], best[1]
            # print(f"{i}-th best is {bestGPU} with mem {bestMem}")
            selected.append(str(bestGPU))

        print("Setting GPUs to: {}".format(",".join(selected)))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(selected)
    except BaseException as e:
        print("GPU not available: " + str(e))


def clear_screen():
    try:
        os.system('clear')
    except Exception as e:
        try:
            os.system('cls')
        except Exception:
            pass


class ProgressManager:
    '''
    Possible vars of bar_format:
          l_bar, bar, r_bar,
          n, n_fmt, total, total_fmt,
          percentage, elapsed, elapsed_s,
          ncols, nrows, desc, unit,
          rate, rate_fmt, rate_noinv,
          rate_noinv_fmt, rate_inv, rate_inv_fmt,
          postfix, unit_divisor, remaining, remaining_s
    '''

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

    def show_header(self):
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

    def update_state(self, msg):
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


choice = lambda *args: random.choice(args)
uniform = lambda *args: random.uniform(*args)
normal = lambda *args: random.normalvariate(*args)
randint = lambda *args: random.randint(*args)


def loguniform(*args):
    log_min, log_max, *base = args
    base = base[0] if len(base) > 0 else 10

    log_min = math.log(log_min) / math.log(base)
    log_max = math.log(log_max) / math.log(base)

    return base ** (random.uniform(log_min, log_max))
