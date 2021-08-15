import bisect
import datetime
import time

from pydgn.training.event.handler import EventHandler


# Decorator class. Ideas taken from "Fluent Python" book
class Profiler:
    def __init__(self, threshold):
        self.threshold = threshold
        self.callback_elapsed = {}
        self.callback_calls = {}

    def __call__(self, cls):
        self.callback_elapsed[cls.__class__.__name__] = {}
        self.callback_calls[cls.__class__.__name__] = {}

        def clock(func):
            def clocked(*args, **kwargs):
                class_name = cls.__class__.__name__
                callback_name = func.__name__
                t0 = time.time()
                # print(f'Callback {cls.__class__.__name__}: calling {func.__name__} with args {args} and kwargs {kwargs}')
                result = func(cls, *args, **kwargs)
                elapsed = time.time() - t0

                if callback_name in self.callback_elapsed[class_name]:
                    self.callback_elapsed[class_name][callback_name] += elapsed
                    self.callback_calls[class_name][callback_name] += 1
                else:
                    self.callback_elapsed[class_name][callback_name] = elapsed
                    self.callback_calls[class_name][callback_name] = 1

                return result

            return clocked

        # Relies on closures
        class ClockedCallback(EventHandler):

            @clock
            def on_fetch_data(self, state):
                return cls.on_fetch_data(state)

            @clock
            def on_fit_start(self, state):
                return cls.on_fit_start(state)

            @clock
            def on_fit_end(self, state):
                return cls.on_fit_end(state)

            @clock
            def on_epoch_start(self, state):
                return cls.on_epoch_start(state)

            @clock
            def on_epoch_end(self, state):
                return cls.on_epoch_end(state)

            @clock
            def on_training_epoch_start(self, state):
                return cls.on_training_epoch_start(state)

            @clock
            def on_training_epoch_end(self, state):
                return cls.on_training_epoch_end(state)

            @clock
            def on_eval_epoch_start(self, state):
                return cls.on_eval_epoch_start(state)

            @clock
            def on_eval_epoch_end(self, state):
                return cls.on_eval_epoch_end(state)

            @clock
            def on_training_batch_start(self, state):
                return cls.on_training_batch_start(state)

            @clock
            def on_training_batch_end(self, state):
                return cls.on_training_batch_end(state)

            @clock
            def on_eval_batch_start(self, state):
                return cls.on_eval_batch_start(state)

            @clock
            def on_eval_batch_end(self, state):
                return cls.on_eval_batch_end(state)

            @clock
            def on_backward(self, state):
                return cls.on_backward(state)

            @clock
            def on_forward(self, state):
                return cls.on_forward(state)

            @clock
            def on_compute_metrics(self, state):
                return cls.on_compute_metrics(state)

        return ClockedCallback

    def report(self):
        total_time_experiment = 0.
        profile_str = f'{"*" * 25} Profiler {"*" * 25} \n \n'
        profile_str += f'Threshold: {self.threshold} \n \n'

        for class_name, v in self.callback_elapsed.items():

            sorted_avg_elapsed = []

            for callback_name, v1 in v.items():
                n = self.callback_calls[class_name][callback_name]

                # Depending on when a KeyboardInterrupt is triggered
                if n == 0:
                    continue

                avg_elapsed = v1 / n

                if avg_elapsed > self.threshold:
                    bisect.insort(sorted_avg_elapsed, (avg_elapsed, v1, callback_name))

            # Release resources
            v.clear()

            profile_str += f'{class_name} \n \n'

            for (avg_elapsed, total_elapsed, callback_name) in reversed(sorted_avg_elapsed):
                total_time_experiment += total_elapsed
                profile_str += f'\t {callback_name} --> Avg: {avg_elapsed} s, Total: {str(datetime.timedelta(seconds=total_elapsed))} \n'
            profile_str += '\n'

        profile_str += f'Total time of the experiment: {str(datetime.timedelta(seconds=total_time_experiment))} \n \n'
        profile_str += f'{"*" * 60}'
        return profile_str
