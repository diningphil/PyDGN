import os
import json
import operator
import traceback
import torch.multiprocessing as mp
import concurrent.futures
from log.Logger import Logger


class HoldOutSelection:
    """
    Class implementing a Hold-Out technique to do Model Selection
    """

    def __init__(self, max_processes, higher_is_better=True):
        """
        Initialized a Hold-out procedure for Model Selection (selection of best hyper-parameters configurations)
        :param max_processes: the number of parallel processes to run all possible hyper-parameters configurations.
        """
        self.inner_folds = 1
        self.max_processes = max_processes
        self.higher_is_better = higher_is_better
        if higher_is_better:
            self.operator = operator.gt
        else:
            self.operator = operator.lt

        # Create the experiments folder straight away
        self._CONFIG_BASE = 'config_'
        self._CONFIG_FILENAME = 'config_results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'

    def process_results(self, folder, no_configurations):
        """
        Chooses the best hyper-parameters configuration using the HIGHEST validation mean score
        :param folder: a folder which holds all configurations results after K folds
        :param no_configurations: number of possible configurations
        """

        best_vl = -float('inf') if self.higher_is_better else float('inf')

        for i in range(1, no_configurations+1):
            try:
                config_filename = os.path.join(folder, self._CONFIG_BASE + str(i),
                                               self._CONFIG_FILENAME)

                with open(config_filename, 'r') as fp:
                    config_dict = json.load(fp)

                vl = config_dict['VL_score']

                if self.operator(vl, best_vl):
                    best_i = i
                    best_vl = vl
                    best_config = config_dict

            except Exception as e:
                print(e)

        return best_config, best_i

    def model_selection(self, outer_fold_id, dataset_getter, experiment_class, exp_path, model_configs, debug, other, snd_queue):
        """
        Performs model selection by launching each configuration in parallel, unless debug is True.
        :param outer_fold_id: id of the outer assessment fold
        :param dataset_getter: an object that handles the creation of dataloaders. See datasets.provider.DataProvider
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param exp_path: The folder in which to store all results
        :param model_configs: an object storing all possible model configurations, e.g. config.base.Grid
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        :param snd_queue: a queue to inform the main process about the progress
        :return: a dictionary holding the best configuration
        """
        HOLDOUT_MS_FOLDER = os.path.join(exp_path, 'HOLDOUT_MS')

        if not os.path.exists(HOLDOUT_MS_FOLDER):
            os.makedirs(HOLDOUT_MS_FOLDER)

        config_id = 0

        # See https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-cuda-note
        mp_context = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_processes, mp_context=mp_context) as pool:

            for config in model_configs:  # generate_grid(model_configs):

                # Create a separate folder for each experiment
                exp_config_name = os.path.join(HOLDOUT_MS_FOLDER, self._CONFIG_BASE + str(config_id + 1))
                if not os.path.exists(exp_config_name):
                    os.makedirs(exp_config_name)

                json_config = os.path.join(exp_config_name, self._CONFIG_FILENAME)
                if not os.path.exists(json_config):
                    if not debug:
                        pool.submit(self._model_selection_helper, outer_fold_id, dataset_getter, experiment_class, config,
                                    config_id, exp_config_name, other, snd_queue)
                    else:  # DEBUG
                        self._model_selection_helper(outer_fold_id, dataset_getter, experiment_class, config,
                                    config_id, exp_config_name, other, snd_queue)
                else:
                    # Do not recompute experiments for this fold.
                    #print(f"Config {json_config} already present! Shutting down to prevent loss of previous experiments")
                    for msg_type in ["START_CONFIG", "END_CONFIG"]:
                        msg = dict(type=msg_type, outer_fold=outer_fold_id, config_id=config_id, inner_fold=0)
                        snd_queue.put(msg)

                config_id += 1

        best_config, best_config_id = self.process_results(HOLDOUT_MS_FOLDER, config_id)

        with open(os.path.join(HOLDOUT_MS_FOLDER, self.WINNER_CONFIG_FILENAME), 'w') as fp:
            json.dump(dict(best_config=best_config, best_config_id=best_config_id), fp)

        return best_config

    def _model_selection_helper(self, outer_fold_id, dataset_getter, experiment_class, config, config_id, exp_config_name, other, snd_queue):
        """
        Helper method that runs model selection for a particular configuration. Validation scores are averaged to
        compute the validation score of a single hyper-parameter configuration.
        :param outer_fold_id: id of the outer assessment fold
        :param dataset_getter: an object that handles the creation of dataloaders. See datasets.provider.DataProvider
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param config: Dictionary holding a specific hyper-parameter configuration
        :param config_id: Configuration ID
        :param exp_config_name: The folder in which to store all results for a specific configuration
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        :param snd_queue: a queue to inform the main process about the progress
        """
        dataset_getter.set_inner_k(0)  # need to stay 0

        # Create the experiment object which will be responsible for running a specific experiment
        experiment = experiment_class(config, exp_config_name)

        # Set up a log file for this experiment (run in a separate process)
        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')
        logger.log('Configuration: \n' + str(experiment.model_config))

        config_filename = os.path.join(experiment.exp_path, self._CONFIG_FILENAME)

        # ------------- PREPARE DICTIONARY TO STORE RESULTS -------------- #

        selection_dict = {
            'config': experiment.model_config.config_dict,
            'TR_score': 0.,
            'VL_score': 0.,
        }

        # Inform the main process about experiment completion
        msg = dict(type="START_CONFIG", outer_fold=outer_fold_id, config_id=config_id, inner_fold=0)
        snd_queue.put(msg)

        try:
            training_score, validation_score = experiment.run_valid(dataset_getter, logger, other)

            for k in training_score.keys():
                selection_dict[f'TR_{k}'] = float(training_score[k])
                selection_dict[f'VL_{k}'] = float(validation_score[k])

            selection_dict['TR_score'] = float(training_score['main_score'])
            selection_dict['VL_score'] = float(validation_score['main_score'])
        except Exception:
            logger.log(traceback.format_exc())
            print(traceback.format_exc())

        # Inform the main process about experiment completion
        msg = dict(type="END_CONFIG", outer_fold=outer_fold_id, config_id=config_id, inner_fold=0)
        snd_queue.put(msg)

        logger.log('TR Score: ' + str(training_score) + ' VL Score: ' + str(validation_score))

        with open(config_filename, 'w') as fp:
            json.dump(selection_dict, fp)
