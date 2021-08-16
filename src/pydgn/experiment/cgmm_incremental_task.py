import os
import shutil

import torch
from torch.utils.data.sampler import SequentialSampler
from torch_geometric.data import Data

from pydgn.experiment.experiment import Experiment
from pydgn.experiment.util import s2c


class CGMMTask(Experiment):

    def __init__(self, model_configuration, exp_path, exp_seed):
        super(CGMMTask, self).__init__(model_configuration, exp_path, exp_seed)
        self.root_exp_path = exp_path  # to distinguish from layers' exp_path
        self.output_folder = os.path.join(exp_path, 'outputs')
        self._concat_axis = self.model_config.layer_config['concatenate_on_axis']

    def _create_extra_dataset(self, prev_outputs_to_consider, mode, depth, only_g=False):
        # Load previous outputs if any according to prev. layers to consider (ALL TENSORS)
        v_outs, e_outs, g_outs, vo_outs, eo_outs, go_outs = self._load_outputs(mode, prev_outputs_to_consider)

        data_list = []

        no_graphs = max(len(v_outs) if v_outs is not None else 0,
                        len(e_outs) if e_outs is not None else 0,
                        len(g_outs) if g_outs is not None else 0,
                        len(vo_outs) if vo_outs is not None else 0,
                        len(eo_outs) if eo_outs is not None else 0,
                        len(go_outs) if go_outs is not None else 0)

        for index in range(no_graphs):
            g = g_outs[index] if g_outs is not None else None
            go = go_outs[index] if go_outs is not None else None
            if not only_g:
                v = v_outs[index] if v_outs is not None else None
                e = e_outs[index] if e_outs is not None else None
                vo = vo_outs[index] if vo_outs is not None else None
                eo = eo_outs[index] if eo_outs is not None else None
                data_list.append(Data(v_outs=v, e_outs=e, g_outs=g,
                                      vo_outs=vo, eo_outs=eo, go_outs=go))
            else:
                data_list.append(Data(g_outs=g, go_outs=go))

        return data_list

    @staticmethod
    def _reorder_shuffled_objects(v_out, e_out, g_out, vo_out, eo_out, go_out, data_loader):
        if type(data_loader.sampler) == SequentialSampler:  # No permutation
            return v_out, e_out, g_out, vo_out, eo_out, go_out

        idxs = data_loader.sampler.permutation  # permutation of the last data_loader iteration

        def reorder(obj, perm):
            assert len(obj) == len(perm) and len(obj) > 0
            return [y for (x, y) in sorted(zip(perm, obj))]

        if v_out is not None:
            # print(len(v_out))
            v_out = reorder(v_out, idxs)

        if e_out is not None:
            raise NotImplementedError('This feature has not been implemented yet!')
            # e_out = reorder(e_out, idxs)

        if g_out is not None:
            g_out = reorder(g_out, idxs)

        if vo_out is not None:
            # print(len(o_out))
            vo_out = reorder(vo_out, idxs)

        if eo_out is not None:
            # print(len(o_out))
            eo_out = reorder(eo_out, idxs)

        if go_out is not None:
            # print(len(o_out))
            go_out = reorder(go_out, idxs)

        return v_out, e_out, g_out, vo_out, eo_out, go_out

    def _load_outputs(self, mode, prev_outputs_to_consider):

        outs_dict = {
            'vertex_outputs': None,
            'edge_outputs': None,
            'graph_outputs': None,
            'vertex_other_outputs': None,
            'edge_other_outputs': None,
            'graph_other_outputs': None,
        }

        # The elements of prev_outputs_to_consider will be concatenated in
        # reverse, i.e., if prev_outputs_to_consider = 1,2,3...L
        # the contribution of layer L will appear in position 0 across
        # self._concat_axis, then L-1 in position 1 and so on
        # this is because a hyper-parameter l_prec=1 means "previous layer"
        # and prev_outputs_to_consider will be = L,L-1,...1
        # so we want to reorder layers from 1 to L
        for prev in prev_outputs_to_consider:
            for path, o_key in [(os.path.join(self.output_folder, mode, f'vertex_output_{prev}.pt'), 'vertex_outputs'),
                                (os.path.join(self.output_folder, mode, f'edge_output_{prev}.pt'), 'edge_outputs'),
                                (os.path.join(self.output_folder, mode, f'graph_output_{prev}.pt'), 'graph_outputs'),
                                (os.path.join(self.output_folder, mode, f'vertex_other_output_{prev}.pt'),
                                 'vertex_other_outputs'),
                                (os.path.join(self.output_folder, mode, f'edge_other_output_{prev}.pt'),
                                 'edge_other_outputs'),
                                (os.path.join(self.output_folder, mode, f'graph_other_output_{prev}.pt'),
                                 'graph_other_outputs'), ]:
                if os.path.exists(path):
                    out = torch.load(path)
                    outs = outs_dict[o_key]

                    if outs is None:
                        # print('None!')
                        outs = [None] * len(out)

                    # print(path, o_key, len(out))
                    # print(out[0].shape)

                    for graph_id in range(len(out)):  # iterate over graphs

                        outs[graph_id] = out[graph_id] if outs[graph_id] is None \
                            else torch.cat((out[graph_id], outs[graph_id]), self._concat_axis)

                    outs_dict[o_key] = outs

        return outs_dict['vertex_outputs'], outs_dict['edge_outputs'], \
               outs_dict['graph_outputs'], outs_dict['vertex_other_outputs'], \
               outs_dict['edge_other_outputs'], outs_dict['graph_other_outputs']

    def _store_outputs(self, mode, depth, v_tensor_list, e_tensor_list=None, g_tensor_list=None,
                       vo_tensor_list=None, eo_tensor_list=None, go_tensor_list=None):

        if not os.path.exists(os.path.join(self.output_folder, mode)):
            os.makedirs(os.path.join(self.output_folder, mode))

        if v_tensor_list is not None:
            vertex_filepath = os.path.join(self.output_folder, mode, f'vertex_output_{depth}.pt')
            torch.save([torch.unsqueeze(v_tensor, self._concat_axis) for v_tensor in v_tensor_list], vertex_filepath)
        if e_tensor_list is not None:
            edge_filepath = os.path.join(self.output_folder, mode, f'edge_output_{depth}.pt')
            torch.save([torch.unsqueeze(e_tensor, self._concat_axis) for e_tensor in e_tensor_list], edge_filepath)
        if g_tensor_list is not None:
            graph_filepath = os.path.join(self.output_folder, mode, f'graph_output_{depth}.pt')
            torch.save([torch.unsqueeze(g_tensor, self._concat_axis) for g_tensor in g_tensor_list], graph_filepath)
        if vo_tensor_list is not None:
            vertex_other_filepath = os.path.join(self.output_folder, mode, f'vertex_other_output_{depth}.pt')
            torch.save([torch.unsqueeze(vo_tensor, self._concat_axis) for vo_tensor in vo_tensor_list],
                       vertex_other_filepath)
        if eo_tensor_list is not None:
            edge_other_filepath = os.path.join(self.output_folder, mode, f'edge_other_output_{depth}.pt')
            torch.save([torch.unsqueeze(eo_tensor, self._concat_axis) for eo_tensor in eo_tensor_list],
                       edge_other_filepath)
        if go_tensor_list is not None:
            graph_other_filepath = os.path.join(self.output_folder, mode, f'graph_other_output_{depth}.pt')
            torch.save([torch.unsqueeze(go_tensor, self._concat_axis) for go_tensor in go_tensor_list],
                       graph_other_filepath)

    def run_valid(self, dataset_getter, logger):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        batch_size = self.model_config.layer_config['batch_size']
        arbitrary_logic_batch_size = self.model_config.layer_config['arbitrary_function_config']['batch_size']
        shuffle = self.model_config.layer_config['shuffle'] \
            if 'shuffle' in self.model_config.layer_config else True
        arbitrary_logic_shuffle = self.model_config.layer_config['arbitrary_function_config']['shuffle'] \
            if 'shuffle' in self.model_config.layer_config['arbitrary_function_config'] else True

        # Instantiate the Dataset
        dim_node_features = dataset_getter.get_dim_node_features()
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()

        layers = []
        l_prec = self.model_config.layer_config['previous_layers_to_use'].split(',')
        concatenate_axis = self.model_config.layer_config['concatenate_on_axis']
        max_layers = self.model_config.layer_config['max_layers']
        assert concatenate_axis > 0, 'You cannot concat on the first axis for design reasons.'

        dict_per_layer = []

        stop = False
        depth = 1
        while not stop and depth <= max_layers:

            # Change exp path to allow Stop & Resume
            self.exp_path = os.path.join(self.root_exp_path, f'layer_{depth}')

            # load output will concatenate in reverse order
            prev_outputs_to_consider = [(depth - int(x)) for x in l_prec if (depth - int(x)) > 0]

            train_out = self._create_extra_dataset(prev_outputs_to_consider, mode='train', depth=depth)
            val_out = self._create_extra_dataset(prev_outputs_to_consider, mode='validation', depth=depth)
            train_loader = dataset_getter.get_inner_train(batch_size=batch_size, shuffle=shuffle, extra=train_out)
            val_loader = dataset_getter.get_inner_val(batch_size=batch_size, shuffle=shuffle, extra=val_out)

            # Instantiate the Model
            new_layer = self.create_incremental_model(dim_node_features, dim_edge_features, dim_target, depth,
                                                      prev_outputs_to_consider)

            # Instantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
            incremental_training_wrapper = self.create_incremental_wrapper(new_layer)

            train_loss, train_score, train_out, \
            val_loss, val_score, val_out, \
            _, _, _ = incremental_training_wrapper.train(train_loader=train_loader,
                                                         validation_loader=val_loader,
                                                         test_loader=None,
                                                         max_epochs=self.model_config.layer_config['epochs'],
                                                         logger=logger)

            for loader, out, mode in [(train_loader, train_out, 'train'), (val_loader, val_out, 'validation')]:
                v_out, e_out, g_out, vo_out, eo_out, go_out = out

                # Reorder outputs, which are produced in shuffled order, to the original arrangement of the dataset.
                v_out, e_out, g_out, vo_out, eo_out, go_out = self._reorder_shuffled_objects(v_out, e_out, g_out,
                                                                                             vo_out, eo_out, go_out,
                                                                                             loader)

                # Store outputs
                self._store_outputs(mode, depth, v_out, e_out, g_out, vo_out, eo_out, go_out)

            # Consider all previous layers now, i.e. gather all the embeddings
            prev_outputs_to_consider = [l for l in range(1, depth + 1)]
            prev_outputs_to_consider.reverse()  # load output will concatenate in reverse order

            train_out = self._create_extra_dataset(prev_outputs_to_consider, mode='train', depth=depth)
            val_out = self._create_extra_dataset(prev_outputs_to_consider, mode='validation', depth=depth)
            train_loader = dataset_getter.get_inner_train(batch_size=arbitrary_logic_batch_size,
                                                          shuffle=arbitrary_logic_shuffle, extra=train_out)
            val_loader = dataset_getter.get_inner_val(batch_size=arbitrary_logic_batch_size,
                                                      shuffle=arbitrary_logic_shuffle, extra=val_out)

            # Change exp path to allow Stop & Resume
            self.exp_path = os.path.join(self.root_exp_path, f'layer_{depth}_stopping_criterion')

            # Stopping criterion based on training of the model
            stop = new_layer.stopping_criterion(depth, max_layers, train_loss, train_score, val_loss, val_score,
                                                dict_per_layer, self.model_config.layer_config, logger=logger)

            # Change exp path to allow Stop & Resume
            self.exp_path = os.path.join(self.root_exp_path, f'layer_{depth}_arbitrary_config')

            if stop:

                if 'CA' in self.model_config.layer_config:
                    # ECGMM
                    dim_features = new_layer.dim_node_features, new_layer.C * new_layer.depth + new_layer.CA * new_layer.depth if not new_layer.unibigram else (
                                                                                                                                                                       new_layer.C + new_layer.CA + new_layer.C * new_layer.C) * new_layer.depth
                else:
                    # CGMM
                    dim_features = new_layer.dim_node_features, new_layer.C * new_layer.depth if not new_layer.unibigram else (
                                                                                                                                      new_layer.C + new_layer.C * new_layer.C) * new_layer.depth

                config = self.model_config.layer_config['arbitrary_function_config']
                device = config['device']

                predictor_class = s2c(config['predictor'])
                model = predictor_class(dim_node_features=dim_features,
                                        dim_edge_features=0,
                                        dim_target=dim_target,
                                        config=config)

                predictor_wrapper = self._create_wrapper(config, model, device, log_every=self.model_config.log_every)

                train_loss, train_score, _, \
                val_loss, val_score, _, \
                _, _, _ = predictor_wrapper.train(train_loader=train_loader,
                                                  validation_loader=val_loader,
                                                  test_loader=None,
                                                  max_epochs=config['epochs'],
                                                  logger=logger)

                d = {'train_score': train_score, 'validation_score': val_score}
            else:
                d = {}

            # Append layer
            layers.append(new_layer)
            dict_per_layer.append(d)

            # Give priority to arbitrary function
            stop = d['stop'] if 'stop' in d else stop

            depth += 1

        # CLEAR OUTPUTS TO SAVE SPACE
        for mode in ['train', 'validation']:
            shutil.rmtree(os.path.join(self.output_folder, mode), ignore_errors=True)

        return dict_per_layer[-1]['train_score'], dict_per_layer[-1]['validation_score']

    def run_test(self, dataset_getter, logger):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR ANY REASON
        :return: (training accuracy, test accuracy)
        """
        batch_size = self.model_config.layer_config['batch_size']
        arbitrary_logic_batch_size = self.model_config.layer_config['arbitrary_function_config']['batch_size']
        shuffle = self.model_config.layer_config['shuffle'] \
            if 'shuffle' in self.model_config.layer_config else True
        arbitrary_logic_shuffle = self.model_config.layer_config['arbitrary_function_config']['shuffle'] \
            if 'shuffle' in self.model_config.layer_config['arbitrary_function_config'] else True

        # Instantiate the Dataset
        dim_node_features = dataset_getter.get_dim_node_features()
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()

        layers = []
        l_prec = self.model_config.layer_config['previous_layers_to_use'].split(',')
        concatenate_axis = self.model_config.layer_config['concatenate_on_axis']
        max_layers = self.model_config.layer_config['max_layers']
        assert concatenate_axis > 0, 'You cannot concat on the first axis for design reasons.'

        dict_per_layer = []

        stop = False
        depth = 1
        while not stop and depth <= max_layers:

            # Change exp path to allow Stop & Resume
            self.exp_path = os.path.join(self.root_exp_path, f'layer_{depth}')

            prev_outputs_to_consider = [(depth - int(x)) for x in l_prec if (depth - int(x)) > 0]

            train_out = self._create_extra_dataset(prev_outputs_to_consider, mode='train', depth=depth)
            val_out = self._create_extra_dataset(prev_outputs_to_consider, mode='validation', depth=depth)
            test_out = self._create_extra_dataset(prev_outputs_to_consider, mode='test', depth=depth)

            train_loader = dataset_getter.get_outer_train(batch_size=batch_size, shuffle=shuffle, extra=train_out)
            val_loader = dataset_getter.get_outer_val(batch_size=batch_size, shuffle=shuffle, extra=val_out)
            test_loader = dataset_getter.get_outer_test(batch_size=batch_size, shuffle=shuffle, extra=test_out)

            # Instantiate the Model
            new_layer = self.create_incremental_model(dim_node_features, dim_edge_features, dim_target,
                                                      depth, prev_outputs_to_consider)

            # Instantiate the wrapper (it handles the training loop and inference phase by abstracting the specifics)
            incremental_training_wrapper = self.create_incremental_wrapper(new_layer)

            train_loss, train_score, train_out, \
            val_loss, val_score, val_out, \
            test_loss, test_score, test_out = incremental_training_wrapper.train(train_loader=train_loader,
                                                                                 validation_loader=val_loader,
                                                                                 test_loader=test_loader,
                                                                                 max_epochs=
                                                                                 self.model_config.layer_config[
                                                                                     'epochs'],
                                                                                 logger=logger)

            for loader, out, mode in [(train_loader, train_out, 'train'),
                                      (val_loader, val_out, 'validation'),
                                      (test_loader, test_out, 'test')]:
                v_out, e_out, g_out, vo_out, eo_out, go_out = out

                # Reorder outputs, which are produced in shuffled order, to the original arrangement of the dataset.
                v_out, e_out, g_out, vo_out, eo_out, go_out = self._reorder_shuffled_objects(v_out, e_out, g_out,
                                                                                             vo_out, eo_out, go_out,
                                                                                             loader)

                # Store outputs
                self._store_outputs(mode, depth, v_out, e_out, g_out, vo_out, eo_out, go_out)

            # Consider all previous layers now, i.e. gather all the embeddings
            prev_outputs_to_consider = [l for l in range(1, depth + 1)]

            train_out = self._create_extra_dataset(prev_outputs_to_consider, mode='train', depth=depth,
                                                   only_g_outs=True)
            val_out = self._create_extra_dataset(prev_outputs_to_consider, mode='validation', depth=depth,
                                                 only_g_outs=True)
            test_out = self._create_extra_dataset(prev_outputs_to_consider, mode='test', depth=depth, only_g_outs=True)
            train_loader = dataset_getter.get_outer_train(batch_size=arbitrary_logic_batch_size,
                                                          shuffle=arbitrary_logic_shuffle, extra=train_out)
            val_loader = dataset_getter.get_outer_val(batch_size=arbitrary_logic_batch_size,
                                                      shuffle=arbitrary_logic_shuffle, extra=val_out)
            test_loader = dataset_getter.get_outer_test(batch_size=arbitrary_logic_batch_size,
                                                        shuffle=arbitrary_logic_shuffle, extra=test_out)

            # Change exp path to allow Stop & Resume
            self.exp_path = os.path.join(self.root_exp_path, f'layer_{depth}_stopping_criterion')

            # Stopping criterion based on training of the model
            stop = new_layer.stopping_criterion(depth, max_layers, train_loss, train_score, val_loss, val_score,
                                                dict_per_layer, self.model_config.layer_config,
                                                logger=logger)

            # Change exp path to allow Stop & Resume
            self.exp_path = os.path.join(self.root_exp_path, f'layer_{depth}_arbitrary_config')

            if stop:

                if 'CA' in self.model_config.layer_config:
                    # ECGMM
                    dim_features = new_layer.dim_node_features, new_layer.C * new_layer.depth + new_layer.CA * new_layer.depth if not new_layer.unibigram else (
                                                                                                                                                                       new_layer.C + new_layer.CA + new_layer.C * new_layer.C) * new_layer.depth
                else:
                    # CGMM
                    dim_features = new_layer.dim_node_features, new_layer.C * new_layer.depth if not new_layer.unibigram else (
                                                                                                                                      new_layer.C + new_layer.C * new_layer.C) * new_layer.depth

                config = self.model_config.layer_config['arbitrary_function_config']
                device = config['device']

                predictor_class = s2c(config['predictor'])
                model = predictor_class(dim_node_features=dim_features,
                                        dim_edge_features=0,
                                        dim_target=dim_target,
                                        config=config)

                predictor_wrapper = self._create_wrapper(config, model, device, log_every=self.model_config.log_every)

                train_loss, train_score, _, \
                val_loss, val_score, _, \
                test_loss, test_score, _ = predictor_wrapper.train(train_loader=train_loader,
                                                                   validation_loader=val_loader,
                                                                   test_loader=test_loader,
                                                                   max_epochs=config['epochs'],
                                                                   logger=logger)

                d = {'train_score': train_score, 'validation_score': val_score, 'test_score': test_score}
            else:
                d = {}

            # Append layer
            layers.append(new_layer)
            dict_per_layer.append(d)

            # Give priority to arbitrary function
            stop = d['stop'] if 'stop' in d else stop

            depth += 1

        # CLEAR OUTPUTS TO SAVE SPACE
        for mode in ['train', 'validation', 'test']:
            shutil.rmtree(os.path.join(self.output_folder, mode), ignore_errors=True)

        # Use last training and test scores
        return dict_per_layer[-1]['train_score'], dict_per_layer[-1]['test_score']
