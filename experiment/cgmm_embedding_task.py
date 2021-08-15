import os
import shutil

import torch

from pydgn.experiment.cgmm_incremental_task import CGMMTask


# This works with graph classification only
class EmbeddingCGMMTask(CGMMTask):

    def run_valid(self, dataset_getter, logger):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        batch_size = self.model_config.layer_config['batch_size']
        shuffle = self.model_config.layer_config['shuffle'] \
            if 'shuffle' in self.model_config.layer_config else True

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
            if os.path.exists(os.path.join(self.root_exp_path, f'layer_{depth + 1}')):
                # print("skip layer", depth)
                depth += 1
                continue

            # load output will concatenate in reverse order
            prev_outputs_to_consider = [(depth - int(x)) for x in l_prec if (depth - int(x)) > 0]

            train_out = self._create_extra_dataset(prev_outputs_to_consider, mode='train', depth=depth)
            val_out = self._create_extra_dataset(prev_outputs_to_consider, mode='validation', depth=depth)
            train_loader = dataset_getter.get_inner_train(batch_size=batch_size, shuffle=False, extra=train_out)
            val_loader = dataset_getter.get_inner_val(batch_size=batch_size, shuffle=False, extra=val_out)

            # ==== # WARNING: WE ARE JUSTPRECOMPUTING OUTER_TEST EMBEDDINGS FOR SUBSEQUENT CLASSIFIERS
            # WE ARE NOT TRAINING ON TEST (EVEN THOUGH UNSUPERVISED)
            # ==== #

            test_out = self._create_extra_dataset(prev_outputs_to_consider, mode='test', depth=depth)
            test_loader = dataset_getter.get_outer_test(batch_size=batch_size, shuffle=False, extra=test_out)

            # ==== #

            # Instantiate the Model
            new_layer = self.create_incremental_model(dim_node_features, dim_edge_features, dim_target, depth,
                                                      prev_outputs_to_consider)

            # Instantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
            incremental_training_wrapper = self.create_incremental_wrapper(new_layer)

            train_loss, train_score, train_out, \
            val_loss, val_score, val_out, \
            _, _, test_out = incremental_training_wrapper.train(train_loader=train_loader,
                                                                validation_loader=val_loader,
                                                                test_loader=test_loader,
                                                                max_epochs=self.model_config.layer_config['epochs'],
                                                                logger=logger)

            for loader, out, mode in [(train_loader, train_out, 'train'), (val_loader, val_out, 'validation'),
                                      (test_loader, test_out, 'test')]:
                v_out, e_out, g_out, vo_out, eo_out, go_out = out

                # Reorder outputs, which are produced in shuffled order, to the original arrangement of the dataset.
                v_out, e_out, g_out, vo_out, eo_out, go_out = self._reorder_shuffled_objects(v_out, e_out, g_out,
                                                                                             vo_out, eo_out, go_out,
                                                                                             loader)

                # Store outputs
                self._store_outputs(mode, depth, v_out, e_out, g_out, vo_out, eo_out, go_out)

            depth += 1

        # NOW LOAD ALL EMBEDDINGS AND STORE THE EMBEDDINGS DATASET ON a torch file.

        # Consider all previous layers now, i.e. gather all the embeddings
        prev_outputs_to_consider = [l for l in range(1, depth + 1)]
        prev_outputs_to_consider.reverse()  # load output will concatenate in reverse order

        # Retrieve only the graph embeddings to save memory.
        # In CGMM classfication task (see other experiment file), I will ignore the outer val and reuse the inner val as validation, as I cannot use the splitter.
        train_out = self._create_extra_dataset(prev_outputs_to_consider, mode='train', depth=depth, only_g=True)
        val_out = self._create_extra_dataset(prev_outputs_to_consider, mode='validation', depth=depth, only_g=True)
        test_out = self._create_extra_dataset(prev_outputs_to_consider, mode='test', depth=depth, only_g=True)

        # Necessary info to give a unique name to the dataset (some hyper-params like epochs are assumed to be fixed)
        embeddings_folder = self.model_config.layer_config['embeddings_folder']
        max_layers = self.model_config.layer_config['max_layers']
        unibigram = self.model_config.layer_config['unibigram']
        C = self.model_config.layer_config['C']
        CA = self.model_config.layer_config['CA'] if 'CA' in self.model_config.layer_config else None
        aggregation = self.model_config.layer_config['aggregation']
        infer_with_posterior = self.model_config.layer_config['infer_with_posterior']
        outer_k = dataset_getter.outer_k
        inner_k = dataset_getter.inner_k
        # ====

        if not os.path.exists(os.path.join(embeddings_folder, dataset_getter.dataset_name)):
            os.makedirs(os.path.join(embeddings_folder, dataset_getter.dataset_name))

        unigram_dim = C + CA if CA is not None else C
        assert unibigram == True
        # Retrieve unigram if necessary
        for unib in [False, True]:
            base_path = os.path.join(embeddings_folder, dataset_getter.dataset_name,
                                     f'{max_layers}_{unib}_{C}_{CA}_{aggregation}_{infer_with_posterior}_{outer_k + 1}_{inner_k + 1}')
            train_out_emb = torch.cat([d.g_outs if unib else d.g_outs[:, :, :unigram_dim] for d in train_out], dim=0)
            torch.save(train_out_emb, base_path + '_train.torch')
            val_out_emb = torch.cat([d.g_outs if unib else d.g_outs[:, :, :unigram_dim] for d in val_out], dim=0)
            torch.save(val_out_emb, base_path + '_val.torch')
            test_out_emb = torch.cat([d.g_outs if unib else d.g_outs[:, :, :unigram_dim] for d in test_out], dim=0)
            torch.save(test_out_emb, base_path + '_test.torch')

        # CLEAR OUTPUTS
        for mode in ['train', 'validation', 'test']:
            shutil.rmtree(os.path.join(self.output_folder, mode), ignore_errors=True)

        return {'main_score': torch.zeros(1)}, {'main_score': torch.zeros(1)}

    def run_test(self, dataset_getter, logger):
        return {'main_score': torch.zeros(1)}, {'main_score': torch.zeros(1)}
