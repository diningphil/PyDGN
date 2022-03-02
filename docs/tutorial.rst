Tutorial
======================
Knowing how to set up valid YAML configuration files is fundamental to properly use **PyDGN**. Custom behavior with
more advanced functionalities can be generally achieved by subclassing the individual modules we provide,
but this is very much dependent on the specific research project.

Data Preprocessing
***********************

The ML pipeline starts with the creation of the dataset and of the data splits. The general template that we can use is
the following, with an explanation of each field as a comment:

.. code-block:: yaml

    splitter:
      root:  # folder where to store the splits
      class_name:  # dotted path to splitter class
      args:
        n_outer_folds:  # number of outer folds for risk assessment
        n_inner_folds:  # number of inner folds for model selection
        seed:
        stratify:  # target stratification: works for graph classification tasks only
        shuffle:  # whether to shuffle the indices prior to splitting
        inner_val_ratio:  # percentage of validation for hold-out model selection. this will be ignored when the number of inner folds is > than 1
        outer_val_ratio:  # percentage of validation data to extract for risk assessment final runs
        test_ratio:  # percentage of test to extract for hold-out risk assessment. this will be ignored when the number of outer folds is > than 1
    dataset:
      root:  # path to data root folder
      class_name:  # dotted path to dataset class
      args:  # arguments to pass to the dataset class
        arg_name1:
        arg_namen:
      transform: # on the fly transforms: useful for social datasets with no node features (with an example)
        - class_name: pydgn.data.transform.ConstantIfEmpty
          args:
            value: 1
      # pre_transform:  # transform data and store it at dataset creation time
      # pre_filter:  # filter data and store it at dataset creation time


Data Splitting
-------------------

We provide a general :class:`~pydgn.data.splitter.Splitter` class that is able to split a dataset of multiple graphs. The most important parameters
are arguably ``n_outer_folds`` and ``n_inner_folds``, which represent the way in which we want to perform **risk assessment**
and **model selection**. For instance:

 * ``n_outer_folds=10`` and ``n_inner_folds=1``: 10-fold external Cross Validation (CV) on test data, with hold-out model selection inside each of the 10 folds,
 * ``n_outer_folds=5`` and ``n_inner_folds=3``: Nested CV,
 * ``n_outer_folds=1`` and ``n_inner_folds=1``: Simple Hold-out model assessment and selection, or ``train/val/test`` split.

We assume that the difference between **risk assessment** and **model selection** is clear to the reader.
If not, please refer to `Samy Bengio's lecture (Part 3) <https://bengio.abracadoudou.com/lectures/theory.pdf>`_.

Here's an snippet of a potential configuration file that splits a graph classification dataset:

.. code-block:: yaml

    splitter:
      root: examples/DATA_SPLITS/CHEMICAL
      class_name: pydgn.data.splitter.Splitter
      args:
        n_outer_folds: 10
        n_inner_folds: 1
        seed: 42
        stratify: True
        shuffle: True
        inner_val_ratio: 0.1
        outer_val_ratio: 0.1
        test_ratio: 0.1

Dataset Creation
-------------------

To create your own dataset, you should implement the :class:`~pydgn.data.dataset.DatasetInterface` interface. For
instance, we provide a wrapper around the `TUDataset <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.TUDataset>`_
dataset of Pytorch Geometric in :class:`~pydgn.data.dataset.TUDatasetInterface`, which you can check to get an idea.

Here's an snippet of a potential configuration file that downloads and processes the ``PROTEINS`` graph classification dataset:

.. code-block:: yaml

    dataset:
      root: DATA/
      class_name: pydgn.data.dataset.TUDataset
      args:
        root: DATA/
        name: PROTEINS

You can also apply ``transform``, ``pre_transform`` and ``pre_filter`` that follow the same semantic of PyG.

Once our data configuration file is ready, we can create the dataset using (for the example above)

.. code-block:: python
    pydgn-dataset --config-file examples/DATA_CONFIGS/config_PROTEINS.yml

Experiment Setup
**********************

Once we have created a dataset and its data splits, it is time to implement our model and define a suitable task.
Every model must implement the :class:`~pydgn.model.interface.ModelInterface` interface, and it can optionally use a
readout module that must implement the :class:`~pydgn.model.interface.ReadoutInterface`.

At this point, it is time to define the experiment. The general template that we can use is the following, with an
explanation of each field as a comment:

.. code-block:: python

    # Dataset and Splits
    data_root:  # path to DATA root folder (same as in data config file)
    dataset_class:  # dotted path to dataset class
    dataset_name:  # dataset name (same as in data config file)
    data_splits_file:  # path to data splits file


    # Hardware
    device:  # cpu | cuda
    max_cpus:  # > 1 for parallelism
    max_gpus: # > 0 for gpu usage (device must be cuda though)
    gpus_per_task:  # percentage of gpus to allocate for each task


    # Data Loading
    dataset_getter:  # dotted path to dataset provider class
    data_loader:
      class_name:  # dotted path to data loader class
      args:
        num_workers :
        pin_memory:
        # possibly other arguments (we set `worker_init_fn`, `sampler` and `shuffle`, so do not override)


    # Reproducibility
    seed: 42


    # Experiment
    result_folder:  # path of the folder where to store results
    exp_name:  # name of the experiment
    experiment:  # dotted path to experiment class
    higher_results_are_better:  # model selection: should we select based on max (True) or min (False) main score?
    evaluate_every:  # evaluate on train/val/test every `n` epochs and log results
    final_training_runs:  # how many final (model assessment) training runs to perform to mitigate bad initializations

    # Grid Search
    # if only 1 configuration is selected, any inner model selection will be skipped
    grid:
      supervised_config:
        model:  # dotted path to model class
        checkpoint:  # whether to keep a checkpoint of the last epoch to resume training
        shuffle:  # whether to shuffle the data
        batch_size:  # batch size
        epochs:  # number of maximum training epochs

        # Model specific arguments #

        # TBD

        # ------------------------ #

        # Optimizer (with an example - 3 possible alternatives)
        optimizer:
          - class_name: pydgn.training.callback.optimizer.Optimizer
            args:
              optimizer_class_name: torch.optim.Adam
              lr:
                - 0.01
                - 0.001
              weight_decay: 0.
          - class_name: pydgn.training.callback.optimizer.Optimizer
            args:
              optimizer_class_name: torch.optim.Adagrad
              lr:
                - 0.1
              weight_decay: 0.

        # Scheduler (optional)
        scheduler: null

        # Loss metric (with an example of Additive Loss)
        loss:
          - class_name: pydgn.training.callback.metric.AdditiveLoss
            args:
              loss_1: pydgn.training.callback.metric.MulticlassClassification
              loss_2: pydgn.training.callback.metric.MulticlassClassification

        # Score metric (with an example of Multi Score)
        scorer:
          - class_name: pydgn.training.callback.metric.MultiScore
            args:
              main_scorer: pydgn.training.callback.metric.MulticlassAccuracy
              my_second_metric: pydgn.training.callback.metric.ToyMetric

        # Readout (optional)
        readout:

        # Training engine
        engine: pydgn.training.engine.TrainingEngine

        # Gradient clipper (optional)
        gradient_clipper: null

        # Early stopper (optional, with an example of "patience" early stopping on the validation score)
        early_stopper:
          - class_name:
              - pydgn.training.callback.early_stopping.PatienceEarlyStopper
            args:
              patience:
                - 5
              # SYNTAX: (train_,validation_)[name_of_the_scorer_or_loss_to_monitor] -> we can use MAIN_LOSS or MAIN_SCORE
              monitor: validation_main_score
              mode: max  # is best the `max` or the `min` value we are monitoring?
              checkpoint: True  # store the best checkpoint

        # Plotter of metrics
        plotter: pydgn.training.callback.plotter.Plotter


Data Information
-----------------


Hardware
-----------------


Data Loading
-----------------


Experiment Details
--------------------



Grid Search
--------------

There is one config file ``examples/MODEL_CONFIGS/config_SupToyDGN.yml`` that you can check.


Random Search
--------------

Specify a ``num_samples`` in the config file with the number of random trials, replace ``grid``
with ``random``, and specify a sampling method for each hyper-parameter. We provide different sampling methods:

- choice --> pick at random from a list of arguments
- uniform --> pick uniformly from min and max arguments
- normal --> sample from normal distribution with mean and std
- randint --> pick at random from min and max
- loguniform --> pick following the recprocal distribution from log_min, log_max, with a specified base

There is one config file ``examples/MODEL_CONFIGS/config_SupToyDGN_RandomSearch.yml`` that you can check.


Experiment
--------------

.. code-block:: python

    """This example demonstrates a simple BLE client that scans for devices,
    connects to a device (GATT server) of choice and continuously reads a characteristic on that device.

    The GATT Server in this example runs on an ESP32 with Arduino. For the
    exact script used for this example see `here <https://github.com/nkolban/ESP32_BLE_Arduino/blob/6bad7b42a96f0aa493323ef4821a8efb0e8815f2/examples/BLE_notify/BLE_notify.ino/>`_
    """

    from bluepy.btle import *
    from simpleble import SimpleBleClient, SimpleBleDevice

    # The UUID of the characteristic we want to read and the name of the device # we want to read it from
    Characteristic_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
    Device_Name = "MyESP32"

    # Define our scan and notification callback methods
    def myScanCallback(client, device, isNewDevice, isNewData):
        client._yes = True
        print("#MAC: " + device.addr + " #isNewDevice: " +
            str(isNewDevice) + " #isNewData: " + str(isNewData))
    # TODO: NOTIFICATIONS ARE NOT SUPPORTED YET
    # def myNotificationCallback(client, characteristic, data):
    #     print("Notification received!")
    #     print("  Characteristic UUID: " + characteristic.uuid)
    #     print("  Data: " + str(data))

    # Instantiate a SimpleBleClient and set it's scan callback
    bleClient = SimpleBleClient()
    bleClient.setScanCallback(myScanCallback)
    # TODO: NOTIFICATIONS ARE NOT SUPPORTED YET
    # bleClient.setNotificationCallback(myNotificationCollback)

    # Error handling to detect Keyboard interrupt (Ctrl+C)
    # Loop to ensure we can survive connection drops
    while(not bleClient.isConnected()):
        try:
            # Search for 2 seconds and return a device of interest if found.
            # Internally this makes a call to bleClient.scan(timeout), thus
            # triggering the scan callback method when nearby devices are detected
            device = bleClient.searchDevice(name="MyESP32", timeout=2)
            if(device is not None):
                # If the device was found print out it's info
                print("Found device!!")
                device.printInfo()

                # Proceed to connect to the device
                print("Proceeding to connect....")
                if(bleClient.connect(device)):

                    # Have a peek at the services provided by the device
                    services = device.getServices()
                    for service in services:
                        print("Service ["+str(service.uuid)+"]")

                    # Check to see if the device provides a characteristic with the
                    # desired UUID
                    counter = bleClient.getCharacteristics(
                        uuids=[Characteristic_UUID])[0]
                    if(counter):
                        # If it does, then we proceed to read its value every second
                        while(True):
                            # Error handling ensures that we can survive from
                            # potential connection drops
                            try:
                                # Read the data as bytes and convert to string
                                data_bytes = bleClient.readCharacteristic(
                                    counter)
                                data_str = "".join(map(chr, data_bytes))

                                # Now print the data and wait for a second
                                print("Data: " + data_str)
                                time.sleep(1.0)
                            except BTLEException as e:
                                # If we get disconnected from the device, keep
                                # looping until we have reconnected
                                if(e.code == BTLEException.DISCONNECTED):
                                    bleClient.disconnect()
                                    print(
                                        "Connection to BLE device has been lost!")
                                    break
                                    # while(not bleClient.isConnected()):
                                    #     bleClient.connect(device)

                else:
                    print("Could not connect to device! Retrying in 3 sec...")
                    time.sleep(3.0)
            else:
                print("Device not found! Retrying in 3 sec...")
                time.sleep(3.0)
        except BTLEException as e:
            # If we get disconnected from the device, keep
            # looping until we have reconnected
            if(e.code == BTLEException.DISCONNECTED):
                bleClient.disconnect()
                print(
                    "Connection to BLE device has been lost!")
                break
        except KeyboardInterrupt as e:
            # Detect keyboard interrupt and close down
            # bleClient gracefully
            bleClient.disconnect()
            raise e