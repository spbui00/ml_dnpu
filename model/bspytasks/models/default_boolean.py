import torch

from brainspy.processors.dnpu import DNPU
from brainspy.processors.processor import Processor
from brainspy.utils.pytorch import TorchUtils


class DefaultCustomModel(DNPU):
    def __init__(self, configs):
        # For this simple example, we just need a simple instance of a DNPU, but where input indices are defined
        # already in the configs. The input indices are the electrodes that will be receiving the two dimensional
        # data for the boolean gates task.

        # In order to load a surrogate model, the data can be extracted from the training_data.pt
        # generated during the training with the smg.

        # This data contains the info dictionary, required to know, among other things,
        # the structure used in the neural network for training the device
        # (In this example 5 layers of 90 nodes each, with ReLU as activation function).
        # Additionally, this file contains the model_state_dict, which
        # contains the weight values for the trained neural network simulating the DNPU.

        # The following line, is very similar to that used for initialising the hardware in notebook
        # number 2. But it now contains the info dictionary and the model_state_dict keys.
        if configs['processor_type'] == 'simulation':
            model_data = torch.load(configs['model_dir'],
                                    map_location=TorchUtils.get_device())
            super(DefaultCustomModel, self).__init__(
                Processor(configs,
                          model_data['info'],
                          model_data['model_state_dict'],
                          average_plateaus=False), [configs['input_indices']])
        else:
            super(DefaultCustomModel, self).__init__(
                Processor(configs, average_plateaus=False),
                data_input_indices=[configs['input_indices']],
            )

        # Additonally, we know that the data that we will be receiving for our example will be in a range from -1 to 1.
        # brains-py supports automatic transformation of the inputs, to the voltage ranges of the selected input indices.
        # This is done with the following line:
        self.add_input_transform([0, 1])
