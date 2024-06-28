import os
import torch
from brainspy.utils.pytorch import TorchUtils

from brainspy.utils.io import load_configs
from brainspy.utils import manager

from bspytasks.boolean.tasks.classifier import boolean_task
from bspytasks.models.default_boolean import DefaultCustomModel

# log_dir = "output/and_gate"
# logger = Logger(log_dir)

def custom_transform(sample):
    inputs, targets = sample
    inputs = torch.where(inputs == -0.5, torch.tensor(0.0), inputs)
    inputs = torch.where(inputs == 1.0, torch.tensor(0.5), inputs)
    return inputs, targets

# Load the configuration
configs = load_configs("configs/boolean_gd.yaml")

# Load the surrogate model
model_path = 'output/conv_model/100000/training_data.pt'
configs['processor']['model_dir'] = model_path

# Define the criterion and algorithm
criterion = manager.get_criterion(configs["algorithm"]['criterion'])
# algorithm = manager.get_algorithm(configs["algorithm"]['optimizer'])
algorithm = manager.get_algorithm(configs["algorithm"]['type'])

# Train the model
results = boolean_task(configs, DefaultCustomModel, criterion, algorithm)
results_dir = results['main_dir']

reloaded_configs = load_configs(os.path.join(results_dir,'reproducibility','configs.yaml'))
training_data = torch.load(os.path.join(results_dir,'reproducibility','training_data.pickle'), map_location=TorchUtils.get_device())
print(training_data['model_state_dict']['control_voltages'])