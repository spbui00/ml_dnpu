import os
import torch
from brainspy.utils.pytorch import TorchUtils

from brainspy.utils.io import load_configs

results_dir = "output/boolean_gd/[0, 1, 1, 1]_2024_06_26_113239"

reloaded_configs = load_configs(os.path.join(results_dir,'reproducibility','configs.yaml'))
training_data = torch.load(os.path.join(results_dir,'reproducibility','training_data.pickle'), map_location=TorchUtils.get_device())
print(training_data['model_state_dict']['control_voltages'])