import numpy as np
import yaml
from brainspy.utils.io import load_configs
from bspysmg.data.sampling import Sampler
from bspysmg.data.postprocess import post_process

FILE_PATH = '../samples/100000_input_pairs'

# Load the data from IO.dat
# data = np.loadtxt(f'{FILE_PATH}/IO.dat', delimiter=' ')

# # remove first row
# data = data[1:, :]

# # Separate inputs and outputs
# inputs = data[:, :-1]  
# outputs = data[:, -1]

# # Ensure outputs are 2D (N, 1) instead of 1D (N,)
# outputs = outputs.reshape(-1, 1)

# # Load sampling_configs from YAML file
# with open('configs/sampling_configs.yaml', 'r') as file:
#     sampling_configs = yaml.safe_load(file)

# # Save the data in .npz format
# np.savez(f'{FILE_PATH}/postprocessed_data.npz', inputs=inputs, outputs=outputs, sampling_configs=sampling_configs)

inputs, outputs, info_dictionary = post_process(f'{FILE_PATH}', clipping_value=None)
print(f"max out {outputs.max()} max min {outputs.min()} shape {outputs.shape}")