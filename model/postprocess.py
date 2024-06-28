import numpy as np
import yaml
from brainspy.utils.io import load_configs
from bspysmg.data.sampling import Sampler
from bspysmg.data.postprocess import post_process

FILE_PATH = '../samples/100000'

inputs, outputs, info_dictionary = post_process(f'{FILE_PATH}', clipping_value=None)
print(f"max out {outputs.max()} max min {outputs.min()} shape {outputs.shape}")