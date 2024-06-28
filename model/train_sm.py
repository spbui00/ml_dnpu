"""
Author: Thai Ha Bui
Description: This script is used to train a surrogate model using the BRAINSPY package. It loads the configuration, generates the surrogate model, and saves it to a specified directory.
"""

import yaml
from bspysmg.model.training import generate_surrogate_model

# Load the configuration
with open('configs/smg_configs.yaml', 'r') as file:
    smg_configs = yaml.safe_load(file)

# Generate the surrogate model
saved_dir = generate_surrogate_model(smg_configs)

print(f"Model training completed and saved in {saved_dir}")