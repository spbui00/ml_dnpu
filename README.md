# Bachelor thesis project

This repository serves as tool for the bachelor thesis project of Thai Ha Bui, June 2024, Department of Applied Mathematics
Faculty of Electrical Engineering, Mathematics and Computer Science, University of Twente, Netherlands. It contains data and code for the project.

Thesis paper: "[A minimal nanoneuron capable of nonlinear classification](https://r.search.yahoo.com/_ylt=Awr.QIP_kwRpszEkDApnXQx.;_ylu=Y29sbwMEcG9zAzgEdnRpZAMEc2VjA3Ny/RV=2/RE=1761936511/RO=10/RU=https%3a%2f%2fessay.utwente.nl%2f102266%2f/RK=2/RS=C81m_SOuTQNPxxUW_qe8b7ct3Qs-)"

## Setup

To set up the environment and dependencies for this project, follow the instructions provided in the following repositories:

- **KMC Sampling**: [MUTUEL/MCNetwork](https://github.com/MUTUEL/MCNetwork)
- **BRAINSPY Package**: [BraiNEdarwin/brains-py](https://github.com/BraiNEdarwin/brains-py)

# Model Folder

The `model` folder contains all the necessary files and scripts related to the custom model that uses the BRAINSPY package used in this project. The folder contains additional scripts for training, evaluating, and utilizing the custom model.

## Files in the Model Folder

### postprocess.py
This script is used for post-processing the data generated during the sampling phase (you have to have the samples first. For instance, if you can generate samples from a kinetic Monte Carlo simulation [here](https://github.com/spbui00/MCNetwork)). It loads the data, processes it, and saves it in a `.npz` format. The script uses the `post_process` function from the `bspysmg.data.postprocess` module.

### train_sm.py
This script is responsible for generating the surrogate model based on the configurations provided in a YAML file. It uses the `generate_surrogate_model` function from the `bspysmg.model.training` module.

### surrogate_model.py
This script defines a `SurrogateModel` class that loads a neural network model from a checkpoint and provides methods for making predictions and validating gate configurations. It also includes a main section for testing the model with different input voltages.

### train_functionality.py
This script handles the training of a custom model for a boolean task. It loads configurations, sets up the model, criterion, and algorithm, and then trains the model using the `boolean_task` function from the `bspytasks.boolean.tasks.classifier` module.

## Configuration Files

The `configs` folder contains various YAML configuration files that are used by the scripts in the `model` folder. These configurations include parameters for training the surrogate model, boolean task configurations, and sampling configurations.

- **sampling_configs.yaml**: Configuration for the data sampling process.
- **smg_configs.yaml**: Configuration for generating the surrogate model.
- **boolean_gd.yaml**: Configuration for the boolean task training.
