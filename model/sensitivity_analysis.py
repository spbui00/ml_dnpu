"""
Author: Thai Ha Bui
Description: This script contains functions for performing sensitivity analysis on surrogate models. It includes methods for loading models, perturbing inputs, and visualizing sensitivity results.
"""

from surrogate_model import SurrogateModel
import numpy as np
import matplotlib.pyplot as plt

def load_models(data_paths):
    models = [SurrogateModel(data_path) for data_path in data_paths]
    return models

def perturb_inputs(base_input, perturbation, index):
    perturbed_input = base_input.copy()
    perturbed_input[index] += perturbation
    return perturbed_input

def sensitivity_analysis(models, base_inputs, perturbations):
    """
    Perform sensitivity analysis on the surrogate models.
    
    Args:
        models (list of SurrogateModel): List of surrogate models.
        base_inputs (np.ndarray): Array of base input samples.
        perturbations (list of float): List of perturbation magnitudes.
    
    Returns:
        np.ndarray: Sensitivity values for each model, feature, and perturbation.
    """
    num_features = base_inputs.shape[1]
    sensitivities = np.zeros((len(models), num_features, len(perturbations)))
    
    for p_idx, perturbation in enumerate(perturbations):
        for i in range(num_features):
            perturbed_inputs = np.array([perturb_inputs(base_input, perturbation, i) for base_input in base_inputs])
            base_predictions = np.array([model.predict(base_inputs).detach().numpy() for model in models])
            perturbed_predictions = np.array([model.predict(perturbed_inputs).detach().numpy() for model in models])
            
            for j in range(len(models)):
                sensitivities[j, i, p_idx] = np.abs(perturbed_predictions[j] - base_predictions[j]).mean()
    
    return sensitivities

def visualize_sensitivity(sensitivities, perturbations):
    num_models, num_features, num_perturbations = sensitivities.shape
    x = np.arange(num_features)
    
    # Rename feature names to "Input 1, 2, ..."
    input_names = [f'Input {i+1}' for i in range(num_features)]
    
    plt.figure(figsize=(12, 6))
    
    # Define colors and line styles
    colors = ['blue', 'orange', 'green', 'red']
    line_styles = ['-', '--', '-.', ':']
    
    for p_idx, perturbation in enumerate(perturbations):
        for i in range(num_models):
            plt.plot(x, sensitivities[i, :, p_idx], label=f'Model {i+1}, Perturbation {perturbation}', color=colors[p_idx], linestyle=line_styles[i])
    
    plt.xticks(x, input_names, rotation=45)
    plt.xlabel('Input Features')
    plt.ylabel('Sensitivity')
    plt.legend(loc='upper right', framealpha=0.7)  # Adjust legend location and transparency
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_paths = [
        "output/conv_model/training_data_2024_05_28_100232",
        # "output/conv_model/training_data_2024_05_26_151130",
        # "output/conv_model/training_data_2024_05_27_122553",
        # "output/conv_model/training_data_2024_05_28_124422",
        "output/conv_model/training_data_2024_06_09_112047",
        "output/conv_model/100000",
        "output/conv_model/70k_1e5"
        # "output/conv_model/100000_input_pairs_1"
    ]
    
    models = load_models(data_paths)
    
    # Define a larger set of base inputs and feature names
    base_inputs = np.random.uniform(-1, 1, (100, 7))  # Example base inputs
    
    # Define different perturbation magnitudes
    perturbations = [-0.2, -0.5, 0.2, 0.5]
    
    # Perform sensitivity analysis
    sensitivities = sensitivity_analysis(models, base_inputs, perturbations)
    
    # Visualize the sensitivity analysis results
    visualize_sensitivity(sensitivities, perturbations)