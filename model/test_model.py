"""
Author: Thai Ha Bui
Description: This script contains functions for loading data, calculating RMSE, normalized RMSE, separation line, and the main function for testing the surrogate model.
"""

import numpy as np
import matplotlib.pyplot as plt
from surrogate_model import SurrogateModel
from sklearn.metrics import mean_squared_error

AND = [['00', '01', '10'], ['11']]
OR = [['00'], ['01', '10', '11']]
XOR = [['00', '11'], ['01', '10']]

def load_data(file_path):
    data = np.loadtxt(file_path)
    inputs = data[:, :-1]
    outputs = data[:, -1]
    return inputs, outputs

def calculate_rmse(predictions, targets):
    return np.sqrt(mean_squared_error(targets, predictions))

def calculate_normalized_rmse(predictions, targets):
    rmse = calculate_rmse(predictions, targets)
    range_of_values = np.max(targets) - np.min(targets)
    normalized_rmse = (rmse / range_of_values) * 100
    return normalized_rmse

def calculate_separation_line(combinations, true_outputs, zero_groups, one_groups):
    highest_zero = max([val for comb, val in zip(combinations, true_outputs) if comb in zero_groups])
    lowest_one = min([val for comb, val in zip(combinations, true_outputs) if comb in one_groups])
    separation_line = (highest_zero + lowest_one) / 2
    return separation_line

def main():
    # provide the path to the surrogate model
    data_path = "output/conv_model/100000"
    # provide the path to the IO data (data from simulation or device)
    io_data_path = "../samples/boolean/2024-06-26_00-36-06 (xor)/IO.dat"

    # Load the surrogate model
    sm = SurrogateModel(data_path)

    # Load the data
    inputs, true_outputs = load_data(io_data_path)

    # Get predictions from the surrogate model
    predictions = []
    for input_data in inputs:
        prediction = sm.predict(input_data).item()
        predictions.append(prediction)

    predictions = np.array(predictions)

    # Calculate RMSE
    normalized_rmse = calculate_normalized_rmse(predictions, true_outputs)

    # Generate x-axis labels for combinations
    combinations = [''.join(map(str, map(int, input_data[2:5:2] > 0))) for input_data in inputs]

    # Plot the results
    plt.figure()
    gate_configs = ['00', '01', '10', '11']
    true_values = {config: [] for config in gate_configs}
    predicted_values = {config: [] for config in gate_configs}

    for comb, true_val, pred_val in zip(combinations, true_outputs, predictions):
        true_values[comb].append(true_val)
        predicted_values[comb].append(pred_val)

    avg_predicted_values = [np.mean(predicted_values[config]) for config in gate_configs]

    plt.step(gate_configs, avg_predicted_values, label='Predictions', color='blue', where='mid', linewidth=1)
    plt.scatter(gate_configs, avg_predicted_values, color='blue', s=50)  # Add points for clarity

    for config in gate_configs:
        config_numeric = gate_configs.index(config)  # Convert config to a numeric value
        plt.scatter(np.array([config_numeric] * len(true_values[config])), true_values[config], color='red', alpha=0.6, s=10, label='Simulation Values' if config == '00' else "")  # Add true values points with smaller size

    # Calculate the separation line
    separation_line = calculate_separation_line(combinations, true_outputs, XOR[0], XOR[1])
    highest_zero = max([val for comb, val in zip(combinations, true_outputs) if comb in XOR[0]])
    lowest_one = min([val for comb, val in zip(combinations, true_outputs) if comb in XOR[1]])
    # Plot the separation line
    plt.axhline(y=separation_line, color='black', linestyle='--', linewidth=1)
    plt.text(1.5, separation_line, f'Separation Line = ({highest_zero:.2f} + {lowest_one:.2f}) / 2 = {separation_line:.2f}', ha='center', va='bottom', color='black')

    plt.xticks(range(len(gate_configs)), gate_configs) 
    plt.xlabel('Input pairs')
    plt.ylabel('Current (nA)')
    plt.title(f'RMSE: {normalized_rmse:.2f}%')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()