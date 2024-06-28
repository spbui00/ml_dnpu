"""
Author: Thai Ha Bui
Description: This script is designed to compare the uncertainties of KMC steps in different directories. It loads the uncertainties from each directory and plots the distributions of the uncertainties. It also calculates the mean uncertainties for each directory and plots them.
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    data = np.loadtxt(file_path, comments='#', delimiter=' ')
    return data[:, -2], data[:, -1]  # Return the output values and uncertainties

def plot_uncertainty_distributions(data_dict):
    plt.figure(figsize=(12, 6))
    for label, (outputs, uncertainties) in data_dict.items():
        plt.hist(uncertainties, bins=150, alpha=0.7, label=f'{label}', histtype='barstacked', linewidth=1, density=True)
    # plt.title('Distributions of KMC Steps Uncertainties')
    plt.xlabel('Uncertainty (nA)')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()

def plot_mean_uncertainties(uncertainties):
    labels = list(uncertainties.keys())
    values = list(uncertainties.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['#62A0CB', '#FFA555', '#6CBC6C'])
    # plt.title('Mean Uncertainties of KMC Steps')
    plt.xlabel('KMC Steps')
    plt.ylabel('Mean Uncertainty')
    plt.show()

if __name__ == "__main__":
    steps_list = ['1e3', '1e4', '1e5']
    data_dict = {}
    uncertainties = {}

    for steps in steps_list:
        file_path = f'../samples/{steps}/IO_uncertainties.dat'
        outputs, uncertainty_values = load_data(file_path)
        data_dict[f'{steps} steps'] = (outputs, uncertainty_values)
        uncertainties[f'{steps} steps'] = np.mean(uncertainty_values)  # Use the mean of the uncertainties

    plot_uncertainty_distributions(data_dict)
    plot_mean_uncertainties(uncertainties)