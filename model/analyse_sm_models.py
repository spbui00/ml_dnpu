from surrogate_model import SurrogateModel
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

"""
This script is used to compare the performance of multiple surrogate models.
It generates a set of random inputs and compares the predictions of each model.
The script then calculates the mean squared error (MSE) between each pair of models.
The script also calculates the root mean squared error (RMSE) and the normalized mean squared error (NMSE).
"""

def generate_inputs(num_inputs):
    return np.random.uniform(-1, 1, (num_inputs, 7))

def load_models(data_paths):
    models = [SurrogateModel(data_path) for data_path in data_paths]
    return models

def compare_models(models, input_data):
    predictions = [model.predict(input_data).detach().numpy() for model in models]  # Detach and convert tensor to numpy array
    predictions = np.array(predictions).squeeze()  # Remove single-dimensional entries
    
    # Ensure predictions are 2D arrays with at least two elements
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
    if predictions.shape[1] == 1:
        predictions = np.hstack([predictions, predictions])
    
    # Calculate pairwise MSE
    num_models = len(models)
    mse_matrix = np.zeros((num_models, num_models))
    
    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                mse_matrix[i, j] = mean_squared_error(predictions[i], predictions[j])
    
    return mse_matrix, predictions

def compare_multiple_inputs(data_paths, num_inputs):
    models = load_models(data_paths)
    
    # Generate random inputs in the range [-1, 1]
    input_data_list = [np.random.uniform(-1, 1, 7) for _ in range(num_inputs)]  # Always 7 features
    
    all_mse_values = []
    all_predictions = []
    
    for idx, input_data in enumerate(input_data_list):
        mse_matrix, predictions = compare_models(models, input_data)
        all_mse_values.extend(mse_matrix[np.triu_indices(len(models), k=1)])  # Collect upper triangle values excluding diagonal
        all_predictions.extend(predictions.flatten())
    
    return all_mse_values, all_predictions

def visualize_mse_distribution(mse_values):
    plt.figure(figsize=(10, 6))
    plt.hist(mse_values, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title('Distribution of MSE Values Between Model Predictions')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def load_data(file_path):
    data = np.loadtxt(file_path)
    features = data[:, :-1] 
    true_outputs = data[:, -1]  
    return features, true_outputs

def predict_and_evaluate(model, features, true_outputs):
    predicted_outputs = model.predict(features).detach().numpy()  # Detach and convert to numpy
    predicted_outputs = np.squeeze(predicted_outputs)  # Ensure the shape is consistent
    rmse = np.sqrt(mean_squared_error(true_outputs, predicted_outputs))
    return predicted_outputs, rmse

def plot_output_distribution(model, num_inputs=10000, bins=50):
    """
    Plots the distribution of the model's outputs given the input data.
    
    Parameters:
    - model: The surrogate model to use for predictions.
    - num_inputs: Number of input sets to compare.
    - bins: Number of bins for the histogram.
    """
    input_data = generate_inputs(num_inputs)

    # Predict outputs using the model
    predicted_outputs = model.predict(input_data).detach().numpy()
    predicted_outputs = np.squeeze(predicted_outputs) 
    
    # Plot the histogram of the predicted outputs
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_outputs, bins=bins, alpha=0.75, color='blue', edgecolor='black')
    plt.title('Output Histogram')
    plt.xlabel('Raw output (nA)')
    plt.ylabel('Counts')
    plt.grid(False)
    plt.show()

def plot_true_vs_predicted(true_outputs, predicted_outputs):
    plt.figure(figsize=(10, 6))
    plt.scatter(true_outputs, predicted_outputs, alpha=0.6, color='#0000F7', edgecolor='k', s=20)
    plt.plot([min(true_outputs), max(true_outputs)], [min(true_outputs), max(true_outputs)], color='red', linestyle='--')
    plt.title('True vs. Predicted Outputs')
    plt.xlabel('True Outputs')
    plt.ylabel('Predicted Outputs')
    plt.grid(True)
    plt.show() 

def plot_predictions_line(true_outputs, predicted_outputs, subset_size=100):
    indices = np.random.choice(len(true_outputs), subset_size, replace=False)
    true_subset = true_outputs[indices]
    predicted_subset = predicted_outputs[indices]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(subset_size), true_subset, label='True Outputs', marker='o', color='black')
    plt.plot(range(subset_size), predicted_subset, label='Predicted Outputs', marker='x', linestyle='--', color='red')
    plt.title('Comparison of True and Predicted Outputs (Line Plot)')
    plt.xlabel('Sample Index')
    plt.ylabel('Output Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_mse_values(mse_values, all_predictions):
    # Visualize the distribution of MSE values
    visualize_mse_distribution(mse_values)
    
    # Calculate RMSE
    rmse_values = np.sqrt(mse_values)
    
    # Calculate the range of the output values
    output_min = np.min(all_predictions)
    output_max = np.max(all_predictions)
    output_range = output_max - output_min
    
    # Calculate NMSE (normalized by the range of the output values)
    nmse_values = mse_values / (output_range ** 2)
    
    # Print summary statistics
    print(f"Mean MSE: {np.mean(mse_values)}")
    print(f"Standard Deviation of MSE: {np.std(mse_values)}")
    print(f"Minimum MSE: {np.min(mse_values)}")
    print(f"Maximum MSE: {np.max(mse_values)}")
    
    # Print RMSE summary statistics
    print(f"Mean RMSE: {np.mean(rmse_values)}")
    print(f"Standard Deviation of RMSE: {np.std(rmse_values)}")
    print(f"Minimum RMSE: {np.min(rmse_values)}")
    print(f"Maximum RMSE: {np.max(rmse_values)}")
    
    # Print NMSE summary statistics
    print(f"Mean NMSE: {np.mean(nmse_values)}")
    print(f"Standard Deviation of NMSE: {np.std(nmse_values)}")
    print(f"Minimum NMSE: {np.min(nmse_values)}")
    print(f"Maximum NMSE: {np.max(nmse_values)}")

if __name__ == "__main__":

    # models trained on same data (10 000)
    data_paths = [
        "output/conv_model/training_data_2024_05_28_145003",
        "output/conv_model/training_data_2024_05_28_124422",
        "output/conv_model/training_data_2024_06_09_112047",
        "output/conv_model/100000"
    ]
    
    # ====================================================================================
    # Compare multiple SM

    # num_inputs = 10000  # Number of input sets to compare
    
    # mse_values, all_predictions = compare_multiple_inputs(data_paths, num_inputs)

    # analyze_mse_values(mse_values, all_predictions)
    # ====================================================================================

    # ====================================================================================
    # analyse one model

    model = SurrogateModel(data_path='output/conv_model/100000')
    samples_path = "../samples/100000_input_pairs/IO.dat"

    features, true_outputs = load_data(samples_path)

    predicted_outputs, rmse = predict_and_evaluate(model, features, true_outputs)

    # Calculate mean squared error
    mse = np.mean((true_outputs - predicted_outputs) ** 2)

    print(f"Mean Squared Error: {mse}")

    plot_true_vs_predicted(true_outputs, predicted_outputs)
    plot_predictions_line(true_outputs, predicted_outputs, 100)

    # ====================================================================================

    # ====================================================================================
    # distribution of outputs

    # model = load_models([data_paths[2]])[0]

    # plot_output_distribution(model, num_inputs=1000000, bins=100)
    

    


