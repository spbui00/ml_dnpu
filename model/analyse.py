import torch
from brainspy.processors.simulation.model import NeuralNetworkModel
from brainspy.utils.pytorch import TorchUtils

DATA_PATH = "output/conv_model/training_data_2024_05_20_173625"

# Load the saved training data
checkpoint = torch.load(f'{DATA_PATH}/training_data.pt')

# Initialize the model architecture
model = NeuralNetworkModel(checkpoint['info']['model_structure'])
model = TorchUtils.format(model)

# Load the saved state dictionary into the model
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Example input tensor (replace with your actual input data)
# Input data from IO.dat
# 0.477687 0.332834 0.415886 -0.753566 0.854283 0.920716 0.711827 res:  0.0528936
input_data = torch.tensor([[0.477687, 0.332834, 0.415886, -0.753566, 0.854283, 0.920716, 0.711827]])

# Move input data to the appropriate device
input_data = input_data.to(TorchUtils.get_device())

# Make predictions
with torch.no_grad():
    predictions = model(input_data)

# Print the predictions
print(predictions)