import torch
from brainspy.processors.simulation.model import NeuralNetworkModel
from brainspy.utils.pytorch import TorchUtils

class SurrogateModel:
    def __init__(self, data_path: str):
        """
        Initializes the SurrogateModel with the path to the training data.

        Args:
            data_path (str): Path to the directory containing the training data .pt file.
        """
        self.data_path = data_path
        self.model = self._load_model()

    def _load_model(self) -> NeuralNetworkModel:
        """
        Loads the neural network model from the training data checkpoint.

        Returns:
            NeuralNetworkModel: The loaded neural network model.
        """
        checkpoint = torch.load(f'{self.data_path}/training_data.pt')
        model = NeuralNetworkModel(checkpoint['info']['model_structure'])
        model = TorchUtils.format(model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def predict(self, input_data: list) -> float:
        """
        Predicts the output for the given input data using the loaded model.

        Args:
            input_data (list): A list of input values.

        Returns:
            float: The predicted output value.
        """
        input_tensor = torch.tensor([input_data]).to(TorchUtils.get_device())
        with torch.no_grad():
            predictions = self.model(input_tensor)
        return predictions.item()  # Convert tensor to a single value