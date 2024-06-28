import torch
from brainspy.processors.simulation.model import NeuralNetworkModel
from brainspy.utils.pytorch import TorchUtils
import numpy as np
import matplotlib.pyplot as plt

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

    def predict(self, input_data: list) -> torch.Tensor:
        """
        Predicts the output for the given input data using the loaded model.

        Args:
            input_data (list): A list of input values.

        Returns:
            float: The predicted output value.
        """
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        
        # Convert input data to float32
        input_data = input_data.astype(np.float32)
        
        # Convert input data to a tensor
        input_tensor = torch.tensor(input_data).unsqueeze(0).to(TorchUtils.get_device())
        
        # Get predictions from the model
        predictions = self.model(input_tensor)
        
        return predictions
    
    # cross check values 00 01 10 11 (0.0 for 0 and 0.5 for 1) at indices 2 and 4, the input_voltages are other 5 voltages
    def validate_gate(self, input_voltages):
        hard_copy = input_voltages.copy()
        
        ZERO = -1.2
        ONE = 0.6

        # ZERO = 0
        # ONE = 0.5

        # Generate input voltages for different gate configurations
        input_voltages_00 = hard_copy[:2] + [ZERO] + hard_copy[2:3] + [ZERO] + hard_copy[3:]
        input_voltages_01 = hard_copy[:2] + [ZERO] + hard_copy[2:3] + [ONE] + hard_copy[3:]
        input_voltages_10 = hard_copy[:2] + [ONE] + hard_copy[2:3] + [ZERO] + hard_copy[3:]
        input_voltages_11 = hard_copy[:2] + [ONE] + hard_copy[2:3] + [ONE] + hard_copy[3:]

        print(input_voltages_00)
        print(input_voltages_01)
        print(input_voltages_10)
        print(input_voltages_11)

        # Get predictions
        predictions = [
            self.predict(input_voltages_00).item(),
            self.predict(input_voltages_01).item(),
            self.predict(input_voltages_10).item(),
            self.predict(input_voltages_11).item()
        ]

        # Print predictions
        print("00: ", predictions[0])
        print("01: ", predictions[1])
        print("10: ", predictions[2])
        print("11: ", predictions[3])

        # Plotting
        gate_configs = ['00', '01', '10', '11']

        plt.step(gate_configs, predictions, label='Prediction', color='blue', where='mid')
        plt.scatter(gate_configs, predictions, color='blue')  # Add points for clarity
        plt.xlabel('Gate Configuration')
        plt.ylabel('Current (nA)')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # data_path = "output/conv_model/training_data_2024_06_09_112047"
    data_path = "output/conv_model/100000"
    # data_path = "output/conv_model/100000_input_pairs_1"
    # data_path = "output/conv_model/70k_1e5"
    # data_path = "output/conv_model/70k_1e5_4000"
    # data_path = "output/conv_model/training_data_2024_06_15_220024"

    # Instantiate the SurrogateModel
    sm = SurrogateModel(data_path)

    # sm.validate_gate([-0.0839,  0.1401,  0.0581,  0.7374,  0.4974]) #XOR
    # sm.validate_gate([ -0.0854,  1.4072, -0.7048,  1.0727,  0.2209]) #AND
    # sm.validate_gate([0.5426, -0.9029, 0.6333, -0.4036, -0.4998]) #AND
    # sm.validate_gate([0.2039,  0.2379, -0.4031,  0.6261,  0.8185]) #1001
    # sm.validate_gate([-0.2401, -1.0018,  0.4030,  0.8886,  0.456]) #XOR custom fit
    # sm.validate_gate([0.5209, -0.0266, -0.6812, -1.0801,  0.1685]) #OR custom fit
    # sm.validate_gate([0.0145, 0.1273, 1.0213, 0.8672, 0.8862]) #AND custom fit
    # sm.validate_gate([0.0622,  2.2786,  0.0152,  0.7830, -0.0778]) #XOR custom fit 100 000 1e4
    sm.validate_gate([0.5520, -1.0111,  0.5123, -0.1667, -0.3607]) #XOR custom fit 70k 1e5
    # sm.validate_gate([0.0286, 0.7840, 0.5002, 0.8496, 0.8247]) #XOR custom fit 70k 1e5