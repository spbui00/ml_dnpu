from surrogate_model import SurrogateModel

# Usage example:
if __name__ == "__main__":
    data_path = "output/conv_model/training_data_2024_05_25_160221"
    model = SurrogateModel(data_path)
    input_data = [0.0, 0.4297, 0.0, -0.2494, -0.5721, 0.2002, -0.8976]
    predictions = model.predict(input_data)
    print(predictions)