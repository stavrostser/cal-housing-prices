import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ========= CustomDataset class =========

class CustomDataset(Dataset):
    """
    A custom dataset class for handling data directly from a pandas dataframe.

    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Assuming the last column is the output/target
        data = torch.tensor(row[:-1].values, dtype=torch.float32)
        target = torch.tensor(row[-1], dtype=torch.float32)
        return data, target
    

class CustomDatasetAutoencoder(Dataset):
    """
    A custom dataset class for handling data directly from a pandas dataframe.
    To be used in autoencoder models where the input and output are the same.

    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        data = torch.tensor(row.values, dtype=torch.float32)
        target = data.clone() # Autoencoder output is the same as the input. Clone the data to have different memory reference.
        return data, target


# ========= Neural Network and Training classes =========

class SimpleNN(torch.nn.Module):
    def __init__(self, input_size=3, output_size=1, neurons=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, neurons)
        self.fc1a = torch.nn.Linear(neurons, neurons) # add another layer
        # self.fc1b = torch.nn.Linear(neurons, neurons) # add another layer
        self.fc2 = torch.nn.Linear(neurons, output_size)

    def forward(self, x):
        # print("input: ", x.shape)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc1a(x)  # add another layer
        x = torch.nn.functional.relu(x) 
        # x = self.fc1b(x)  # add another layer
        # x = torch.nn.functional.relu(x) 
        x = self.fc2(x)
        # print("output: ", x.shape)
        return x
    
class NeuralNetworkRegressor:
    def __init__(self, input_size=5, output_size=1, neurons=10, learning_rate=0.001, batch_size=32, epochs=100):
        self.norm_params = {}
        self.model = SimpleNN(input_size=input_size, output_size=output_size, neurons=neurons)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def normalize(self, dataframe, train=False):
        # Mean 0, Standard Deviation 1 Normalization
        for column in dataframe.columns:
            mean = dataframe[column].mean() if train else self.norm_params[column][0]
            std = dataframe[column].std() if train else self.norm_params[column][1]
            standardized_data = (dataframe[column] - mean) / std
            if train:
                self.norm_params[column] = (mean, std)
            dataframe[column] = standardized_data
        return dataframe

        # Min-Max Normalization
        # for column in dataframe.columns:
        #     min_value = dataframe[column].min() if train else self.norm_params[column][0]
        #     max_value = dataframe[column].max() if train else self.norm_params[column][1]
        #     if train:
        #         self.norm_params[column] = (min_value, max_value)
        #     dataframe[column] = (dataframe[column] - min_value) / (max_value - min_value)
        return dataframe
    
    def denormalize(self, dataframe):
        for column in dataframe.columns:
             # Retrieve normalization parameters
            mean, std = self.norm_params[column] 

            # Mean 0, Standard Deviation 1 De-normalization
            dataframe[column] = dataframe[column] * std + mean
        return dataframe
    
        # Min-Max De-normalization
        # for column in dataframe.columns:
        #     min_value, max_value = self.norm_params[column]
        #     dataframe[column] = dataframe[column] * (max_value - min_value) + min_value
        return dataframe

    def train(self, dataframe):

        dataframe = self.normalize(dataframe.copy(), train=True)
        dataset = CustomDataset(dataframe)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0

            for data, target in train_loader:

                data, target = data.to(self.device), target.to(self.device)
                # print(data.shape, target.shape)
                
                self.model.zero_grad()
                output = self.model(data)
                loss = self.loss_function(output, target.view(-1, 1))
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            print(f"Epoch: {epoch + 1}, Loss: {average_loss:.4f}")

    def predict(self, dataframe):
        # assume the last column in the dataframe is the target initialized to 0 and will be replaced by the prediction
        dataframe = self.normalize(dataframe, train=False)
        dataset = CustomDataset(dataframe)

        self.model.eval()

        for i in range(len(dataset)):
            data, _ = dataset[i]
            data = data.to(self.device)
            output = self.model(data)
            dataframe.iloc[i, -1] = output.item()

        dataframe = self.denormalize(dataframe)
        return dataframe




class SimpleNNAutoencoder(torch.nn.Module):
    def __init__(self, input_size=3, neurons=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, neurons)
        self.fc2 = torch.nn.Linear(neurons, 1)
        self.fc3 = torch.nn.Linear(1, neurons)
        self.fc4 = torch.nn.Linear(neurons, input_size)

    def forward(self, x):
        # print("input: ", x.shape)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        compressed_output = self.fc2(x)
        x = torch.nn.functional.relu(compressed_output)
        x = self.fc3(x)
        x = torch.nn.functional.relu(x)
        x = self.fc4(x)
        # print("output: ", x.shape)
        return x, compressed_output
    
class NeuralNetworkRegressorAutoencoder:
    def __init__(self, input_size=5, neurons=10, learning_rate=0.001, batch_size=32, epochs=100):
        self.norm_params = {}
        self.model = SimpleNNAutoencoder(input_size=input_size, neurons=neurons)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def normalize(self, dataframe, train=False):
        # Min-Max Normalization
        for column in dataframe.columns:
            min_value = dataframe[column].min() if train else self.norm_params[column][0]
            max_value = dataframe[column].max() if train else self.norm_params[column][1]
            if train:
                self.norm_params[column] = (min_value, max_value)
            dataframe[column] = (dataframe[column] - min_value) / (max_value - min_value)
        return dataframe
    
    def denormalize(self, dataframe):
        # Min-Max De-normalization
        for column in dataframe.columns:
            min_value, max_value = self.norm_params[column]
            dataframe[column] = dataframe[column] * (max_value - min_value) + min_value
        return dataframe

    def train(self, dataframe):

        dataframe = self.normalize(dataframe.copy(), train=True)
        dataset = CustomDatasetAutoencoder(dataframe)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0

            for data, target in train_loader:

                data, target = data.to(self.device), target.to(self.device)
                # print(data.shape, target.shape)
                
                self.model.zero_grad()
                output, compressed_output = self.model(data) # only the final output is used for training
                loss = self.loss_function(output, target)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            print(f"Epoch: {epoch + 1}, Loss: {average_loss:.4f}")

    def predict(self, dataframe):
        # add a column to the dataframe to store the autoencoder output. Only output the autoencoder compressed output

        dataframeOutput = dataframe.copy()

        dataframe = self.normalize(dataframe, train=False)
        dataset = CustomDatasetAutoencoder(dataframe)

        self.model.eval()

        for i in range(len(dataset)):
            data, _ = dataset[i]
            data = data.to(self.device)
            output, compressed_output = self.model(data)

            # Save compressed output to return
            # TODO: Remove hardcoded value
            dataframeOutput.at[i, 'autoencoder_output'] = compressed_output.item()

            # Save the reconstructed output of each column as well to return
            for j, column in enumerate(dataframe.columns):
                dataframeOutput.at[i, column] = output[j].item() 

        # Denormalize the dataframe input because it is passed by reference
        dataframe = self.denormalize(dataframe)

        # Denormalize only the re-constructed columns in the prediction dataframe
        columns = dataframe.columns
        dataframeOutput[columns] = self.denormalize(dataframeOutput[columns])

        return dataframeOutput