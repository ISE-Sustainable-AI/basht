import torch.nn as nn
import torch
from torch.optim import Adam


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_config, output_size, learning_rate, weight_decay):
        """
        A simple Feed Forward Network with a Sigmoid Activation Function after the final layer.
        Performs binary classification and thus uses binary cross entropy loss.
        Args:
            input_size ([type]): [description]
            hidden_layer_config ([type]): [description]
            output_size ([type]): [description]
            learning_rate ([type]): [description]
            weight_decay ([type]): [description]
        """
        super().__init__()
        self.layers = self._construct_layer(
            input_size=input_size, hidden_layer_config=hidden_layer_config, output_size=output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU()
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _construct_layer(self, input_size, hidden_layer_config, output_size):
        """
        Generic Layer construction.
        Args:
            input_size ([type]): [description]
            hidden_layer_config ([type]): [description]
            output_size ([type]): [description]
        Returns:
            [type]: [description]
        """
        layers = nn.ModuleList([])
        for hidden_layer_size in hidden_layer_config:
            layers.append(nn.Linear(input_size, hidden_layer_size))
            input_size = hidden_layer_size
        layers.append(nn.Linear(input_size, output_size))
        return layers

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def loss_function(self, x, target):
        return self.criterion(x, target)

    def predict(self, logits):
        probabilities = self.softmax(logits)
        predictions = probabilities.to(dtype=torch.int16).argmax(dim=1)
        return predictions

    def train_step(self, x, y):
        logits = self(x)
        loss = self.loss_function(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_step(self, x):
        logits = self(x)
        predictions = self.predict(logits)
        return predictions
