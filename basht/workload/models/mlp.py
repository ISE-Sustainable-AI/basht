import torch.nn as nn
import torch
from torch.optim import Adam
from basht.workload.models.model_interface import ObjModel


class MLP(nn.Module, ObjModel):

    name = "MLP"

    def __init__(
        self, input_size: int, output_size: int, hidden_layer_config: list = None,
            learning_rate: float = 1e-3, weight_decay: float = 1e-6):
        """
        A simple linear Feed Forward Network.
        Performs classification and thus uses cross entropy loss. Default values always have to exists, except
        for input and output sizes.
        Args:
            input_size ([type]): [description]
            hidden_layer_config ([type]): [description]
            output_size ([type]): [description]
            learning_rate ([type]): [description]
            weight_decay ([type]): [description]
        """
        super().__init__()
        if not hidden_layer_config:
            hidden_layer_config = [15]
        self.layers = self._construct_layer(
            input_size=input_size, hidden_layer_config=hidden_layer_config, output_size=output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
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
            x = layer(x)
        x = self.layers[-1](x)
        return x

    def loss_function(self, x, target):
        return self.criterion(x, target)

    def predict(self, x):
        # TODO make argmax predictor customizable
        logits = self(x)
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
        predictions = self.predict(x)
        return predictions.detach()
