from basht.workload.models.model_interface import ObjModel
import torch.nn as nn
from torch.optim import Adam


class CNN(nn.Module, ObjModel):

    def __init__(
        self, input_size: int, output_size: int, hidden_layer_config: list = None,
            learning_rate: float = 1e-3, weight_decay: float = 1e-6):
        super().__init__()
        """Implementation of LeNet-1, according to:
        LeNet-5: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf (required no subsampling of the data)
        """

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

    def forward(self):
        # Conv
        #Samp
        #Conv
        #Samp
        #Linear
        # Linear
        # Linear + softmax
        # LOSS: cross entropy
        pass
