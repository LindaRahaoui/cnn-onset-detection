import torch.nn as nn
import torch
import torch.nn.functional as F
import madmom

class MadmomBatchNormLayer(nn.Module):
    def __init__(self, madmom_layer):
        super(MadmomBatchNormLayer, self).__init__()
        num_features = madmom_layer.inv_std.shape[1]
        self.bn = nn.BatchNorm2d(num_features)
        self.bn.weight.data = torch.tensor(madmom_layer.inv_std[0], dtype=torch.float32)
        self.bn.bias.data = torch.tensor(madmom_layer.mean[0], dtype=torch.float32)

    def forward(self, x):
        return self.bn(x)

class MadmomConvolutionalLayer(nn.Module):
    def __init__(self, madmom_layer):
        super(MadmomConvolutionalLayer, self).__init__()
        in_channels = madmom_layer.weights.shape[1]
        out_channels = madmom_layer.weights.shape[0]
        kernel_size = (madmom_layer.weights.shape[2], madmom_layer.weights.shape[3])
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=madmom_layer.stride,
                              padding=(0 if madmom_layer.pad == 'valid' else kernel_size[0] // 2))
        self.conv.weight.data = torch.tensor(madmom_layer.weights, dtype=torch.float32)
        self.conv.bias.data = torch.tensor(madmom_layer.bias, dtype=torch.float32)
        self.activation = madmom_layer.activation_fn

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            if self.activation == 'tanh':
                x = torch.tanh(x)
            elif self.activation == 'relu':
                x = torch.relu(x)
            # Ajoutez d'autres fonctions d'activation si nécessaire
        return x


class MadmomMaxPoolLayer(nn.Module):
    def __init__(self, madmom_layer):
        super(MadmomMaxPoolLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=tuple(madmom_layer.size), 
                                 stride=tuple(madmom_layer.stride))

    def forward(self, x):
        return self.pool(x)

class MadmomStrideLayer(nn.Module):
    def __init__(self, madmom_layer):
        super(MadmomStrideLayer, self).__init__()
        self.block_size = madmom_layer.block_size

    def forward(self, x):
        return x[:, :, ::self.block_size, ::self.block_size]


class MadmomFeedForwardLayer(nn.Module):
    def __init__(self, madmom_layer):
        super(MadmomFeedForwardLayer, self).__init__()
        self.fc = nn.Linear(madmom_layer.weights.shape[0], 
                            madmom_layer.weights.shape[1])
        self.fc.weight.data = torch.tensor(madmom_layer.weights, dtype=torch.float32)
        self.fc.bias.data = torch.tensor(madmom_layer.bias, dtype=torch.float32)
        self.activation = madmom_layer.activation_fn

    def forward(self, x):
        x = self.fc(x)
        if self.activation is not None:
            if self.activation == 'tanh':
                x = torch.tanh(x)
            elif self.activation == 'relu':
                x = torch.relu(x)
            # Ajoutez d'autres fonctions d'activation si nécessaire
        return x

class MadmomToTorchNN(nn.Module):
    def __init__(self, madmom_model):
        super(MadmomToTorchNN, self).__init__()
        self.layers = nn.ModuleList()
        for layer in madmom_model.layers:
            print(f"Converting layer: {type(layer)} with attributes {layer.__dict__}")
            self.layers.append(self.convert_layer(layer))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def convert_layer(self, madmom_layer):
        if isinstance(madmom_layer, madmom.ml.nn.layers.BatchNormLayer):
            return MadmomBatchNormLayer(madmom_layer)
        elif isinstance(madmom_layer, madmom.ml.nn.layers.ConvolutionalLayer):
            return MadmomConvolutionalLayer(madmom_layer)
        elif isinstance(madmom_layer, madmom.ml.nn.layers.MaxPoolLayer):
            return MadmomMaxPoolLayer(madmom_layer)
        elif isinstance(madmom_layer, madmom.ml.nn.layers.StrideLayer):
            return MadmomStrideLayer(madmom_layer)
        elif isinstance(madmom_layer, madmom.ml.nn.layers.FeedForwardLayer):
            return MadmomFeedForwardLayer(madmom_layer)
        else:
            raise ValueError(f"Unsupported layer type: {type(madmom_layer)}")