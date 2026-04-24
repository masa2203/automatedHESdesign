from typing import List, Union

import torch
import torch.nn as nn


class ANN(nn.Module):
    """
    Implements an Artificial Neural Network (ANN) model for time-series forecasting.

    This model uses a fully connected feedforward architecture to process sequences of input data. The input
    data is flattened into a single vector before being passed through multiple dense layers with specified
    activation functions. The final layer produces the output predictions.

    This architecture is suitable for tasks that require a simple yet effective approach to mapping input
    features to output predictions.

    :param in_dim: The number of features in the input data.
    :param out_dim: The dimensionality of the output predictions.
    :param window_size: The size of the window to process the input data (number of time steps).
    :param activation: The activation function to use between layers of the ANN. Defaults to nn.ReLU().
    :param ann_net_shape: The structure of the ANN, defined as a tuple of integers where each integer represents
                          the number of neurons in a layer. This tuple does not include the output layer size.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            window_size: int,
            activation: nn.modules.activation = nn.ReLU(),
            ann_net_shape: tuple[int] = (32, 32),  # not including output size!
    ):
        """
        Initializes the ANN model with specified dimensions and layers.

        :param in_dim: The number of features in the input data.
        :param out_dim: The dimensionality of the output predictions.
        :param window_size: The size of the window to process the input data (number of time steps).
        :param activation: The activation function to use between layers of the ANN. Defaults to nn.ReLU().
        :param ann_net_shape: The structure of the ANN, defined as a tuple of integers where each integer represents
                              the number of neurons in a layer. This tuple does not include the output layer size.
        """
        super(ANN, self).__init__()

        # Compute the input size for the first layer (flattened input)
        in_features = in_dim * window_size

        # Define the ANN layers
        ann_layers = [nn.Linear(in_features=in_features, out_features=ann_net_shape[0]), activation]
        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            ann_layers.append(activation)

            # Add the output layer
        ann_layers.append(nn.Linear(in_features=ann_net_shape[-1], out_features=out_dim))

        # Wrap the layers in a sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the ANN model.

        The input data is first flattened into a single vector and then passed through the ANN layers. The final
        layer produces the output predictions.

        :param x: The input data tensor, expected to have a shape (batch, sequence, features).
        :return: The output predictions of the model, shaped according to the out_dim parameter. This output is
                 suitable for tasks that require simple input-output mapping, such as time-series forecasting.
        """
        # Flatten the input so it's [feature1_hour1, feature2_hour1, ..., feature1_hour2, feature2_hour2, ...]
        x = x.view(x.size(0), -1)

        # Pass through the ANN layers
        x = self.ann(x)

        return x


class LSTM(nn.Module):
    """
    Implements an LSTM-based model for time-series forecasting.

    This model integrates an LSTM layer for handling sequential data with a subsequent ANN for prediction,
    making it suitable for tasks that involve predicting future values in a time series based on past observations.

    :param in_dim: The number of features in the input data.
    :param out_dim: The dimensionality of the output predictions.
    :param activation: The activation function to use between layers of the ANN. Defaults to nn.ReLU().
    :param lstm_layer_size: The number of features in the hidden state h of each LSTM cell. Defaults to 32.
    :param lstm_num_layer: The number of layers in the LSTM. Defaults to 2.
    :param ann_net_shape: The structure of the ANN, defined as a list of integers where each integer
                          represents the number of neurons in a layer. This list does not include the output layer size.
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 activation: nn.modules.activation = nn.ReLU(),
                 lstm_layer_size: int = 32,
                 lstm_num_layer: int = 2,
                 ann_net_shape: List[int] = [32, 32],  # not including output size!
                 ):
        """
        Initializes the LSTM model with specified dimensions and layers.

        :param in_dim: The number of features in the input data.
        :param out_dim: The dimensionality of the output predictions.
        :param activation: The activation function to use between layers of the ANN. Defaults to nn.ReLU().
        :param lstm_layer_size: The number of features in the hidden state h of each LSTM cell. Defaults to 32.
        :param lstm_num_layer: The number of layers in the LSTM. Defaults to 2.
        :param ann_net_shape: The structure of the ANN, defined as a list of integers where each integer represents
            the number of neurons in a layer. This list does not include the output layer size.
        """
        super(LSTM, self).__init__()

        # LSTM with batch_first=True needs (batch, sequence, features)
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=lstm_layer_size,
                            num_layers=lstm_num_layer,
                            batch_first=True)

        # ANN
        ann_layers = [nn.Linear(in_features=lstm_layer_size,
                                out_features=ann_net_shape[0]),
                      activation]
        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            ann_layers.append(activation)

        # Output layer
        ann_layers.append(nn.Linear(in_features=ann_net_shape[-1], out_features=out_dim))

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, x):
        """
        Defines the forward pass of the LSTM model.

        :param x: The input data tensor, expected to have a shape (batch, sequence, features) due to batch_first=True
            configuration in LSTM layer.
        :return: The output predictions of the model, shaped according to the out_dim parameter.
        """
        x, _ = self.lstm(x)
        x = self.ann(x[:, -1, :])  # batch_first=True
        return x


class CNN(nn.Module):
    """
    Implements a Convolutional Neural Network (CNN) model for time-series forecasting.

    This model employs a 1D CNN to process sequences of data by capturing spatial dependencies within the input
    features. The output from the CNN layers is then passed through an Artificial Neural Network (ANN) to make
    the final forecast. This architecture is suitable for tasks that require the extraction of local and
    temporal features from the input data.
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 window_size: int,
                 activation: nn.modules.activation = nn.ReLU(),
                 cnn_net_shape: List[int] = [32],  # not including input dimension!
                 cnn_kernel_size: int = 3,
                 cnn_stride: int = 1,
                 ann_net_shape: List[int] = [32, 32],  # not including output size!
                 ):
        """
        Initializes the CNN model with specified dimensions, layers, and configurations.

        :param in_dim: The number of features in the input data.
        :param out_dim: The dimensionality of the output predictions.
        :param window_size: The size of the window to process the input data.
        :param activation: The activation function to use between layers of the CNN and ANN. Defaults to nn.ReLU().
        :param cnn_net_shape: The structure of the CNN, defined as a list of integers where each integer represents
                              the number of output channels for a convolutional layer. This list does not include the
                              input dimension.
        :param cnn_kernel_size: The size of the kernel to use in convolutional layers.
        :param cnn_stride: The stride of the convolutional layers.
        :param ann_net_shape: The structure of the ANN, defined as a list of integers where each integer represents
                              the number of neurons in a layer. This list does not include the output layer size.
        """
        super(CNN, self).__init__()

        # 1D CNN needs (batch, features, sequence)
        # CNN input layer
        cnn_layers = [nn.Conv1d(in_channels=in_dim,
                                out_channels=cnn_net_shape[0],
                                kernel_size=cnn_kernel_size,
                                stride=cnn_stride),
                      activation]
        for i in range(len(cnn_net_shape) - 1):
            cnn_layers.append(nn.Conv1d(in_channels=cnn_net_shape[i],
                                        out_channels=cnn_net_shape[i + 1],
                                        kernel_size=cnn_kernel_size,
                                        stride=cnn_stride))
            cnn_layers.append(activation)

        # Wrap with sequential module
        self.cnn = nn.Sequential(*cnn_layers)

        # ANN
        # Compute size of CNN output after flattening
        in_features = ((window_size - cnn_kernel_size) // cnn_stride) + 1
        for i in range(len(cnn_net_shape) - 1):
            in_features = ((in_features - cnn_kernel_size) // cnn_stride) + 1
        in_features *= cnn_net_shape[-1]

        # Define ANN Layers
        ann_layers = [nn.Linear(in_features=in_features,
                                out_features=ann_net_shape[0]),
                      activation]
        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            ann_layers.append(activation)

        # Output layer
        ann_layers.append(nn.Linear(in_features=ann_net_shape[-1], out_features=out_dim))

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, x):
        """
        Defines the forward pass of the CNN model.

        :param x: The input data tensor, expected to have a shape (batch, sequence, features).
        :return: The output predictions of the model, shaped according to the out_dim parameter.
        """
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.ann(x)
        return x


class CnnLstmHybrid(nn.Module):
    """
    Implements a hybrid model combining CNN and LSTM layers for time-series forecasting.

    This model uses a 1D Convolutional Neural Network (CNN) to process sequences of data by capturing spatial
    dependencies, followed by Long Short-Term Memory (LSTM) layers to capture temporal dependencies. The output
    from the LSTM layers is then passed through an Artificial Neural Network (ANN) to make the final forecast.
    This architecture is suitable for tasks that require the extraction of both local features (spatial) and
    long-term dependencies (temporal) from the input data.
    """
    def __init__(self,
                 in_dim: int = 2,
                 out_dim: int = 1,
                 activation: nn.modules.activation = nn.ReLU(),
                 cnn_net_shape: List[int] = [32, 32],  # not including input dimension!
                 cnn_kernel_size: int = 3,
                 cnn_stride: int = 1,
                 lstm_layer_size: int = 32,
                 lstm_num_layer: int = 2,
                 ann_net_shape: List[int] = [32, 32],  # not including output size!
                 ):
        """
        Initializes the CnnLstmHybrid model with specified configurations for the CNN, LSTM, and ANN layers.

        :param in_dim: The number of features in the input data for the CNN layer.
        :param out_dim: The dimensionality of the output predictions.
        :param activation: The activation function used between layers. Defaults to nn.ReLU().
        :param cnn_net_shape: The structure of the CNN, defined as a list of integers where each integer represents
                              the number of output channels for a convolutional layer.
        :param cnn_kernel_size: The size of the kernel for the convolutional layers.
        :param cnn_stride: The stride for the convolutional layers.
        :param lstm_layer_size: The number of features in the hidden state h of each LSTM cell.
        :param lstm_num_layer: The number of LSTM layers.
        :param ann_net_shape: The structure of the ANN, defined as a list of integers where each integer represents
                              the number of neurons in a layer.
        """
        super(CnnLstmHybrid, self).__init__()

        # 1D CNN needs (batch, features, sequence)
        # CNN input layer
        cnn_layers = [nn.Conv1d(in_channels=in_dim,
                                out_channels=cnn_net_shape[0],
                                kernel_size=cnn_kernel_size,
                                stride=cnn_stride),
                      activation]
        for i in range(len(cnn_net_shape) - 1):
            cnn_layers.append(nn.Conv1d(in_channels=cnn_net_shape[i],
                                        out_channels=cnn_net_shape[i + 1],
                                        kernel_size=cnn_kernel_size,
                                        stride=cnn_stride))
            cnn_layers.append(activation)

        # Wrap with sequential module
        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM with batch_first=True needs (batch, sequence, features)
        self.lstm = nn.LSTM(input_size=cnn_net_shape[-1],
                            hidden_size=lstm_layer_size,
                            num_layers=lstm_num_layer,
                            batch_first=True)

        # ANN
        ann_layers = [nn.Linear(in_features=lstm_layer_size,
                                out_features=ann_net_shape[0]),
                      activation]
        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            ann_layers.append(activation)

        # Output layer
        ann_layers.append(nn.Linear(in_features=ann_net_shape[-1], out_features=out_dim))

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, x):
        """
        Defines the forward pass of the CnnLstmHybrid model.

        The input data first goes through the CNN layers, where convolutional operations are applied to extract spatial
        features. The output of the CNN layers is then processed by the LSTM layers, which are designed to capture
        temporal dependencies in the sequence data. Finally, the output from the LSTM layers is passed through an
        ANN to produce the final predictions.

        :param x: The input data tensor, expected to have a shape (batch, sequence, features).
                  The data is first permuted to match the input requirements of the CNN layers and then reverted for
                  the LSTM layers.
        :return: The output predictions of the model, shaped according to the out_dim parameter. This output is suitable
                 for tasks that require both spatial and temporal feature extraction, such as time-series forecasting.
        """
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)  # '_' are the hidden states
        x = self.ann(x[:, -1, :])  # Pick last element of sequence
        return x


class CnnLstmAttentionHybrid(nn.Module):
    """
    Implements a hybrid model combining CNN, LSTM, and Multi-Head Attention layers for time-series forecasting.

    This model integrates a 1D Convolutional Neural Network (CNN) to process sequences of data by capturing spatial
    dependencies, followed by Long Short-Term Memory (LSTM) layers to capture temporal dependencies. On top of this,
    a Multi-Head Attention (MHA) mechanism is applied to enhance the model's ability to focus on different parts of
    the sequence for making predictions. The output from the MHA layer is then passed through an Artificial Neural
    Network (ANN) to make the final forecast.

    This architecture is suitable for complex tasks that require the extraction of both local features (spatial),
    long-term dependencies (temporal), and contextual relationships within the sequence.
    """
    def __init__(self,
                 in_dim: int = 2,
                 out_dim: int = 1,
                 activation: nn.modules.activation = nn.ReLU(),
                 cnn_net_shape: List[int] = [32, 32],  # not including input dimension!
                 cnn_kernel_size: int = 3,
                 cnn_stride: int = 1,
                 lstm_layer_size: int = 32,
                 lstm_num_layer: int = 2,
                 mha_num_heads: int = 4,
                 mha_dropout: float = 0.1,
                 ann_net_shape: List[int] = [32, 32],  # not including output size!
                 ):
        """
        Initializes the CnnLstmAttentionHybrid model with specified configurations for the CNN, LSTM, MHA, and ANN
        layers.

        :param in_dim: The number of features in the input data for the CNN layer.
        :param out_dim: The dimensionality of the output predictions.
        :param activation: The activation function used between layers. Defaults to nn.ReLU().
        :param cnn_net_shape: The structure of the CNN, defined as a list of integers where each integer represents
                              the number of output channels for a convolutional layer.
        :param cnn_kernel_size: The size of the kernel for the convolutional layers.
        :param cnn_stride: The stride for the convolutional layers.
        :param lstm_layer_size: The number of features in the hidden state h of each LSTM cell.
        :param lstm_num_layer: The number of LSTM layers.
        :param mha_num_heads: The number of attention heads in the Multi-Head Attention layer.
        :param mha_dropout: The dropout rate for the Multi-Head Attention layer.
        :param ann_net_shape: The structure of the ANN, defined as a list of integers where each integer represents
                              the number of neurons in a layer.
        """
        super(CnnLstmAttentionHybrid, self).__init__()

        # 1D CNN needs (batch, features, sequence)
        # CNN input layer
        cnn_layers = [nn.Conv1d(in_channels=in_dim,
                                out_channels=cnn_net_shape[0],
                                kernel_size=cnn_kernel_size,
                                stride=cnn_stride),
                      activation]
        for i in range(len(cnn_net_shape) - 1):
            cnn_layers.append(nn.Conv1d(in_channels=cnn_net_shape[i],
                                        out_channels=cnn_net_shape[i + 1],
                                        kernel_size=cnn_kernel_size,
                                        stride=cnn_stride))
            cnn_layers.append(activation)

        # Wrap with sequential module
        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM with batch_first=True needs (batch, sequence, features)
        self.lstm = nn.LSTM(input_size=cnn_net_shape[-1],
                            hidden_size=lstm_layer_size,
                            num_layers=lstm_num_layer,
                            batch_first=True)

        # Multi-head Attention
        self.attention = nn.MultiheadAttention(embed_dim=lstm_layer_size,
                                               num_heads=mha_num_heads,
                                               dropout=mha_dropout,
                                               batch_first=True)

        # ANN
        ann_layers = [nn.Linear(in_features=lstm_layer_size,
                                out_features=ann_net_shape[0]),
                      activation]
        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            ann_layers.append(activation)

        # Output layer
        ann_layers.append(nn.Linear(in_features=ann_net_shape[-1], out_features=out_dim))

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, x):
        """
        Defines the forward pass of the CnnLstmAttentionHybrid model.

        The input data is first processed by the CNN layers, extracting spatial features. The processed data is then
        fed into the LSTM layers, capturing temporal dependencies. Following this, the Multi-Head Attention mechanism
        is applied, allowing the model to focus on different parts of the sequence. Finally, the output is passed
        through an ANN to produce the final predictions.

        :param x: The input data tensor, expected to have a shape (batch, sequence, features).
                  The data is processed through various layers, with permutations as required for CNN and LSTM layers.
        :return: The output predictions of the model, shaped according to the out_dim parameter. This output is
                 designed for tasks that require both spatial and temporal feature extraction, such as time-series
                 forecasting.
        """
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x)
        x = self.ann(x[:, -1, :])
        return x


model_dict = {
    'CNN': CNN,
    'LSTM': LSTM,
    'Hybrid': CnnLstmHybrid,
    'AttentionHybrid': CnnLstmAttentionHybrid,
}

nets = {
    'tiny1': [16],
    'small1': [32],
    'medium1': [64],
    'large1': [128],
    'tiny2': [16, 16],
    'small2': [32, 32],
    'medium2': [64, 64],
    'large2': [128, 128],
}
