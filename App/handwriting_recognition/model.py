"""
model.py — Handwriting Recognition Model Architecture

Defines a hybrid ResNet-BiLSTM model for sequence recognition:
  - Input: (height, width, channels) image
  - Backbone: 9 residual blocks (CNN) that extract visual features
  - Decoder: Bidirectional LSTM + Dense softmax for character probabilities
  - Loss: CTC (Connectionist Temporal Classification) — applied during training

The model outputs a probability distribution over (vocab_size + 1) classes
at each time step, where the extra class is the CTC blank token.
"""

from keras import layers
from keras.models import Model

from mltu.tensorflow.model_utils import residual_block


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    """Build and return the ResNet-BiLSTM model.

    Args:
        input_dim (tuple): Input shape (height, width, channels), e.g. (32, 128, 3).
        output_dim (int): Number of unique characters in the vocabulary.
        activation (str): Activation function for residual blocks. Default: 'leaky_relu'.
        dropout (float): Dropout rate applied after each residual block and BiLSTM. Default: 0.2.

    Returns:
        keras.Model: Compiled-ready model with input shape input_dim
                     and output shape (time_steps, output_dim + 1).
    """
    inputs = layers.Input(shape=input_dim, name="input")

    # normalize images here instead in preprocessing step
    input = layers.Lambda(lambda x: x / 255)(inputs)

    x1 = residual_block(input, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x2 = residual_block(x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = residual_block(x5, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = residual_block(x6, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x8 = residual_block(x7, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x9 = residual_block(x8, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model
