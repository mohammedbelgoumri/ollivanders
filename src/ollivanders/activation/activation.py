from ollivanders.layer.layer import Layer


class Activation(Layer):
    """Base class for all activations
    """

    def __init__(self, f, df) -> None:
        super().__init__()
        self.activation = f
        self.d_activation = df

    def forward(self, x):
        self.input = x
        return self.activation(self.input)

    def backward(self, dy, lr):
        return dy * self.d_activation(self.input)
