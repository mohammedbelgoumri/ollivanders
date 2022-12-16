import numpy as np
class Layer:
    """
    Base class for all layers
    """

    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, x):
        """
        ## Forward pass
        + Computes the input of the next layer given x
        """
        raise NotImplementedError

    def backward(self, dy, lr):
        """
        ## Backward pass
        + Updates the layer parameters
        + Computes the gradient of the previous layer (dx) given the current gradient (dy)
        """
        raise NotImplementedError

    def __call__(self, x) -> np.ndarray:
        return self.forward(x)
