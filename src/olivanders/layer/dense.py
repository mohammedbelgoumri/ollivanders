from olivanders.layer.layer import Layer
import numpy as np


class Dense(Layer):
    """A fully connected layer"""

    def __init__(self, x_shape, y_shape) -> None:
        super().__init__()
        weights = np.random.randn(*x_shape, *y_shape)
        biases = np.random.randn(1, *y_shape)
        # parameter matrix
        self.matrix = np.concatenate([weights, biases], axis=0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = np.concatenate([x, [1]], axis=0)
        return np.dot(self.input, self.matrix)

    def backward(self, dy, lr) -> np.ndarray:
        print(self.input.T.shape)
        dw = np.dot(dy, self.input.T)
        self.matrix -= dw * lr
        return np.dot(self.matrix[:-1, :].T, dy)
