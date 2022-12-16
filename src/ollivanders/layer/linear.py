from ollivanders.layer.layer import Layer
import numpy as np


class Linear(Layer):
    """A fully connected layer"""

    def __init__(self, x_shape, y_shape) -> None:
        super().__init__()
        weights = np.random.randn(*y_shape, *x_shape)
        biases = np.random.randn(*y_shape, 1)
        # parameter matrix
        self.matrix = np.concatenate([weights, biases], axis=1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = np.concatenate([x, [1]], axis=0)
        # print(f"{self.matrix.shape} × {self.input.shape}\n")
        return np.dot(self.matrix, self.input)

    def backward(self, dy, lr) -> np.ndarray:
        reshaped = np.reshape(self.input, (-1, 1))
        if len(dy.shape) == 1:
            dy = np.reshape(dy, (-1, 1))
        # print(f"{dy.shape} × {reshaped.T.shape}")
        dw = np.dot(dy, reshaped.T)
        # print(f"{self.matrix[:, :-1].T.shape} × {dy.shape}\n")
        self.matrix -= dw * lr
        return np.dot(self.matrix[:, :-1].T, dy)
