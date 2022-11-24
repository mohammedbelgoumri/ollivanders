from olivanders.activations import Activation
import numpy as np


class Tanh(Activation):
    """Hyperbolic tangent activation
    """

    def __init__(self) -> None:
        super().__init__(
            f=np.tanh,
            df=lambda x: 1 - np.tanh(x) ** 2
        )
