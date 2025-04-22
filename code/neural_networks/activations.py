"""
Author: Sophia Sanborn
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas
"""

import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict



class Activation(ABC):
    """Abstract class defining the common interface for all activation methods."""

    def __call__(self, Z):
        return self.forward(Z)

    @abstractmethod
    def forward(self, Z):
        pass


def initialize_activation(name: str) -> Activation:
    """Factory method to return an Activation object of the specified type."""
    if name == "linear":
        return Linear()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return TanH()
    elif name == "arctan":
        return ArcTan()
    elif name == "relu":
        return ReLU()
    elif name == "softmax":
        return SoftMax()
    else:
        raise NotImplementedError("{} activation is not implemented".format(name))


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = z.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = z.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        return dY


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function:
        f(z) = 1 / (1 + exp(-z))
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        return ...

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        return ...


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = tanh(z).
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return 2 / (1 + np.exp(-2 * Z)) - 1

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = tanh(z).
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        fn = self.forward(Z)
        return dY * (1 - fn ** 2)


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(z) = z if z >= 0
               0 otherwise
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        def relu(elem):
            if elem >=0:
                return elem
            else:
                return 0
        f = np.vectorize(relu)  
                  
        ### YOUR CODE HERE ###
        return np.maximum(Z, np.zeros(Z.shape))

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        
        
        grad = np.zeros((Z.shape[0], Z.shape[1]))
        #index all the lemets when a cinditon is true
        # you cna specify a condition for array Z and assign that to a scalar value
        grad = (Z >= 0).astype(int) * dY
      

        # for i, row in enumerate(Z):
        #     for j, col in enumerate(Z[i]):
        #         dZi = 0
        #         if Z[i][j] >= 0:
        #             dZi = 1
        #         dYi = dY[i][j]
        #         grad[i][j] =  dYi * dZi

        return grad


class SoftMax(Activation):
    def __init__(self):
        super().__init__()
        self.cache = OrderedDict()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for softmax activation.
        Hint: The naive implementation might not be numerically stable.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        fz = np.zeros(Z.shape)
        
        for i in range(len(Z)):
            #each elem is row vector a with elements from 1 through k

            row = Z[i]
            row_max = max(row)
            exponential_row_sum = np.sum(np.exp(row-row_max))
            fz[i] = np.exp(row-row_max) / exponential_row_sum
        #self.cache["fizz"] = fz
        return fz

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for softmax activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
       
        ### YOUR CODE HERE ###
        # iterate over rows and consruct matrix based on jacobian of each row

        
        gradient = np.zeros(dY.shape)
        #for each jacobian we do a dot product
        yhat = self.forward(Z)
        #yhat = self.cache["fizz"]
        for i in range(len(Z)):
            yi = yhat[i]           
            dYda = np.outer(-np.transpose(yi), yi)
            dYda = dYda + np.diag(yi)
            
            
            
            
            #off diagnoal of dyda is y tranpose times y
            # for j in range(len(row)):
            #     if i==j:
            #         yhat = np.exp(row[j]) / np.sum(np.exp(row))
            #         dYdA_point = yhat * (1-yhat)
            #     else:
            #         yjhat = np.exp(row[j]) / np.sum(np.exp(row))
            #         try:
            #             yihat = np.exp(row[i]) / np.sum(np.exp(row))
            #         except IndexError:
            #             yihat = 0
            #         dYdA_point = yihat * (-yjhat)
            #     dYdA_row[j] = dYdA_point
            

            #get dL/dY at row i
            curr_dY = dY[i]
            #print(dYdA.shape)
            #print(curr_dY.shape)
            dLda = np.matmul(curr_dY, dYda)
            gradient[i] = dLda

        return gradient


class ArcTan(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        return np.arctan(Z)

    def backward(self, Z, dY):
        return dY * 1 / (Z ** 2 + 1)
