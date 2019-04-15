import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.ops import variables
from tensorflow import fft, ifft


class LSSGD(optimizer.Optimizer):
  """Optimizer that implements the gradient descent algorithm.
  """

  def __init__(self, learning_rate, sigma, use_locking=False, name="GradientDescent"):
    """Construct a new gradient descent optimizer.
    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(LSSGD, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._sigma = sigma
    var_info_dict = {}
    # precompute the FFT coefficients of the Laplacian. 
    for var in variables.trainable_variables():
      N = 1
      for sz in var.shape.as_list(): 
        N *= sz

      c = np.zeros(shape=N)
      c[0] = -2.
      c[1] = 1. 
      c[N - 1] = 1.
      c_fft = np.fft.fft(c) 
      coef = 1. / (1. - sigma * c_fft)
      coef = tf.constant(coef, dtype=tf.complex64)
      var_info_dict[var.name] = (N, coef)
      self._var_info_dict = var_info_dict

  def _fft_solver(self, grad, varname):
    grad_shape = grad.shape.as_list() 
    N, coef = self._var_info_dict[varname]
    grad = tf.reshape(grad, shape=[N])
    grad = tf.cast(grad, tf.complex64)
    # import ipdb; ipdb.set_trace()
    grad = tf.real(ifft(fft(grad) * coef))
    grad = tf.reshape(grad, shape=grad_shape)
    return grad

  def _apply_dense(self, grad, var):
    grad = self._fft_solver(grad, var.name)
    return training_ops.apply_gradient_descent(
        var,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, handle):
    raise NotImplementedError

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    raise NotImplementedError

  def _apply_sparse_duplicate_indices(self, grad, var):
    raise NotImplementedError

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
