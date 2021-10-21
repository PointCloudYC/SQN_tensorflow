import tensorflow as tf
import numpy as np
from tf_interpolate import three_nn, three_interpolate

class GroupPointTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with self.test_session():
      features = tf.constant(np.random.random((1,8,16)).astype('float32')) # features, (1,8,16)
      print(features)
      xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
      xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
      dist, idx = three_nn(xyz1, xyz2) # (1,128,3), (1,128,3)
      weight = tf.ones_like(dist)/3.0 # (1,128,3)
      interpolated_features = three_interpolate(features, idx, weight) # (1,128,16)
      print(interpolated_features)
      err = tf.test.compute_gradient_error(features, (1,8,16), interpolated_features, (1,128,16))
      print(err)
      self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main() 
