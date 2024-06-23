import unittest
import tensorflow as tf
from src.differential_privacy import apply_differential_privacy, compute_privacy_epsilon

class TestDifferentialPrivacy(unittest.TestCase):
    def setUp(self):
        self.model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, activation='softmax')])
        self.learning_rate = 0.001
        self.l2_norm_clip = 1.0
        self.noise_multiplier = 1.1
        self.num_microbatches = 32
        self.loss = 'sparse_categorical_crossentropy'

    def test_apply_differential_privacy(self):
        model = apply_differential_privacy(self.model, self.loss, self.learning_rate, self.l2_norm_clip, self.noise_multiplier, self.num_microbatches)
        self.assertIsNotNone(model)

    def test_compute_privacy_epsilon(self):
        epsilon = compute_privacy_epsilon(60000, 32, self.noise_multiplier, 10)
        self.assertTrue(epsilon > 0)

if __name__ == '__main__':
    unittest.main()
