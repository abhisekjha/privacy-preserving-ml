import unittest
from src.federated_learning import create_federated_data, build_federated_model

class TestFederatedLearning(unittest.TestCase):
    def test_create_federated_data(self):
        # Mock data and client IDs for testing
        data = tf.data.Dataset.from_tensor_slices({'x': tf.random.uniform([100, 28, 28, 1]), 'y': tf.random.uniform([100], maxval=10, dtype=tf.int32)})
        client_ids = ['client_1', 'client_2']
        federated_data = create_federated_data(data, client_ids, preprocess_fn)
        self.assertEqual(len(federated_data), len(client_ids))

    def test_build_federated_model(self):
        model = build_federated_model()
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
