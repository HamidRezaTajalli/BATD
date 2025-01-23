import unittest
import torch
import numpy as np
from models.SAINT import SAINTModel  # Adjust the import path as necessary

class TestSAINTModelForward(unittest.TestCase):

    def setUp(self):
        # Mock data object with necessary attributes
        self.data_obj = type('DataObject', (object,), {
            'num_classes': 3,
            'feature_names': ['feature1', 'feature2', 'feature3'],
            'cat_cols': [0, 1],
            'cat_cols_idx': [0, 1],
            'num_cols_idx': [2],
            'FTT_n_categories': [3, 4]
        })()

        # Initialize the model
        self.model = SAINTModel(self.data_obj)

    def test_forward(self):
        # Create a mock input tensor with batch size 5 and 3 features
        X = torch.tensor([
            [1, 2, 0.5],
            [0, 1, 0.3],
            [2, 3, 0.7],
            [1, 0, 0.2],
            [0, 2, 0.6]
        ], dtype=torch.float32)

        # Run the forward method
        output = self.model.forward(X)

        # Check the output shape
        expected_shape = (5, self.data_obj.num_classes)  # Assuming output is logits for each class
        self.assertEqual(output.shape, expected_shape, "Output shape mismatch")

        # Check the output type
        self.assertIsInstance(output, torch.Tensor, "Output is not a torch.Tensor")

        # Check if the output contains finite values
        self.assertTrue(torch.isfinite(output).all(), "Output contains non-finite values")

if __name__ == '__main__':
    unittest.main()