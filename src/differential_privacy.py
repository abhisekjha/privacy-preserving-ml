import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

def apply_differential_privacy(model, loss, learning_rate, l2_norm_clip, noise_multiplier, num_microbatches):
    """
    Apply differential privacy to the given model.
    """
    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def compute_privacy_epsilon(num_samples, batch_size, noise_multiplier, epochs):
    """
    Compute the epsilon value for differential privacy.
    """
    if noise_multiplier == 0:
        return float('inf')
    return compute_dp_sgd_privacy.compute_dp_sgd_privacy(num_samples, batch_size, noise_multiplier, epochs)[0]
