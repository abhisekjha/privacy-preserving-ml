import tensorflow as tf
import tensorflow_federated as tff

def create_federated_data(data, client_ids, preprocess_fn):
    """
    Create federated data from a central dataset.
    """
    return [preprocess_fn(data.create_tf_dataset_for_client(x)) for x in client_ids]

def preprocess_fn(dataset):
    """
    Preprocess the dataset.
    """
    def batch_format_fn(element):
        return collections.OrderedDict(
            x=tf.reshape(element['x'], [-1, 28, 28, 1]),
            y=tf.reshape(element['y'], [-1, 1])
        )
    return dataset.repeat(10).shuffle(100).batch(20).map(batch_format_fn).prefetch(10)

def build_federated_model():
    """
    Build and compile a federated learning model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return tff.learning.from_keras_model(
        model,
        input_spec=preprocess_fn.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
