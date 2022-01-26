import uuid
import numpy as np
import tensorflow as tf
import valohai


def log_metadata(epoch, logs):
    """Helper function to log training metrics"""
    with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])


valohai.prepare(
    step='train-model',
    image='tensorflow/tensorflow:2.6.0',
    default_inputs={
        'dataset': 'https://valohaidemo.blob.core.windows.net/mnist/preprocessed_mnist.npz',
    },
    default_parameters={
        'learning_rate': 0.001,
        'epochs': 5,
    },
)

input_path = valohai.inputs('dataset').path()
with np.load(input_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=valohai.parameters('learning_rate').value)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy'])

callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
model.fit(x_train, y_train, epochs=valohai.parameters('epochs').value, callbacks=[callback])

test_loss, test_accuracy = model.evaluate(x_test,  y_test, verbose=2)

with valohai.logger() as logger:
    logger.log('test_accuracy', test_accuracy)
    logger.log('test_loss', test_loss)


suffix = uuid.uuid4()
output_path = valohai.outputs().path(f'model-{suffix}.h5')
model.save(output_path)
