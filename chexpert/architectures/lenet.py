import tensorflow as tf
import os


def build_basic_conv_model(model_dir: str, num_classes: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(224, 224, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes),
    ])
    model.summary()
    dot_img_file = os.path.join(model_dir, 'model_viz.png')
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    return model
