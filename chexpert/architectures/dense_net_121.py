"""DenseNet121."""
import tensorflow as tf
import os
import logging


def build_model(num_classes):
    # Create the base model from the pre-trained model MobileNet V2
    # inputs = tf.keras.Input(shape=(224, 224, 1))
    base_model = tf.keras.applications.DenseNet121(input_shape=(224, 224, 1),
                                                   include_top=False,
                                                   weights=None)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    predictions = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    return model


def build_from_gender_model(num_classes, gender_model_dir):
    logging.info("loading gender pre-trained DenseNet121 from %s...", gender_model_dir)
    base_model = tf.keras.models.load_model(os.path.join(gender_model_dir, 'saved_model', 'my_model'))
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    base_model.summary()
    logging.info("Removing prediction layer...")
    base_model = tf.keras.Model(base_model.input, base_model.layers[-2].output)
    base_model.summary()
    logging.info("Adding new prediction layer...")
    top_layer = tf.keras.layers.Dense(num_classes, name='predictions_new')

    inputs = tf.keras.Input(shape=(224, 224, 1))
    x = base_model(inputs, training=False)

    output_new = top_layer(x)
    base_model = tf.keras.Model(inputs, output_new)
    # base_model = tf.keras.Model(base_model.input, output_new)
    base_model.summary()
    for layer in base_model.layers:
        if layer.trainable:
            logging.info("Trainable layer: %s", layer)
    return base_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gender_model_dir = '/home/lotem.m/chexpert_exp/Gender_DenseNet121'
    build_from_gender_model(num_classes=5, gender_model_dir=gender_model_dir)


