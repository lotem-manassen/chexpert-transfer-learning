import pandas as pd
import os

import logging
from typing import Optional
import tensorflow as tf


# Base directory of the chexpert data:
DATA_DIR = '/home/lotem.m/'

# Classes:
CLASSES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def preprocess_train_csv() -> pd.DataFrame:
    """Preprocess the csv train csv file.

    Apply the following preprocessing steps:

    1. Drop lateral views. Validate that after the dropping, each patient holds a single frontal image.
    2. Treat blank labels as negatives, assign them with 0.
    3. Add images full paths to Path column.

    :return: Pandas data-frame with preprocessed data.
    """
    train_csv_path = os.path.join(DATA_DIR, 'CheXpert-v1.0-small', 'train.csv')
    train_df = pd.read_csv(train_csv_path)
    train_df['patient_id'] = train_df['Path'].apply(lambda x: x.split('/')[2])
    logging.info("Total number of images before preprocessing: %d", len(train_df))
    logging.info("Total number of patients before preprocessing: %d", len(train_df['patient_id'].unique()))

    # drop lateral views
    train_df = train_df[train_df['Frontal/Lateral'] != 'Lateral']
    assert train_df['Frontal/Lateral'].unique() == ['Frontal']
    # parse patient id from file path, and add as another column:
    train_df['patient_id'] = train_df['Path'].apply(lambda x: x.split('/')[2])
    logging.info("Final Total number of images before preprocessing: %d", len(train_df))
    logging.info("Final Number of unique patients in training-set: %d", len(train_df['patient_id'].unique()))

    train_df = train_df.fillna(0)
    for pathology in CLASSES:
        assert sorted(train_df[pathology].unique()) == [-1, 0, 1]

    train_df['Path'] = train_df['Path'].apply(lambda x: os.path.join(DATA_DIR, x))
    train_df['Sex'] = pd.Categorical(train_df['Sex'])
    train_df['Sex'] = train_df.Sex.cat.codes
    train_df = train_df.astype({'Path': 'str', 'Sex': 'category', 'Frontal/Lateral': 'category', 'AP/PA': 'category',
                                'patient_id': 'str'})

    return train_df


def create_train_tf_dataset(batch_size: int, shuffle: bool, repeat: bool, train_size: Optional[int] = None):
    """Create tf.dataset for the train-set.

    :param batch_size:
    :param shuffle:
    :param repeat:
    :param train_size:
    :return:
    """
    train_df = preprocess_train_csv()
    train_df = train_df.sample(frac=1, random_state=142)
    if train_size is not None:
        train_df = train_df[:train_size]

    train_ds = tf.data.Dataset.from_tensor_slices(train_df.to_dict('list'))

    def parse_image_labels(features_dict):

        image = tf.io.read_file(features_dict['Path'])
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [224, 224])

        label = tf.stack([features_dict[x] for x in CLASSES])
        features_dict['image'] = image
        features_dict['ground_truth'] = label
        return features_dict

    train_ds = train_ds.map(parse_image_labels)
    if repeat:
        train_ds = train_ds.repeat()
    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=500)

    train_ds = train_ds.batch(batch_size)
    return train_ds


def preprocess_validation_csv():
    val_csv_path = os.path.join(DATA_DIR, 'CheXpert-v1.0-small', 'valid.csv')
    val_df = pd.read_csv(val_csv_path)
    val_df['patient_id'] = val_df['Path'].apply(lambda x: x.split('/')[2])
    logging.info("Total number of images before preprocessing: %d", len(val_df))
    logging.info("Total number of patients before preprocessing: %d", len(val_df['patient_id'].unique()))

    # drop lateral views
    val_df = val_df[val_df['Frontal/Lateral'] != 'Lateral']
    assert val_df['Frontal/Lateral'].unique() == ['Frontal']

    # parse patient id from file path, and add as another column:
    val_df['patient_id'] = val_df['Path'].apply(lambda x: x.split('/')[2])
    logging.info("Final Total number of images: %d", len(val_df))
    logging.info("Final Number of unique patients in training-set: %d", len(val_df['patient_id'].unique()))

    for pathology in CLASSES:
        assert sorted(val_df[pathology].unique()) == [0, 1]

    val_df['Path'] = val_df['Path'].apply(lambda x: os.path.join(DATA_DIR, x))
    val_df['Sex'] = pd.Categorical(val_df['Sex'])
    val_df['Sex'] = val_df.Sex.cat.codes
    val_df = val_df.astype({'Path': 'str', 'Sex': 'category', 'Frontal/Lateral': 'category', 'AP/PA': 'category',
                                'patient_id': 'str'})

    return val_df


def create_validation_tf_dataset():
    """Create tf.dataset for the val-set."""
    val_df = preprocess_validation_csv()
    val_ds = tf.data.Dataset.from_tensor_slices(val_df.to_dict('list'))
    def parse_image_labels(features_dict):

        image = tf.io.read_file(features_dict['Path'])
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [224, 224])

        label = tf.stack([features_dict[x] for x in CLASSES])
        features_dict['image'] = image
        features_dict['ground_truth'] = label
        return features_dict
    val_ds = val_ds.map(parse_image_labels)
    val_ds = val_ds.batch(32)
    return val_ds


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        # Disable first GPU
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        # Logical device was not created for first GPU
        assert len(logical_devices) == len(physical_devices) - 1
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    # with tf.device('/device:GPU:2'):
    logging.basicConfig(level=logging.INFO)

    # train_df_processed = preprocess_train_csv()
    # print(train_df_processed.head())

    train_ds = create_train_tf_dataset(batch_size=1, repeat=False, shuffle=True)
    print()

        # preprocess_validation_csv()
        # create_validation_tf_dataset()