"""Main module to run experiments."""
import os
import logging
import shutil
from chexpert import train
from chexpert import train_gender
from chexpert import train_age
from chexpert.architectures import lenet, dense_net_121
import tensorflow as tf


BASE_DIR = '/home/lotem.m/chexpert_exp'


def init_model_dir(model_name):
    model_dir = os.path.join(BASE_DIR, model_name)
    if os.path.isdir(model_dir):
        answer = input("model dir already exists. Override?\n [yes/no]")
        while answer != 'yes' and answer != 'no':
            answer = input("model dir already exists. Override?\n [yes/no]")
        if answer == 'yes':
            shutil.rmtree(model_dir)
        elif answer == 'no':
            logging.info("Exiting...")
            exit()
    os.makedirs(model_dir)
    return model_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    physical_devices = tf.config.list_physical_devices('GPU')
    # try:
        # Disable first GPU
    tf.config.set_visible_devices(physical_devices[1], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
        # Logical device was not created for first GPU
        # assert len(logical_devices) == len(physical_devices) - 1
    # except:
        # Invalid device or cannot modify virtual devices once initialized.
        # pass
    #
    # Train Baseline Model with different train sizes:
    #

    # for train_size in [101, 1001, 5001, 10001, 50001, 100001]:
    #     model_dir = init_model_dir(f'DenseNet121_{train_size}')
    #     # lenet_model = lenet.build_basic_conv_model(model_dir=model_dir, num_classes=5)
    #     model = dense_net_121.build_model(num_classes=5)
    #     train.train(model_dir=model_dir, num_epochs=10, model=model, batch_size=32, train_size=train_size)

    # model_dir = init_model_dir(f'DenseNet121_all_train_size')
    # model = dense_net_121.build_model(num_classes=5)
    # train.train(model_dir=model_dir, num_epochs=5, model=model, batch_size=32, train_size=None)

    #
    # Train Gender Model:
    #

    # model_dir = init_model_dir(f'Gender_DenseNet121')
    # model = dense_net_121.build_model(num_classes=1)
    # train_gender.train(model_dir=model_dir, num_epochs=3, model=model, batch_size=32, train_size=None)

    #
    # Train Age Model:
    #
    # model_dir = init_model_dir(f'Age_DenseNet121_L1_loss')
    # model = dense_net_121.build_model(num_classes=1)
    # train_age.train(model_dir=model_dir, num_epochs=3, model=model, batch_size=32, train_size=None)

    #
    # Transfer learning on Gender:
    #
    gender_model_dir = '/home/lotem.m/chexpert_exp/Gender_DenseNet121'
    for train_size in [101, 1001, 5001, 10001, 50001, 100001]:
        model_dir = init_model_dir(f'transfer_learning_gender_DenseNet121_50trainable_{train_size}')
        model = dense_net_121.build_from_gender_model(num_classes=5, gender_model_dir=gender_model_dir)
        train.train(model_dir=model_dir, num_epochs=999999, model=model, batch_size=32, train_size=train_size)

    #
    # Transfer learning on age:
    #
    # age_model_dir = '/home/lotem.m/chexpert_exp/Age_DenseNet121_L1_loss'
    # for train_size in [101, 1001, 5001, 10001, 50001, 100001]:
    #     model_dir = init_model_dir(f'transfer_learning_age_DenseNet121_20trainable_{train_size}')
    #     model = dense_net_121.build_from_gender_model(num_classes=5, gender_model_dir=age_model_dir)
    #     train.train(model_dir=model_dir, num_epochs=999999, model=model, batch_size=32, train_size=train_size)