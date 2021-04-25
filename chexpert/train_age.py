"""Train network to classify the age."""
import os
import tensorflow as tf
import numpy as np
from chexpert import dataset
from chexpert import utils


def create_per_class_metrics():
    pre_class_metrics = {}
    class_name = 'age'
    pre_class_metrics[class_name] = {}
    pre_class_metrics[class_name]['epoch_loss_avg'] = tf.keras.metrics.Mean()
    pre_class_metrics[class_name]['epoch_accuracy'] = tf.keras.metrics.BinaryAccuracy()
    pre_class_metrics[class_name]['tp_metrics'] = tf.keras.metrics.TruePositives(name=f'tp_{class_name}')
    pre_class_metrics[class_name]['fp_metrics'] = tf.keras.metrics.FalsePositives(name=f'fp_{class_name}')
    pre_class_metrics[class_name]['tn_metrics'] = tf.keras.metrics.TrueNegatives(name=f'tn_{class_name}')
    pre_class_metrics[class_name]['fn_metrics'] = tf.keras.metrics.FalseNegatives(name=f'fn_{class_name}')
    pre_class_metrics[class_name]['precision_metrics'] = tf.keras.metrics.Precision(
        name=f'precision_{class_name}')
    pre_class_metrics[class_name]['recall_metrics'] = tf.keras.metrics.Recall(name=f'recall_{class_name}')
    pre_class_metrics[class_name]['auc'] = tf.keras.metrics.AUC(name=f'auc_{class_name}')
    return pre_class_metrics


def update_per_class_metrics(pre_class_metrics, labels_batch, probs, logits, loss_fn):
    class_loss = loss_fn(labels_batch, logits)
    class_name = 'age'
    pre_class_metrics[class_name]['epoch_loss_avg'].update_state(class_loss)
    pre_class_metrics[class_name]['epoch_accuracy'].update_state(labels_batch, probs)
    pre_class_metrics[class_name]['tp_metrics'].update_state(labels_batch, probs)
    pre_class_metrics[class_name]['fp_metrics'].update_state(labels_batch, probs)
    pre_class_metrics[class_name]['tn_metrics'].update_state(labels_batch, probs)
    pre_class_metrics[class_name]['fn_metrics'].update_state(labels_batch, probs)
    pre_class_metrics[class_name]['precision_metrics'].update_state(labels_batch, probs)
    pre_class_metrics[class_name]['recall_metrics'].update_state(labels_batch, probs)
    pre_class_metrics[class_name]['auc'].update_state(labels_batch, probs)


def train(model_dir, num_epochs, model, batch_size, train_size):
    train_summary_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'test'))
    #
    # Create Datasets:
    #
    train_ds = dataset.create_train_tf_dataset(batch_size=batch_size, shuffle=True, repeat=False,
                                               train_size=train_size)
    if train_size is None:
        train_size = len(train_ds) * batch_size
    # assert len(train_ds) * batch_size == train_size
    val_ds = dataset.create_validation_tf_dataset()

    #
    # Define optimizer:
    #
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    global_step = 0
    for epoch in range(num_epochs):
        #
        # Tensorboard summaries:
        #
        epoch_loss_avg = tf.keras.metrics.Mean()
        # epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
        # pre_class_metrics = create_per_class_metrics()

        #
        # Start training loop:
        #
        for i, batch_data in enumerate(train_ds):
            images_batch, labels_batch = batch_data['image'], batch_data[
                'Age']  # Labels shape: [batch_size, 1], [1, 0, 0, 1]
            global_step += 1
            if global_step > 18000:
                break
            #
            # Train Step:
            #
            with tf.GradientTape() as tape:
                logits = model(images_batch, training=True)
                loss_value = loss_fn(labels_batch, logits)
                grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            logits = model(images_batch, training=True)
            # probs = tf.sigmoid(logits)

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # epoch_accuracy.update_state(labels_batch, probs)
            # update_per_class_metrics(pre_class_metrics, labels_batch, probs, logits, loss_fn)

            if global_step % 10 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('batch_loss', epoch_loss_avg.result(), step=global_step)
                    # tf.summary.scalar('accuracy', epoch_accuracy.result(), step=global_step)
                    # class_name = 'gender'
                    # for metric_name, metric in pre_class_metrics[class_name].items():
                    #     tf.summary.scalar(f'{class_name}_{metric_name}', metric.result(), step=global_step)

            if global_step % 50 == 0:
                print(f"Epoch {epoch}: Iteration: {i}, Loss: {epoch_loss_avg.result()}")

            if global_step % 100 == 0:
                #
                # Evaluate on validation set:
                #
                val_epoch_loss_avg = tf.keras.metrics.Mean()
                # val_pre_class_metrics = create_per_class_metrics()
                # val_epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

                # ground_truths = []
                # predictions = []
                for batch_data in val_ds:
                    images_batch, labels_batch = batch_data['image'], batch_data[
                        'Age']
                    logits = model(images_batch, training=False)
                    # probs = tf.sigmoid(logits)  # [batch_size, num_classes]
                    loss_value = loss_fn(labels_batch, logits)
                    val_epoch_loss_avg.update_state(loss_value)
                    # val_epoch_accuracy.update_state(labels_batch, probs)
                    # update_per_class_metrics(val_pre_class_metrics, labels_batch, probs, logits, loss_fn)

                    # ground_truths += labels_batch.numpy().tolist()
                    # probs_numpy = probs.numpy()
                    # probs_numpy = [x[0] for x in probs_numpy]
                    # predictions += probs_numpy

                print(f"Validation-Set: Epoch [{epoch + 1}] Iteration: [{global_step + 1}], "
                      f"Loss: {val_epoch_loss_avg.result():.3f}")
                with test_summary_writer.as_default():
                    # tf.summary.scalar('epoch_accuracy', val_epoch_accuracy.result(), step=global_step)
                    tf.summary.scalar('epoch_loss', val_epoch_loss_avg.result(), step=global_step)
                    # class_name = 'gender'
                    # for metric_name, metric in val_pre_class_metrics[class_name].items():
                    #     tf.summary.scalar(f'{class_name}_{metric_name}', metric.result(), step=global_step)

                # class_name = 'gneder'
                # per_class_gt = ground_truths
                # per_class_preds = predictions
                # roc_auc = utils.plot_roc(class_name, per_class_gt, per_class_preds,
                #                          train_size,
                #                          test_summary_writer, global_step)
                #
                # precision_recall_auc = utils.plot_precision_recall(class_name, per_class_gt, per_class_preds,
                #                                                    train_size,
                #                                                    test_summary_writer, global_step)
                # with test_summary_writer.as_default():
                #     tf.summary.scalar(f'{class_name}_epoch_roc_auc', roc_auc, step=global_step)
                #     tf.summary.scalar(f'{class_name}_epoch_precision_recall_auc', precision_recall_auc,
                #                       step=global_step)
        #
        # End of Epoch:
        #
        epoch_loss = epoch_loss_avg.result()
        # epoch_accuracy = epoch_accuracy.result()
        # print(f"End of Epoch {epoch + 1}:, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}")
        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', epoch_loss, step=global_step)
            # tf.summary.scalar('epoch_accuracy', epoch_accuracy, step=global_step)
            # class_name = 'gender'
            # for metric_name, metric in pre_class_metrics[class_name].items():
            #     tf.summary.scalar(f'epoch_{class_name}_{metric_name}', metric.result(), step=epoch)
        if global_step > 18000:
            break

    if not os.path.isdir(os.path.join(model_dir, 'saved_model')):
        os.mkdir(os.path.join(model_dir, 'saved_model'))
    model.save(os.path.join(model_dir, 'saved_model', 'my_model'))

    # return roc_auc, precision_recall_auc
