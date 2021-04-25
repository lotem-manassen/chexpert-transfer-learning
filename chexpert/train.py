"""Training module."""
import os
import tensorflow as tf
import numpy as np
from chexpert import dataset
from chexpert import utils


def create_per_class_metrics():
    pre_class_metrics = {}
    for class_name in dataset.CLASSES:
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


def update_per_class_metrics(pre_class_metrics, labels_batch, probs, logits, loss_fn, mask=None):
    for class_ind, class_name in enumerate(dataset.CLASSES):
        class_labels = labels_batch[:, class_ind]
        class_predictions = probs[:, class_ind]
        if mask is not None:
            sample_weight = mask[:, class_ind]
        else:
            sample_weight = None
        class_loss = loss_fn(labels_batch[:, class_ind], logits[:, class_ind], sample_weight=sample_weight)
        if mask is not None:
            sample_weight = mask[:, class_ind]
        else:
            sample_weight = None
        pre_class_metrics[class_name]['epoch_loss_avg'].update_state(class_loss)
        pre_class_metrics[class_name]['epoch_accuracy'].update_state(class_labels, class_predictions,
                                                                     sample_weight=sample_weight)
        pre_class_metrics[class_name]['tp_metrics'].update_state(class_labels, class_predictions,
                                                                 sample_weight=sample_weight)
        pre_class_metrics[class_name]['fp_metrics'].update_state(class_labels, class_predictions,
                                                                 sample_weight=sample_weight)
        pre_class_metrics[class_name]['tn_metrics'].update_state(class_labels, class_predictions,
                                                                 sample_weight=sample_weight)
        pre_class_metrics[class_name]['fn_metrics'].update_state(class_labels, class_predictions,
                                                                 sample_weight=sample_weight)
        pre_class_metrics[class_name]['precision_metrics'].update_state(class_labels, class_predictions,
                                                                        sample_weight=sample_weight)
        pre_class_metrics[class_name]['recall_metrics'].update_state(class_labels, class_predictions,
                                                                     sample_weight=sample_weight)
        pre_class_metrics[class_name]['auc'].update_state(class_labels, class_predictions, sample_weight=sample_weight)


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
    # Define optimzer and loss function:
    #
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    global_step = 0
    for epoch in range(num_epochs):
        #
        # Tensorboard summaries:
        #
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
        pre_class_metrics = create_per_class_metrics()

        #
        # Start training loop:
        #
        for i, batch_data in enumerate(train_ds):
            images_batch, labels_batch = batch_data['image'], batch_data[
                'ground_truth']  # Labels shape: [batch_size, num_classes], [[1, 0, 1], [0, 1, 1]]
            mask = tf.cast(labels_batch != -1, dtype=tf.float32)
            labels_batch = tf.reshape(labels_batch, [labels_batch.shape[0], len(dataset.CLASSES), 1])
            global_step += 1
            if global_step > 18000:
                break
            #
            # Train Step:
            #
            with tf.GradientTape() as tape:
                logits = model(images_batch, training=True)
                logits = tf.reshape(logits, [logits.shape[0], len(dataset.CLASSES), 1])
                loss_value = loss_fn(labels_batch, logits, sample_weight=mask)
                grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            logits = model(images_batch, training=True)
            logits = tf.reshape(logits, [logits.shape[0], len(dataset.CLASSES), 1])
            probs = tf.sigmoid(logits)  # --> shape of : [batch_size, num_classes], [[0.8, 0.7, 0.33], [0.2, 0.9, 0.1]]

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            epoch_accuracy.update_state(labels_batch, probs, sample_weight=mask)
            update_per_class_metrics(pre_class_metrics, labels_batch, probs, logits, loss_fn, mask)

            if global_step % 10 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('batch_loss', epoch_loss_avg.result(), step=global_step)
                    tf.summary.scalar('accuracy', epoch_accuracy.result(), step=global_step)
                    for class_name in dataset.CLASSES:
                        for metric_name, metric in pre_class_metrics[class_name].items():
                            tf.summary.scalar(f'{class_name}_{metric_name}', metric.result(), step=global_step)

            if global_step % 50 == 0:
                print(f"Epoch [{epoch + 1}] Global Step [{global_step + 1}]: Loss: {epoch_loss_avg.result():.3f}, "
                      f"Accuracy: {epoch_accuracy.result():.3f}")

            if global_step % 100 == 0:
                #
                # Evaluate on validation set:
                #
                val_epoch_loss_avg = tf.keras.metrics.Mean()
                val_pre_class_metrics = create_per_class_metrics()
                val_epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

                ground_truths = np.array([]).reshape(0, len(dataset.CLASSES))
                predictions = np.array([]).reshape(0, len(dataset.CLASSES))
                for batch_data in val_ds:
                    images_batch, labels_batch = batch_data['image'], batch_data[
                        'ground_truth']
                    logits = model(images_batch, training=False)
                    probs = tf.sigmoid(logits)  # [batch_size, num_classes]
                    loss_value = loss_fn(labels_batch, logits)
                    val_epoch_loss_avg.update_state(loss_value)
                    val_epoch_accuracy.update_state(labels_batch, probs)
                    update_per_class_metrics(val_pre_class_metrics, labels_batch, probs, logits, loss_fn)

                    ground_truths = np.concatenate([ground_truths, labels_batch.numpy()], axis=0)
                    predictions = np.concatenate([predictions, probs.numpy()], axis=0)

                print(f"Validation-Set: Epoch [{epoch + 1}] Iteration: [{global_step + 1}], "
                      f"Accuracy: {val_epoch_accuracy.result():.3f},"
                      f"Loss: {val_epoch_loss_avg.result():.3f}")
                with test_summary_writer.as_default():
                    tf.summary.scalar('epoch_accuracy', val_epoch_accuracy.result(), step=global_step)
                    tf.summary.scalar('epoch_loss', val_epoch_loss_avg.result(), step=global_step)
                    for class_name in dataset.CLASSES:
                        for metric_name, metric in val_pre_class_metrics[class_name].items():
                            tf.summary.scalar(f'{class_name}_{metric_name}', metric.result(), step=global_step)

                for class_ind, class_name in enumerate(dataset.CLASSES):
                    per_class_gt = ground_truths[:, class_ind]
                    per_class_preds = predictions[:, class_ind]
                    roc_auc = utils.plot_roc(class_name, per_class_gt, per_class_preds,
                                             train_size,
                                             test_summary_writer, global_step)

                    precision_recall_auc = utils.plot_precision_recall(class_name, per_class_gt, per_class_preds,
                                                                       train_size,
                                                                       test_summary_writer, global_step)
                    with test_summary_writer.as_default():
                        tf.summary.scalar(f'{class_name}_epoch_roc_auc', roc_auc, step=global_step)
                        tf.summary.scalar(f'{class_name}_epoch_precision_recall_auc', precision_recall_auc,
                                          step=global_step)
        #
        # End of Epoch:
        #

        epoch_loss = epoch_loss_avg.result()
        epoch_accuracy = epoch_accuracy.result()
        print(f"End of Epoch {epoch + 1}:, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}")
        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', epoch_loss, step=global_step)
            tf.summary.scalar('epoch_accuracy', epoch_accuracy, step=global_step)
            for class_name in dataset.CLASSES:
                for metric_name, metric in pre_class_metrics[class_name].items():
                    tf.summary.scalar(f'epoch_{class_name}_{metric_name}', metric.result(), step=epoch)
        if global_step > 18000:
            break

    if not os.path.isdir(os.path.join(model_dir, 'saved_model')):
        os.mkdir(os.path.join(model_dir, 'saved_model'))
    model.save(os.path.join(model_dir, 'saved_model', 'my_model'))

    # return roc_auc, precision_recall_auc
