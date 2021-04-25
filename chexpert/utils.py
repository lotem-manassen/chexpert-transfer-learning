import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import os
from bokeh.plotting import ColumnDataSource, figure, output_file, show
import io

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def create_dataset(data_dir, batch_size, resize_dim, split):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,

        validation_split=0.2,
        subset=split,
        seed=123,
        image_size=(resize_dim, resize_dim),
        batch_size=batch_size)
    return ds


def present_dataset(dataset):
    class_names = dataset.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


def print_predictions(model, dataset):
    label_map = {0: 'female', 1: 'male'}
    for e in dataset:
        logits = model(e[0], training=False)
        print(logits.numpy())
        pred_class = tf.argmax(logits, axis=1).numpy()
        pred_class_str = [label_map[x] for x in pred_class]
        labels_str = [label_map[x] for x in e[1].numpy()]
        label_preds = [x for x in zip(labels_str, pred_class_str)]
        print(label_preds)


def plot_results(history, epochs, experiment_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    plt.savefig(experiment_name + "/loss_accuracy.png")


def plot_auc_images(name, train_size_list, train_size_str, auc_list, experiment_name):
    plt.figure()
    plt.plot(train_size_list, auc_list, '-ok', color=colors[0])
    plt.ylim([0.7, 1])
    plt.xlabel('Train dataset size')
    plt.ylabel('AUC')
    plt.xscale('log')
    plt.xticks(train_size_list, train_size_str)
    formatted_auc = ['%.3f' % elem for elem in auc_list]
    for i, txt in enumerate(formatted_auc):
        plt.annotate(txt, (train_size_list[i], auc_list[i]))
    plt.savefig(experiment_name + "/" + name + "train_size.png")


def plot_metrics(history, experiment_name):
    print("Plotting metrics")
    metrics = ['loss', 'auc', 'precision', 'recall', 'accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.figure()
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
          plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
          plt.ylim([0,1])
        else:
          plt.ylim([0,1])
        plt.legend()
        plt.savefig(experiment_name + "/" + metric + ".png")
        # plt.savefig("metadata-classifier/" + metric + "-" + experiment_name + ".png")


def plot_cm(labels, predictions, experiment_name, p=0.5):
    print("Plotting cm")
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(experiment_name + "/" + "cm.png")
    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))


def plot_roc(class_name, labels, predictions, train_size, writer, step):
    fig = plt.figure()
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    auc = sklearn.metrics.roc_auc_score(labels, predictions)
    plt.plot(fp, tp, label="AUC={0:.3f}".format(auc), linewidth=2)
    plt.legend(loc='best', prop={"size": 8})
    plt.title(f"{class_name} ROC curve (train size: {train_size})")
    plt.xlabel('False positive rate')
    plt.ylabel('True positives rate')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    im = plot_to_image(fig)
    with writer.as_default():
        tf.summary.image(f"{class_name}_roc_curve", im, step=step)
    return auc


def plot_precision_recall(class_name, labels, predictions, train_size, writer, step):
    fig = plt.figure()
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    auc = sklearn.metrics.average_precision_score(labels, predictions)

    plt.plot(recall, precision, label="AP={0:.3f}".format(auc), linewidth=2)
    plt.legend(loc='best', prop={"size": 8})
    plt.title(f"{class_name} Precision Recall curve (train size: {train_size})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.savefig(experiment_name + "/" + "precision_recall_train_size_" + str(train_size) + ".png")
    # plt.savefig("metadata-classifier/" + experiment_name + "/" + "roc-" + name + ".png")
    im = plot_to_image(fig)
    with writer.as_default():
        tf.summary.image(f"{class_name}_precision_recall", im, step=step)
    return auc


def test(model_name, batch_size, test_ds):
    model = tf.keras.models.load_model(model_name)
    print("Creates predictions for model " + model_name)
    predictions = model.predict(test_ds, batch_size=batch_size)
    labels = np.array([])
    for _, labels_ind_batch in test_ds:
        labels = np.concatenate([labels, labels_ind_batch.numpy()])
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    auc = sklearn.metrics.roc_auc_score(labels, predictions)
    return fp, tp, auc


def plot_multi_model_roc(models_folders, models_names, batch_size, test_ds, exp_name, **kwargs):
    print("Plotting multi model roc")
    plt.figure()
    for model_folder, model_name in zip(models_folders, models_names):
        fp, tp, auc = test(model_folder, batch_size, test_ds)
        plt.plot(100*fp, 100*tp, label=model_name + "- AUC={0:.3f}".format(auc), linewidth=2, **kwargs)
    plt.legend(loc='best', prop={"size":8})
    plt.title(exp_name)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([0, 100.5])
    plt.ylim([0, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(os.path.join("comparison_results", exp_name + ".png"))


def bokeh_plot_roc(model_dir, labels, predictions):
    output_file(os.path.join(model_dir, 'roc_curve.html'))

    fp, tp, probabilities = sklearn.metrics.roc_curve(labels, predictions)

    source = ColumnDataSource(data=dict(
        x=fp,
        y=tp,
        prob=probabilities,
    ))

    TOOLTIPS = [
        ("(fp, tp)", "($x, $y)"),
        ("prob", "@prob"),
    ]

    p = figure(plot_width=400, plot_height=400, tooltips=TOOLTIPS,
               title="ROC Curve")

    # p.circle('x', 'y', size=20, source=source)
    p.line('x', 'y', line_width=4, source=source)
    show(p)
