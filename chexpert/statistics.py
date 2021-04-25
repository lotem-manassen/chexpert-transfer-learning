"""Calculate manual statistics."""
from matplotlib import pyplot as plt
from chexpert import dataset


def regular_auc_per_train_size():
    classes = {}
    Atelectasis_auc = [(100, 0.64), (1000, 0.67), (5000, 0.76), (10000, 0.76), (50000, 0.8), (100000, 0.83)]
    classes['Atelectasis'] = Atelectasis_auc
    Cardiomegaly_auc = [(100, 0.63), (1000, 0.73), (5000, 0.78), (10000, 0.76), (50000, 0.85), (100000, 0.86)]
    classes['Cardiomegaly'] = Cardiomegaly_auc
    Consolidation_auc = [(100, 0.68), (1000, 0.79), (5000, 0.84), (10000, 0.87), (50000, 0.91), (100000, 0.93)]
    classes['Consolidation'] = Consolidation_auc
    Edema_auc = [(100, 0.73), (1000, 0.8), (5000, 0.85), (10000, 0.86), (50000, 0.88), (100000, 0.9)]
    classes['Edema'] = Edema_auc
    Pleural_Effusion_auc = [(100, 0.69), (1000, 0.73), (5000, 0.8), (10000, 0.85), (50000, 0.9), (100000, 0.94)]
    classes['Pleural_Effusion'] = Pleural_Effusion_auc

    labeled_samples = [100, 1000, 5000, 10000, 50000, 100000]
    for class_name, class_auc in classes.items():
        plt.figure()
        plt.plot(labeled_samples, [x[1] for x in class_auc], label=class_name)
        plt.xlabel("#Labeled Samples")
        plt.ylabel("AUC of ROC")
        plt.title(f"{class_name}: AUC test-set per number of labeled samples")
        plt.legend()
        plt.show()

# results for 20 last layers trainable in TL (360K/7M trainable parameters)
def auc_transfer_learning_from_gender():
    classes = {}
    Atelectasis_auc = [(100, 0.65), (1000, 0.71), (5000, 0.74), (10000, 0.77), (50000, 0.78), (100000, 0.77)]
    classes['Atelectasis'] = Atelectasis_auc
    Cardiomegaly_auc = [(100, 0.61), (1000, 0.74), (5000, 0.72), (10000, 0.74), (50000, 0.75), (100000, 0.75)]
    classes['Cardiomegaly'] = Cardiomegaly_auc
    Consolidation_auc = [(100, 0.69), (1000, 0.77), (5000, 0.78), (10000, 0.81), (50000, 0.82), (100000, 0.83)]
    classes['Consolidation'] = Consolidation_auc
    Edema_auc = [(100, 0.69), (1000, 0.79), (5000, 0.81), (10000, 0.84), (50000, 0.85), (100000, 0.84)]
    classes['Edema'] = Edema_auc
    Pleural_Effusion_auc = [(100, 0.69), (1000, 0.78), (5000, 0.79), (10000, 0.8), (50000, 0.8), (100000, 0.78)]
    classes['Pleural_Effusion'] = Pleural_Effusion_auc

    labeled_samples = [100, 1000, 5000, 10000, 50000, 100000]
    for class_name, class_auc in classes.items():
        plt.figure()
        plt.plot(labeled_samples, [x[1] for x in class_auc], label=class_name)
        plt.xlabel("#Labeled Samples")
        plt.ylabel("AUC of ROC")
        plt.title(f"{class_name}: AUC test-set per number of labeled samples (after TL from gender)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    regular_auc_per_train_size()
