import matplotlib.pyplot as plt
import json


def err_disp(iter, train, valid):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('sigmoid_SGD_Cost per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(iter, train, color='#1F77B4', label='Training')
    ax.plot(iter, valid, color='#b41f1f', label='Validation')
    ax.set_xlabel('iter')
    ax.set_ylabel('Cost')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def acc_disp(iter, train, valid):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('sigmoid_SGD_Accuracy per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(iter, train, color='#1F77B4', label='Training')
    ax.plot(iter, valid, color='#b41f1f', label='Validation')
    ax.set_xlabel('iter')
    ax.set_ylabel('Accuracy')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def gender_load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()

    """
    data = {"train_cost": self.train_cost,
            "valid_cost": self.valid_cost,
            "train_accuracy": self.train_acc,
            "valid_accuracy": self.valid_acc}
    """
    train_cost = data["train_cost"]
    train_accuracy = data["train_accuracy"]
    val_cost = data["valid_cost"]
    val_accuracy = data["valid_accuracy"]

    return train_cost, train_accuracy, val_cost, val_accuracy


def age_load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()

    """
    data = {"train_cost": self.train_cost,
            "valid_cost": self.valid_cost,
            "train_accuracy1": self.train_acc1,
            "train_accuracy2": self.train_acc2,
            "valid_accuracy1": self.valid_acc1,
            "valid_accuracy2": self.valid_acc2}
    """
    train_cost = data["train_cost"]
    train_accuracy1 = data["train_accuracy1"]
    train_accuracy2 = data["train_accuracy2"]
    val_cost = data["valid_cost"]
    val_accuracy1 = data["valid_accuracy1"]
    val_accuracy2 = data["valid_accuracy2"]

    return train_cost, train_accuracy1, train_accuracy2, val_cost, val_accuracy1, val_accuracy2


if __name__ == '__main__':
    iter = []
    for i in range(0, 30001, 100):
        iter.append(i)

    path = './Folds/tf/gender_test_fold_is_0/run-5508/train/gender_test_fold_0.json'
    train_cost, train_accuracy, val_cost, val_accuracy = gender_load(path)
    err_disp(iter, train_cost, val_cost)
    acc_disp(iter, train_accuracy, val_accuracy)
