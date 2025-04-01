import matplotlib.pyplot as plt
import numpy as np
import op

from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

def main():
    rng = np.random.default_rng()
    n = 50

    fig, axs = plt.subplots(3, 3)
    fig.suptitle('Perceptron')
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')

    operations = [op.And, op.Or, op.Xor, op.Left, op.Right, op.Impl, op.Lpmi]
    operation_names = ['And', 'Or', 'Xor', 'Bit 1', 'Bit 0', '⇒', '⇐']

    x = rng.binomial(1, 0.5, size=(n, 2))
    x_hack = x[:, [0, 0]]
    for i, binary_op in enumerate(operations):
        y = binary_op(x[:, 0], x[:, 1])

        model = Perceptron()
        model.fit(x_hack, y)

        ax_x, ax_y = i // 3, i % 3
        plot_decision_regions(x.astype(float), y, model, ax=axs[ax_x, ax_y], legend=0)
        axs[ax_x, ax_y].set_title(operation_names[i])
        axs[ax_x, ax_y].axis('off')
    plt.show()

    fig, axs = plt.subplots(3, 3)
    fig.suptitle('DT depth 2')
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')

    for i, binary_op in enumerate(operations):
        y = binary_op(x[:, 0], x[:, 1])

        model = DecisionTreeClassifier(max_depth=2)
        model.fit(x_hack, y)

        ax_x, ax_y = i // 3, i % 3
        plot_decision_regions(x.astype(float), y, model, ax=axs[ax_x, ax_y], legend=0)
        axs[ax_x, ax_y].set_title(operation_names[i])
        axs[ax_x, ax_y].axis('off')
    plt.show()


    fig, axs = plt.subplots(3, 3)
    fig.suptitle('SVM')
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')

    for i, binary_op in enumerate(operations):
        y = binary_op(x[:, 0], x[:, 1])

        model = SVC()
        model.fit(x_hack, y)

        ax_x, ax_y = i // 3, i % 3
        plot_decision_regions(x.astype(float), y, model, ax=axs[ax_x, ax_y], legend=0)
        axs[ax_x, ax_y].set_title(operation_names[i])
        axs[ax_x, ax_y].axis('off')
    plt.show()

if __name__ == '__main__':
    main()