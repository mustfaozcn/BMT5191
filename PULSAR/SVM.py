import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data():
    url = "https://github.com/alexandrehsd/Predicting-Pulsar-Stars/raw/master/pulsar_stars.csv"
    return pd.read_csv(url)


def split_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X, y, X_train, X_test, y_train, y_test


def train_linear_svm(X_train, y_train):
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model


def train_multi_kernel_svm(X_train, y_train):
    svc = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
    svc.fit(X_train, y_train)
    return svc


def train_polynomial_svm(X_train, y_train):
    svc = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovr')
    svc.fit(X_train, y_train)
    return svc


def train_rbf_svm(X_train, y_train):
    svc = svm.SVC(kernel='rbf', gamma='scale', C=1, decision_function_shape='ovr')
    svc.fit(X_train, y_train)
    return svc


def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_accuracy = np.mean(model.predict(X_train) == y_train)
    test_accuracy = np.mean(model.predict(X_test) == y_test)
    print(f"Training Set Accuracy: {train_accuracy}")
    print(f"Test Set Accuracy: {test_accuracy}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test)))
    print("\nClassification Report:\n", classification_report(y_test, model.predict(X_test)))


def plot_svm(X_train, y_train, X_test, y_test, model, kernel):
    train_accuracy = np.mean(model.predict(X_train) == y_train)
    test_accuracy = np.mean(model.predict(X_test) == y_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='autumn', edgecolors='black')
    plt.title(
        f"{kernel.capitalize()} kernel SVM\nTraining Accuracy: {train_accuracy:.2%}, Test Accuracy: {test_accuracy:.2%}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.savefig(f"{kernel}_svm_scatter.png")  # save scatter plot
    plt.show()

    # Confusion matrix
    cm = pd.DataFrame(confusion_matrix(y_test, model.predict(X_test)), columns=['Predicted 0', 'Predicted 1'],
                      index=['Actual 0', 'Actual 1'])
    plt.figure(figsize=(6, 6))

    sns.heatmap(cm, annot=True, cmap='Blues')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{kernel.capitalize()} kernel SVM\nConfusion matrix")

    plt.savefig(f"{kernel}_svm_confusion_matrix.png")  # save confusion matrix
    plt.show()


def visualize_data(X, y):
    plt.scatter(X[' Mean of the integrated profile'], X[' Mean of the DM-SNR curve'], c=y, cmap='viridis')
    plt.xlabel('Mean of the integrated profile')
    plt.ylabel('Mean of the DM-SNR curve')
    plt.title('Pulsar Star Classification')
    plt.savefig("pulsar_star_classification")
    plt.show()


if __name__ == '__main__':
    data = load_data()
    X, y, X_train, X_test, y_train, y_test = split_data(data)

    visualize_data(X, y)

    linear_svm_model = train_linear_svm(X_train, y_train)
    plot_svm(X_train, y_train, X_test, y_test, linear_svm_model, 'linear')

    multi_kernel_svm_model = train_multi_kernel_svm(X_train, y_train)
    plot_svm(X_train, y_train, X_test, y_test, multi_kernel_svm_model, 'multi_kernel')

    polynomial_svm_model = train_polynomial_svm(X_train, y_train)
    plot_svm(X_train, y_train, X_test, y_test, polynomial_svm_model, 'polynomial')

    rbf_svm_model = train_rbf_svm(X_train, y_train)
    plot_svm(X_train, y_train, X_test, y_test, rbf_svm_model, 'rbf')
