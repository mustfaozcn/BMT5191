import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
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

    matrix = confusion_matrix(y_test, model.predict(X_tŸest))
    print("Confusion Matrix:\n", matrix)

    # classification_report çıktısını alın
    report = classification_report(y_test, model.predict(X_test))
    print("\nClassification Report:\n", report)

    # raporu bir dosyaya yazın
    with open(f"{model.kernel}_classification_report.txt", 'w') as f:
        f.write(report)


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


def plot_roc(X_test, y_test, model, kernel):
    y_pred = model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{kernel.capitalize()} kernel SVM\nReceiver operating characteristic')
    plt.legend(loc="lower right")

    plt.savefig(f"{kernel}_svm_roc.png")  # save ROC curve
    plt.show()


if __name__ == '__main__':
    data = load_data()
    X, y, X_train, X_test, y_train, y_test = split_data(data)

    visualize_data(X, y)

    linear_svm_model = train_linear_svm(X_train, y_train)
    evaluate_model(linear_svm_model, X_train, y_train, X_test, y_test)
    plot_svm(X_train, y_train, X_test, y_test, linear_svm_model, 'linear')
    plot_roc(X_test, y_test, linear_svm_model, 'linear')

    multi_kernel_svm_model = train_multi_kernel_svm(X_train, y_train)
    evaluate_model(multi_kernel_svm_model, X_train, y_train, X_test, y_test)
    plot_svm(X_train, y_train, X_test, y_test, multi_kernel_svm_model, 'multi_kernel')
    plot_roc(X_test, y_test, multi_kernel_svm_model, 'multi_kernel')

    polynomial_svm_model = train_polynomial_svm(X_train, y_train)
    evaluate_model(polynomial_svm_model, X_train, y_train, X_test, y_test)
    plot_svm(X_train, y_train, X_test, y_test, polynomial_svm_model, 'polynomial')
    plot_roc(X_test, y_test, polynomial_svm_model, 'polynomial')

    rbf_svm_model = train_rbf_svm(X_train, y_train)
    evaluate_model(rbf_svm_model, X_train, y_train, X_test, y_test)
    plot_svm(X_train, y_train, X_test, y_test, rbf_svm_model, 'rbf')
    plot_roc(X_test, y_test, rbf_svm_model, 'rbf')
