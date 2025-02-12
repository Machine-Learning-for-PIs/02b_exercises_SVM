"""Use soft-margin SVM for face recognition."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def plot_image_matrix(images, titles, h, w, n_row=3, n_col=4) -> None:
    """Plot a matrix of images.

    Args:
        images (np.ndarray): The array of the images.
        titles (np.ndarray or list): The titles of the images.
        h (int): The height of one image.
        w (int): The width of one image.
        n_row (int): The number of rows of images to plot.
        n_col (int): The number of columns of images to plot.
    """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    indices = np.arange(n_row * n_col)
    #np.random.shuffle(indices)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[indices[i]].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[indices[i]], size=12)
        plt.xticks(())
        plt.yticks(())


def cv_svm(xtrain: np.ndarray, ytrain: np.ndarray) -> GridSearchCV:
    """Train and cross-validate a soft-margin SVM classifier with the grid search.

    Define an SVM classifier. Use the grid search and a 5-fold cross-validation
    to find the best value for the hyperparameters 'C' and 'kernel'.

    Args:
        xtrain (np.ndarray): The training data.
        ytrain (np.ndarray): The training labels.

    Returns:
        GridSearchCV: The trained model that was cross-validated with the grid search.
    """
    # 6.1. define dictionary with parameter grids
    # TODO

    # 6.2. initialize svm classifier and perform grid search
    # TODO

    return None


if __name__ == "__main__":
    # 1. load dataset 'Labeled Faces in the Wild';
    # take only classes with at least 70 images; downsize images for speed up
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # 2. gather information about the dataset
    n_samples, h, w = lfw_people.images.shape
    x = lfw_people.data
    n_features = x.shape[1]
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    # print number of samples, number of image features (pixels) and number of classes
    # TODO

    # 3. plot some images of dataset
    titles = [target_names[lfw_people.target[i]] for i in range(12)]
    plot_image_matrix(x, titles, h, w)
    plt.show()

    # 4. split data into training and test data
    # TODO
    
    # 5. use 'StandardScaler' on train data and scale both train and test data
    # TODO

    # 6. use function 'cv_svm' to perform hyperparameter search with cross validation
    # TODO

    # 7. print parameters found with cross-validation
    # TODO

    # 8. compute and print accuracy of best model on test set
    # TODO

    # 9. Plot images together with predicitons
    # TODO adjust your variables x_test, y_test and y_pred:

    # def title(y_pred, y_test, target_names, i):
    #     """Generate title."""
    #     pred_name = target_names[y_pred[i]].rsplit(" ", 1)[-1]
    #     true_name = target_names[y_test[i]].rsplit(" ", 1)[-1]
    #     return "predicted: %s\ntrue:      %s" % (pred_name, true_name)

    # prediction_titles = [
    #     title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])
    # ]

    # plot_image_matrix(x_test, prediction_titles, h, w)
    # plt.show()
