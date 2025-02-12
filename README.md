# Support Vector Machines Exercise

Now we will experiment with a very powerful algorithm, called *Support Vector Machine (SVM)*. SVMs are widely used for classification, but can also solve regression problems, as we will demostrate in our optional task for time series prediction.

### Task 1: Classification with soft-margin SVMs

In the lecture, you have seen that linear hard-margin SVMs fail to classify the data if it is not linearly separable inside the input space. For this reason, we use different kernels that transform the data into a different space, where it is easier to perform the separation. Additionally, some points may be on the wrong side of the hyperplane due to noise. To handle this, we often use *soft-margin* SVMs. Different from the hard-margin SVM, soft-margin SVM allows for certain data points to be on "the wrong side" of the hyperplane based on their distance to said plane and introduces a hyperparameter $C$ to control the effect of the regularization.

We will now apply this algorithm for face recognition. We will use the [Labeled Faces in the Wild Dataset](http://vis-www.cs.umass.edu/lfw/).

1. Starting in the `__main__` function of `src/ex2_soft_margin_svm.py` load the dataset from ``sklearn.datasets.fetch_lfw_people``. This can take a while when running for the first time because it has to download the dataset. For this exercise, we only want classes with at least 70 images per person. To improve the runtime, we resize the images using a resize factor of 0.4.
2. Gather information about the dataset: Print the number of samples, the number of image features (pixels) and the number of classes.
3. Use the provided function `plot_image_matrix` to plot the first 12 images with their corresponding labels as titles.
4. Split the data 80:20 into training and test data. Use `random_state=42` in the split function.
5. SVM trains much better if the input is standartized. Use the `StandardScaler` from `sklearn.preprocessing` on the train set and standartize (Hint: `transform`) both the train and the test set.

A lot of machine learning approaches are configurable. This means that there are parameters that are not learned by the algorithm itself but rather chosen by the developer. These *hyperparameters* have to be chosen in a way to maximize performance. In this case, we have two new parameters we want to evaluate:
* The regularization constant $C$
* and the choice of the kernel function.

Now we need to find the best values for our hyperparameters. Implement the hyperparameter search in the function `cv_svm` following these steps:

6.1. Define a dictionary of parameters that you want to cross-validate. Reasonable values for $C$ range from 0.01 to 1000, so test them as `"C": [10**i for i in range(-2, 4, 1)]`. For kernels it is sufficient to test `linear`, `rgb` and `poly`.

6.2. Initialize your model using the `sklearn.svm.SVC` class. Use the ``sklearn.model_selection.GridSearchCV`` class to find optimal hyperparameters for this task.  

7. Print the parameters of the best estimator found with the function `cv_svm`.

8. Calculate and print the accuracy of the best performing model.

9. Plot the output for some images from the test set using the function `plot_image_matrix`. Plot the predictions and the true labels of images as titles.

### Task 2 (Optional): Time-series prediction with SVM

You can also use SVMs for regression. In this exercise, we will take a brief look at time-series predictions. The goal is to infer new values from a set of old observations. For this we will look at the number of Covid-19 cases.

0. Open `src/ex3_time_series.py`, move to the `__main__` function and have a look at the code. Inspect the dataset closely and make sure you understand what information the columns depict.
1. In the code we generate two arrays: `raw_data` and `raw_data_short`. Plot both curves with the `plot_curve` function. Do you notice any change in behavior in these curves? Is there a point were the rate of change increases? The data that lies before this point won't be considered anymore.

2. With the number of covid cases for the last week (7 days), we want to predict the expected number of cases for the next 5 days. Set the number of days you want to forecast and the number of days that will be taken into account for the forecast.

3. Build the dataset for training and testing:
   * For this, split the data in the following way:
        ```python
        sequence = [10,14,15,19,20,25,26] # Number of cases
        X = [[10,14],
            [14,15],
            [15,19],
            [19,20],
            [20,25]]
        Y = [15, 19, 20, 25, 26] 
        ```
        In this example it means that we use the first 2 days (``[10,14]``) to predict the third day (``[15]``) and the second and third day to predict the fourth and so on. Instead of 2 days, we use 7 days for the prediction.
4. SVMs are not scale invariant, so it is important to normalize the input data. Normalize the data to its maximum value, such that it lies between [0,1]. (Hint: `numpy.amax`). Note, if you normalize the train data, you need to normalize the test data as well!

Now we need to train an SVM regressor and find the best values for the hyperparameters. For this task, we will choose a Gaussian `rbf` kernel and evaluate the following parameters:
* The regularization constant $C$,
* $epsilon$ in the epsilon-SVR model. It specifies the the epsilon-tube, within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value,
* and the $gamma$ parameter of the `rbf` kernel. 

Implement the hyperparameter search in the function `cv_svr` following these steps:

5.1. Define a dictionary of parameters that you want to cross-validate. (Hint: Reasonable values for $epsilon$ range from 0.1 to 0.001 and for $gamma$ you can try the values `auto` and `scale`.)

5.2. Initialize your model using the `sklearn.svm.SVR` class. Use the grid search to find optimal hyperparameters. 

6. Print the parameters of the best estimator found with the function `cv_svr`.
	
7. After that go to the ``recursive_forecast()`` function where the new predictions are recursivley used to generate predictions even further in the future. Implement the TODOs.
8. Use the function `recursive_forecast` to make predictions for the next 5 days. Don't forget to denormalize your predictions. Use `numpy.round` to round the predictions after denormalization. 
9. Plot the predicted results with `plot_prediction`.
