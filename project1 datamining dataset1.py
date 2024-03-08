#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math


# In[91]:


# Load the dataset1
dataset1 = pd.read_csv("E:\data mining prj 1\project1_dataset1.txt", header=None, sep='\t')


# In[92]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target variable
X = dataset1.iloc[:, :-1]
y = dataset1.iloc[:, -1]

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[93]:


import numpy as np
from collections import Counter

def euclidean_distance(row1, row2):
    """
    Calculate the Euclidean distance between two vectors.
    """
    return np.sqrt(np.sum((row1 - row2) ** 2))

def k_nearest_neighbors(X_train, y_train, test_sample, k=5):
    """
    Find the K nearest neighbors of a given test sample.
    """
    distances = []
    for i, train_sample in enumerate(X_train):
        distance = euclidean_distance(test_sample, train_sample)
        distances.append((distance, i))
    
    distances.sort(key=lambda x: x[0])
    neighbors_indices = [distances[i][1] for i in range(k)]
    neighbors_classes = [y_train.iloc[index] for index in neighbors_indices]

    return Counter(neighbors_classes).most_common(1)[0][0]

def predict(X_train, y_train, X_test, k=5):
    """
    Predict the class for each sample in the test set.
    """
    predictions = []
    for test_sample in X_test:
        predicted_class = k_nearest_neighbors(X_train, y_train, test_sample, k)
        predictions.append(predicted_class)
    return predictions

# Predict for all samples in the test set
predictions = predict(X_train, y_train, X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
accuracy


# In[94]:


class GaussianNaiveBayes:
    """Gaussian Naive Bayes Classifier."""

    def fit(self, X, y):
        """Fit the model to the data."""
        self.classes = np.unique(y)
        self.parameters = {}

        for class_ in self.classes:
            # Calculate parameters for each class
            X_c = X[y == class_]
            self.parameters[class_] = {
                "mean": X_c.mean(axis=0),
                "std": X_c.std(axis=0)
            }

    def _calculate_probability(self, x, mean, std):
        """Calculate the probability using Gaussian distribution."""
        exponent = np.exp(- ((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def _calculate_class_probabilities(self, x):
        """Calculate the probabilities for each class."""
        probabilities = {}
        for class_, params in self.parameters.items():
            probabilities[class_] = np.log(1 / len(self.classes))  # Prior probability
            for i in range(len(params["mean"])):
                probabilities[class_] += np.log(
                    self._calculate_probability(x[i], params["mean"][i], params["std"][i]))
        return probabilities

    def predict(self, X):
        """Predict the class for each sample in X."""
        y_pred = []
        for x in X:
            class_probabilities = self._calculate_class_probabilities(x)
            best_class = max(class_probabilities, key=class_probabilities.get)
            y_pred.append(best_class)
        return np.array(y_pred)

# Instantiate and train the model
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

# Make predictions
predictions = gnb.predict(X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
accuracy


# In[95]:


class LinearSVM:
    """Linear Support Vector Machine Classifier."""

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fit the SVM model to the training data.
        """
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent for optimization
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        """
        Predict the class for each sample in X.
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# Prepare the data (ensuring labels are in the format expected by SVM)
y_train_svm = np.where(y_train <= 0, -1, 1)
y_test_svm = np.where(y_test <= 0, -1, 1)

# Instantiate and train the model
svm = LinearSVM()
svm.fit(X_train, y_train_svm)

# Make predictions
predictions = svm.predict(X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test_svm)
accuracy


# In[98]:


class SimplifiedDecisionTreeNode:
    """Represents a simplified node in the decision tree."""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Majority class in this node

def simplified_gini_impurity(y):
    """Calculate the Gini impurity for a list of labels."""
    classes = np.unique(y)
    impurity = 1.0
    for cls in classes:
        p_cls = np.sum(y == cls) / float(len(y))
        impurity -= p_cls ** 2
    return impurity

def simplified_best_split(X, y, n_features):
    """Find the best split considering a subset of features for efficiency."""
    best_feature, best_threshold = None, None
    best_gini = 1  # Start with a Gini of 1 and try to minimize
    sample_indices = np.random.choice(X.shape[1], n_features, replace=False)
    
    for feature_index in sample_indices:
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            gini_left = simplified_gini_impurity(y[left_mask])
            gini_right = simplified_gini_impurity(y[right_mask])
            gini = (np.sum(left_mask) * gini_left + np.sum(right_mask) * gini_right) / len(y)
            
            if gini < best_gini:
                best_gini, best_threshold, best_feature = gini, threshold, feature_index
                
    return best_feature, best_threshold

def simplified_build_tree(X, y, depth, max_depth):
    """Builds the tree recursively but stops after reaching max depth."""
    num_samples_per_class = [np.sum(y == c) for c in np.unique(y)]
    predicted_class = np.argmax(num_samples_per_class)
    node = SimplifiedDecisionTreeNode(value=predicted_class)
    
    # Stop splitting if max depth is reached
    if depth >= max_depth or len(np.unique(y)) == 1:
        return node
    
    feature_index, threshold = simplified_best_split(X, y, int(np.sqrt(X.shape[1])))
    if threshold is not None:
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices
        node.feature_index, node.threshold = feature_index, threshold
        node.left = simplified_build_tree(X[left_indices], y[left_indices], depth + 1, max_depth)
        node.right = simplified_build_tree(X[right_indices], y[right_indices], depth + 1, max_depth)
    
    return node

def simplified_predict(node, x):
    """Predict a single sample using the simplified decision tree."""
    while node.left:
        if x[node.feature_index] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

# Setting up a simplified decision tree classifier class
class SimplifiedDecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = simplified_build_tree(X, y, 0, self.max_depth)

    def predict(self, X):
        return np.array([simplified_predict(self.tree, x) for x in X])

def accuracy_score(y_true, y_pred):
    """
    Calculates the accuracy, which is the proportion of correct predictions over total predictions.
    
    Parameters:
    - y_true: array-like of shape (n_samples,) - True labels of the data.
    - y_pred: array-like of shape (n_samples,) - Predicted labels, as returned by a classifier.
    
    Returns:
    - accuracy: float - The accuracy of the predictions, ranging from 0 to 1.
    """
    # Ensure that the true and predicted labels have the same length
    assert len(y_true) == len(y_pred), "The length of true labels and predicted labels must be the same."
    
    # Calculate the number of correct predictions
    correct_predictions = sum(y_true == y_pred)
    
    # Calculate the total number of predictions
    total_predictions = len(y_true)
    
    # Calculate the accuracy
    accuracy = correct_predictions / total_predictions
    
    return accuracy

# Now, let's instantiate and train a simplified decision tree with a limited depth
simplified_classifier = SimplifiedDecisionTreeClassifier(max_depth=3)
simplified_classifier.fit(X_train, y_train)

# Predict and evaluate on the test set
y_pred_simplified = simplified_classifier.predict(X_test)
accuracy_simplified = accuracy_score(y_test, y_pred_simplified)

accuracy_simplified


# In[101]:


import numpy as np

class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []  # To store classifiers and their weights

    def fit(self, X, y):
        n_samples, _ = X.shape
        w = np.full(n_samples, 1 / n_samples)  # Initial weights

        for _ in range(self.n_clf):
            clf = SimplifiedDecisionTreeClassifier(max_depth=1)  # Weak learner
            clf.fit(X, y)
            predictions = clf.predict(X)

            # Calculate the error weighted by instance weights
            miss = [int(x) for x in (predictions != y)]
            error = sum(w * miss) / sum(w)
            
            # Calculate the classifier's weight
            clf_weight = np.log((1 - error) / error) / 2

            # Update instance weights
            w = w * np.exp(-clf_weight * y * predictions)
            w /= np.sum(w)  # Normalize weights

            self.clfs.append((clf, clf_weight))

    def predict(self, X):
        clf_preds = np.array([clf_weight * clf.predict(X) for clf, clf_weight in self.clfs])
        y_pred = np.sign(np.sum(clf_preds, axis=0))
        return y_pred

# Usage example

ada = AdaBoost(n_clf=10)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_ada)
print(f"Accuracy: {accuracy}")


# In[ ]:




