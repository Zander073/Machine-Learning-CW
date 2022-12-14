"""
Author:
Date:
Description:
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier():
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier):

    def __init__(self):
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y):
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        vals, counts = np.unique(y, return_counts=True)
        majority_val, majority_count = max(zip(vals, counts), key=lambda val_count: val_count[1])
        self.prediction_ = majority_val
        return self

    def predict(self, X):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None:
            raise Exception('Classifier not initialized. Perform a fit first.')

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier):

    def __init__(self):
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y):
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO: START ========== ###
        # part c: set self.probabilities_ according to the training set
        total = Counter(y)
        # P(0) = toal 0/(total 0 + total 1)
        # P(1) = 1 - P(0)
        self.probabilities_ = (float(total[0.0])/(float(total[0.0]) + float(total[1.0])))
        ### ========== TODO: END ========== ###

        return self

    def predict(self, X, seed=1234):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None:
            raise Exception('Classifier not initialized. Perform a fit first.')
        np.random.seed(seed)

        ### ========== TODO: START ========== ###
        # part c: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        y = np.random.choice(2, X.shape[0],p=[self.probabilities_, 1-self.probabilities_])
        ### ========== TODO: END ========== ###
        return y


######################################################################
# functions
######################################################################

def plot_histogram(X, y, Xname, yname):
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets:
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else:
        bins = 10
        align = 'mid'

    # plot
    plt.figure()
    n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
    plt.xlabel(Xname)
    plt.ylabel('Frequency')
    plt.legend() #plt.legend(loc='upper left')
    plt.show(block=True)


def plot_scatter(X, y, Xnames, yname):
    """
    Plots scatter plot of values in X grouped by y.

    Parameters
    --------------------
        X      -- numpy array of shape (n,2), feature values
        y      -- numpy array of shape (n,), target classes
        Xnames -- tuple of strings, names of features
        yname  -- string, name of target
    """

    # plot
    targets = sorted(set(y))
    plt.figure()
    ### ========== TODO: START ========== ###
    # part b: scatterplot
    # hint: use plt.scatter (and set the label)
    # feel free to rewrite this function from scratch if you prefer
    plt.scatter(X[0], X[1], c=y[:])
    ### ========== TODO: END ========== ###
    plt.autoscale(enable=True)
    plt.xlabel(Xnames[0])
    plt.ylabel(Xnames[1])
    plt.legend(loc="upper right")
    plt.show(block=True)


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data('titanic_train.csv', header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features

    #========================================
    # plot histograms of each feature
    """
    print('Plotting...')
    for i in range(d):
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
    """

    ### ========== TODO: START ========== ###
    # part b: make scatterplot of age versus fare
    age = X[:,2]
    fare = X[:,5]
    plot_scatter(X=[age,fare], y=y, Xnames=[Xnames[2], Xnames[5]], yname=yname)
    ### ========== TODO: END ========== ###



    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO: START ========== ###
    # part c: evaluate training error of Random classifier
    print('Classifying using Random...')
    ran_clf = RandomClassifier() 
    ran_clf.fit(X, y)            
    y_pred = ran_clf.predict(X) 
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO: END ========== ###



    print('Done')


if __name__ == '__main__':
    main()
