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

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

######################################################################
# classes
######################################################################

class Classifier(object):
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
        # insert your RandomClassifier code
        total = Counter(y)
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
        # insert your RandomClassifier code
        y = np.random.choice(2, X.shape[0],p=[self.probabilities_, 1-self.probabilities_])
        ### ========== TODO: END ========== ###
        return y

######################################################################
# functions
######################################################################

def error(clf, X, y, ntrials=100, test_size=0.2):
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
        test_size   -- float (between 0.0 and 1.0) or int,
                       if float, the proportion of the dataset to include in the test split
                       if int, the absolute number of test samples

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO: START ========== ###
    # part b: compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    # TODO: for each trial, fit clf on new training data and predict on test
    # TODO: return the average train and test errors over all trials

    train_error = 0
    test_error = 0

    # test size is 20% (80/20 split of the training data)

    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        # Collecting the total sum of the training error and test error to be averaged:
        train_error = train_error + (1 - metrics.accuracy_score(y_train, y_train_pred, normalize=True))
        test_error = test_error + (1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True))

        # Averaging the training error and test error:
    train_error = train_error / ntrials
    test_error = test_error / ntrials
    ### ========== TODO: END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None):
    """Write out predictions to csv file."""
    out = open(filename, 'w')
    f = csv.writer(out)
    if yname:
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


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
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    #========================================
    # train Random classifier on data
    print('Classifying using Random...')
    ran_clf = RandomClassifier() 
    ran_clf.fit(X, y)            
    y_pred = ran_clf.predict(X) 
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)


    ### ========== TODO: START ========== ###
    # part a: evaluate training error of Decision Tree classifier
    print('Classifying using Decision Tree...')
    dec_tree_clf = DecisionTreeClassifier(criterion="entropy")
    dec_tree_clf.fit(X, y)
    y_pred = dec_tree_clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    print()
    ### ========== TODO: END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    # save the classifier -- requires GraphViz and pydot
    """
    import pydot
    from io import StringIO
    from sklearn import tree
    dot_data = StringIO()
    tree.export_graphviz(dec_tree_clf, out_file=dot_data,
                         feature_names=Xnames,
                         class_names=['Died', 'Survived'])
    graph = pydot.graph_from_dot_data(str(dot_data.getvalue()))[0]
    graph.write_pdf('dtree.pdf')
    """

    ### ========== TODO: START ========== ###
    # part b: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    clf_error = error(clf, X, y, test_size=0.2)
    print('    For Majority Vote classifier:')
    print('\t-- training error: %.3f' % clf_error[0])
    print('\t-- testing error: %.3f' % clf_error[1])

    clf_error = error(ran_clf, X, y, test_size=0.2)
    print('    For Random classifier:')
    print('\t-- training error: %.3f' % clf_error[0])
    print('\t-- testing error: %.3f' % clf_error[1])

    clf_error = error(dec_tree_clf, X, y, test_size=0.2)
    print('    For Decision Tree classifier:')
    print('\t-- training error: %.3f' % clf_error[0])
    print('\t-- testing error: %.3f' % clf_error[1])
    print()
    ### ========== TODO: END ========== ###


    ### ========== TODO: START ========== ###
    # part c: investigate decision tree classifier with various depths
    # Uncomment to plot:
    print('Investigating depths...')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # We want our tree depth to go as the following: 1, 2, 3, ..., 18, 19, 20
    max_depth_list = list(range(1, 21)) 

    # Let's get the test errors for majority vote and random classifiers:
    mv_clf = MajorityVoteClassifier()
    mv_clf.fit(X_train, y_train)
    mv_test_pred = mv_clf.predict(X_test)
    mv_test_error = 1 - metrics.accuracy_score(y_test, mv_test_pred, normalize=True)

    ran_clf = RandomClassifier()
    ran_clf.fit(X_train, y_train)
    ran_test_pred = ran_clf.predict(X_test)
    ran_test_error = 1 - metrics.accuracy_score(y_test, ran_test_pred, normalize=True)

    
    mv_clf_test_errors = [mv_test_error] * len(max_depth_list) # Log of testing error for MV
    ran_clf_test_errors = [ran_test_error] * len(max_depth_list) # Log of testing error for Random
    dt_train_errors = [] # Log of training error for DT
    dt_test_errors = [] # Log of testing error for DT

    for x in max_depth_list:
        dtc = DecisionTreeClassifier(max_depth=x) 
        dtc.fit(X_train,y_train)
        y_train_pred = dtc.predict(X_train)
        y_test_pred = dtc.predict(X_test)
        # Append decision tree train and test errors for depth x:
        dt_train_errors.append(1 - metrics.accuracy_score(y_train, y_train_pred, normalize=True))
        dt_test_errors.append(1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True))

    x = np.arange(len(max_depth_list)) + 1
    plt.plot(x, mv_clf_test_errors, label='Majority Vote Test')
    plt.plot(x, ran_clf_test_errors, label='Random Test')
    plt.plot(x, dt_train_errors, label='Decision Tree Train')
    plt.plot(x, dt_test_errors, label='Decision Tree Test')
    plt.ylim(0,1)
    plt.xlabel('Depth') 
    plt.ylabel('Error') 
    plt.legend(loc='upper right')
    # Uncomment if you want to view the graph in real time:
    plt.show() 
    print()
    ### ========== TODO: END ========== ###


    ### ========== TODO: START ========== ###
    # part d: investigate decision tree classifier with various training set sizes
    print('Investigating training set sizes...')
    print('WAIT FOR GRAPH TO CLEAR!!')
    current_batch_size = 0.05
    max_batch = 19 # 95/5
    batch_step = 0.05
    mv_clf = MajorityVoteClassifier()
    ran_clf = RandomClassifier()
    dec_tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=6)

    batch_list = []
    mv_clf_test_error = []
    ran_clf_test_error = []
    dt_clf_train_error = []
    dt_clf_test_error = []
    for i in range(max_batch):
        mv_clf_test_error.append(error(mv_clf, X, y, test_size=current_batch_size)[1])
        ran_clf_test_error.append(error(ran_clf, X, y, test_size=current_batch_size)[1])
        dt_clf_train_error.append(error(dec_tree_clf, X, y, test_size=current_batch_size)[0])
        dt_clf_test_error.append(error(dec_tree_clf, X, y, test_size=current_batch_size)[1])
        batch_list.append(current_batch_size)
        current_batch_size += batch_step
    
    # Clearing previous graph:
    plt.clf()
    x = batch_list
    plt.plot(x, mv_clf_test_error, label='Majority Vote Test')
    plt.plot(x, ran_clf_test_error, label='Random Test')
    plt.plot(x, dt_clf_train_error, label='Decision Tree Train')
    plt.plot(x, dt_clf_test_error, label='Decision Tree Test')
    plt.ylim(0,1)
    plt.xlabel('Batch Size') 
    plt.ylabel('Error') 
    plt.legend(loc='upper right')
    # Uncomment to view graph:
    plt.show()
    ### ========== TODO: END ========== ###



    ### ========== TODO: START ========== ###
    # Contest
    # uncomment write_predictions and change the filename

    # evaluate on test data
    titanic_test = load_data('titanic_test.csv', header=1, predict_col=None)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    clf.fit(X, y)
    X_test = titanic_test.X
    y_pred = clf.predict(X_test)   # take the trained classifier and run it on the test data
    # write_predictions(y_pred, '../data/Zander073_titanic.csv', titanic.yname)
    ### ========== TODO: END ========== ###
    
    print('Done')


if __name__ == '__main__':
    main()
