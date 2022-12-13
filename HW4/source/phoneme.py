"""
Author:
Date:
Description: Perceptron vs Logistic Regression on a Phoneme Dataset
"""

# utilities
from util import *

# scipy libraries
from scipy import stats

# scikit-learn libraries
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import Perceptron, LogisticRegression

######################################################################
# functions
######################################################################

def cv_performance(clf, train_data, kfs):
    """
    Determine classifier performance across multiple trials using cross-validation

    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kfs        -- array of size n_trials
                      each element is one fold from model_selection.KFold

    Returns
    --------------------
        scores     -- numpy array of shape (n_trials, n_fold)
                      each element is the (accuracy) score of one fold in one trial
    """

    n_trials = len(kfs)
    n_folds = kfs[0].n_splits
    scores = np.zeros((n_trials, n_folds))
    ### ========== TODO: START ========== ###
    # part 2: run multiple trials of cross-validation (CV)
    # for each trial, get perf on 1 trial & update scores
    for n in range(n_trials):
        scores[n] = cv_performance_one_trial(clf, train_data, kfs[n])
    ### ========== TODO: END ========== ###
    return scores


def cv_performance_one_trial(clf, train_data, kf):
    """
    Compute classifier performance across multiple folds using cross-validation

    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kf         -- model_selection.KFold

    Returns
    --------------------
        scores     -- numpy array of shape (n_fold, )
                      each element is the (accuracy) score of one fold
    """

    scores = np.zeros(kf.n_splits)

    ### ========== TODO: START ========== ###
    # part 2: run one trial of cross-validation (CV)
    # for each fold, train on its data, predict, and update score
    # hint: check KFold.split and metrics.accuracy_score
    X = train_data.X
    y = train_data.y
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = metrics.accuracy_score(y_pred, y_test)
        scores[i] = score
        i += 1
    ### ========== TODO: END ========== ###
    return scores


######################################################################
# main
######################################################################

def main():
    np.random.seed(1234)

    #========================================
    # load data
    train_data = load_data('phoneme_train.csv')
    test_data = load_data('phoneme_test.csv')
    print(train_data.X)
    print(train_data.y)

    ### ========== TODO: START ========== ###
    # part 1: is data linearly separable? Try Perceptron and LogisticRegression
    # hint: set fit_intercept = True to have offset (i.e., bias)
    # hint: you may also want to try LogisticRegression with C=1e10

    # Perceptron:
    p_clf = Perceptron(fit_intercept=True)
    p_clf.fit(train_data.X, train_data.y)
    score = p_clf.score(test_data.X, test_data.y)
    # print("Perceptron score: ", score)

    # Logistic regerssion:
    l_clf = LogisticRegression(fit_intercept=True, C=1e10)
    l_clf.fit(train_data.X, train_data.y)
    score = l_clf.score(test_data.X, test_data.y)
    # print("LR c=1e10: ", score)

    r_clf = LogisticRegression(fit_intercept=True, C=1)
    r_clf.fit(train_data.X, train_data.y)
    score = r_clf.score(test_data.X, test_data.y)
    # print("LR c=1 score: ", score)
    ### ========== TODO: END ========== ###

   
    ### ========== TODO: START ========== ###
    # parts 3-4: compare classifiers
    # make sure to use same folds across all runs (hint: model_selection.KFold)
    # hint: for standardization, use preprocessing.StandardScaler()
     # K-fold cross validation:
    kf = model_selection.KFold(n_splits = 10, random_state=np.random.randint(1234), shuffle=True)
    kfs = [kf]*10

    print("No preprocessing: ")

    p = (cv_performance(p_clf, train_data, kfs))
    print("Perceptron mean: ", np.mean(p))
    print("Perceptron std: ", np.std(p))

    l = (cv_performance(l_clf, train_data, kfs))
    print("LR mean: ", np.mean(l))
    print("LR std: ", np.std(l))

    r = (cv_performance(r_clf, train_data, kfs))
    print("LR mean: ", np.mean(r))
    print("LR std: ", np.std(r))

    print()
    print("With preprocessing: ")
    data = np.array(train_data.X)
    scaler = preprocessing.StandardScaler().fit(data)
    scaled_train_data = scaler.transform(data)
    scaled_train_data = Data(X=scaled_train_data, y=train_data.y)


    p = (cv_performance(p_clf, scaled_train_data, kfs))
    print("Perceptron mean: ", np.mean(p))
    print("Perceptron std: ", np.std(p))

    l = (cv_performance(l_clf, scaled_train_data, kfs))
    print("LR mean: ", np.mean(l))
    print("LR std: ", np.std(l))

    r = (cv_performance(r_clf, scaled_train_data, kfs))
    print("LR mean: ", np.mean(r))
    print("LR std: ", np.std(r))
    ### ========== TODO: END ========== ###


if __name__ == '__main__':
    main()
