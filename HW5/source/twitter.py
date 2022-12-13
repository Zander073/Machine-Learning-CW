"""
Author: Zander Zemliak
Date: November 1, 2022
Description: Machine learning PS4
"""

from re import I
import numpy as np

from string import punctuation

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle


######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.

    Parameters
    --------------------
        fname  -- string, filename

    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.

    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """

    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    np.savetxt(outfile, vec)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.

    Parameters
    --------------------
        infile    -- string, filename

    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """

    word_list = {}
    with open(infile, 'r') as fid:
        ### ========== TODO: START ========== ###
        # part 1-1: process each line to populate word_list
        tweets = fid.read().replace('\n', ' ')
        tweet_list = extract_words(tweets)
        index = 0
        for word in tweet_list:
            if word not in word_list:
                word_list[word] = index
                index += 1
        ### ========== TODO: END ========== ###
    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.

    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)

    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """

    num_lines = sum(1 for line in open(infile,'r'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    words = []

    with open(infile, 'r') as fid:
        ### ========== TODO: START ========== ###
        # part 1-2: process each line to populate feature_matrix
        i = 0
        for tweet in fid:
            tweet_list = extract_words(tweet)
            for word in tweet_list:
                if word not in words:
                    words.append(word)
                j = word_list.get(word)
                feature_matrix[i, j] = 1
            i += 1
        ### ========== TODO: END ========== ###
    return feature_matrix,words


def test_extract_dictionary(dictionary):
    err = 'extract_dictionary implementation incorrect'

    assert len(dictionary) == 1811

    exp = [('2012', 0),
           ('carol', 10),
           ('ve', 20),
           ('scary', 30),
           ('vacation', 40),
           ('just', 50),
           ('excited', 60),
           ('no', 70),
           ('cinema', 80),
           ('frm', 90)]
    act = [sorted(dictionary.items(), key=lambda it: it[1])[i] for i in range(0, 100, 10)]
    assert exp == act


def test_extract_feature_vectors(X):
    err = 'extract_features_vectors implementation incorrect'

    assert X.shape == (630, 1811)

    exp = np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                    [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    act = X[:10,:10]
    assert (exp == act).all()


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric='accuracy'):
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'

    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 # map points of hyperplane to +1

    ### ========== TODO: START ========== ###
    # part 2-1: compute classifier performance with sklearn metrics
    # hint: sensitivity == recall
    # hint: use confusion matrix for specificity (use the labels param)
    score = 0
    if metric == 'accuracy':
        score = metrics.accuracy_score(y_true,y_label)
    elif metric == 'f1_score':
        score = metrics.f1_score(y_true, y_label)
    elif metric == 'auroc':
        score = metrics.roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        score = metrics.auc(fpr, tpr)
    elif metric == 'precision':
        score = metrics.precision_score(y_true, y_label)
    elif metric == 'sensitivity':
        conf_matrix = metrics.confusion_matrix(y_true, y_label)
        score = conf_matrix[1,1]/float((conf_matrix[1,1]+conf_matrix[1, 0]))
    elif metric == 'specificity':
        conf_matrix = metrics.confusion_matrix(y_true, y_label)
        score = conf_matrix[0,0]/float((conf_matrix[0,0]+conf_matrix[0,1]))
    else:
        print("Error with inputted metric function.")
    return score
    ### ========== TODO: END ========== ###


def test_performance():
    """Ensures performance scores are within epsilon of correct scores."""

    y_true = [ 1,  1, -1,  1, -1, -1, -1,  1,  1,  1]
    y_pred = [ 3.21288618, -1.72798696,  3.36205116, -5.40113156,  6.15356672,
               2.73636929, -6.55612296, -4.79228264,  8.30639981, -0.74368981]
    metrics = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']
    scores  = [     3/10.,      4/11.,   5/12.,        2/5.,          2/6.,          1/4.]

    import sys
    eps = sys.float_info.epsilon

    for i, metric in enumerate(metrics):
        assert abs(performance(y_true, y_pred, metric) - scores[i]) < eps, \
            (metric, performance(y_true, y_pred, metric), scores[i])


def cv_performance(clf, X, y, kf, metric='accuracy'):
    """
    Splits the data, X and y, into k folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    scores = []
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make "continuous-valued" predictions
        y_pred = clf.decision_function(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score):
            scores.append(score)
    return np.array(scores).mean()


def select_param_linear(X, y, kf, metric):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that maximizes the average k-fold CV performance.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """

    print('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)

    ### ========== TODO: START ========== ###
    # part 2-3: select optimal hyperparameter using cross-validation
    # hint: create a new sklearn linear SVC for each value of C
    scores = []
    for c in C_range:
        clf = SVC(C=c,kernel="linear") #we want to find the best c value
        score = cv_performance(clf,X,y,kf,metric = metric)
        scores.append(score)
    print("Scores are, ", scores)
    max_index = scores.index(max(scores))
    return C_range[max_index]
    ### ========== TODO: END ========== ###


def select_param_rbf(X, y, kf, metric='accuracy'):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.

    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metric  -- string, option used to select performance measure

    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """

    print('RBF SVM Hyperparameter Selection based on ' + str(metric) + ':')

    ### ========== TODO: START ========== ###
    # (Optional) part 3-1: create grid, then select optimal hyperparameters using cross-validation
    return 0.0, 1.0
    ### ========== TODO: END ========== ###


def performance_CI(clf, X, y, metric='accuracy'):
    """
    Estimates the performance of the classifier using the 95% CI.

    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure

    Returns
    --------------------
        score        -- float, classifier performance
        lower, upper -- tuple of floats, confidence interval
    """

    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric)


    ### ========== TODO: START ========== ###
    # part 4-2: use bootstrapping to compute 95% confidence interval
    # hint: use np.random.randint(...) to get a random sample from y
    # hint: lower and upper are the values at 2.5% and 97.5% of the scores
    n,d = X.shape
    max_t = 1000
    t = 0
    scores = []

    while t < max_t:
        idx = np.random.randint(X.shape[0], size=int(n/2))
        sample_X = X[idx,:]
        sample_y = y[idx]
        sample_y_pred = clf.predict(sample_X)
        scores.append(performance(sample_y, sample_y_pred, metric))
        t += 1
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return score, lower, upper
    ### ========== TODO: END ========== ###


######################################################################
# main
######################################################################

def main():
    # read the tweets and its labels
    dictionary = extract_dictionary('../data/tweets.txt')
    test_extract_dictionary(dictionary)
    # Turned this into a tuple that returns the vector AND a list of words in order
    input_data = extract_feature_vectors('../data/tweets.txt', dictionary)
    X = input_data[0]
    feature_list = np.array(input_data[1])
    test_extract_feature_vectors(X)
    y = read_vector_file('../data/labels.txt')

    # shuffle data (since file has tweets ordered by movie)
    X, y = shuffle(X, y, random_state=0)

    # set random seed
    np.random.seed(1234)

    # split the data into training (training + cross-validation) and testing set
    X_train, X_test = X[:560], X[560:]
    y_train, y_test = y[:560], y[560:]

    metric_list = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']

    ### ========== TODO: START ========== ###
    test_performance()

    # part 2-2: create stratified folds (5-fold CV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    # part 2-4: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    # Uncomment to view performance across all metrics for a range of C values:
    for metric in metric_list:
        select_param_linear(X, y, skf, metric)  

    # (Optional) part 3-2: for each metric, select optimal hyperparameter for RBF-SVM using CV

    # part 4-1: train linear-kernal SVM with selected hyperparameters
    linear_SVM = SVC(C=1.e+02, kernel='linear')
    linear_SVM.fit(X_train, y_train)
    # part 4-3: use bootstrapping to report performance on test data
    print("Score - 2.5 Perc. - 97.5 Perc.")
    for metric in metric_list:
        print("Metric type: ", metric)
        print(performance_CI(linear_SVM, X_test, y_test, metric=metric))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # part 5: identify important features (hint: use best_clf.coef_[0])
    coef = linear_SVM.coef_[0]

    pos_ind = np.argpartition(coef, -10)[-10:]
    neg_ind = np.argpartition(coef, 10)[:10]

    for i in pos_ind:
        print(feature_list[i], " - ", coef[i])

    print("~~~~~~~~~~~~~~~~~~~")

    for i in neg_ind:
        print(feature_list[i], " - ", coef[i])


    ### ========== TODO: END ========== ###

    ### ========== TODO: START ========== ###
    # part 6: (optional) contest!
    # uncomment out the following, and be sure to change the filename
    """
    X_held = extract_feature_vectors('../data/held_out_tweets.txt', dictionary)
    y_pred = best_clf.decision_function(X_held)
    write_label_answer(y_pred, '../data/YOUR_USERNAME_twitter.txt')
    """
    ### ========== TODO: END ========== ###


if __name__ == '__main__':
    main()
