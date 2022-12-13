"""
Author     :
Date       :
Description: Polynomial Regression
This code was adapted from course material by Jenna Wiens (UMichigan).
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import time


######################################################################
# classes
######################################################################

class Data:

    def __init__(self, X=None, y=None):
        """
        Data class.

        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y


    def load(self, filename):
        """
        Load csv file into X array of features and y array of labels.

        Parameters
        --------------------
            filename -- string, filename
        """

        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '..', 'data', filename)

        # load data
        with open(f, 'r') as fid:
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]


    def plot(self, **kwargs):
        """Plot data."""

        if 'color' not in kwargs:
            kwargs['color'] = 'b'

        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()


# wrapper functions around Data class
def load_data(filename):
    data = Data()
    data.load(filename)
    return data


def plot_data(X, y, **kwargs):
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression():

    def __init__(self, m=1, reg_param=0):
        """
        Ordinary least squares regression.

        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param


    def generate_polynomial_features(self, X):
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].

        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features

        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """

        n,d = X.shape

        ### ========== TODO: START ========== ###
        # part 2: modify to create matrix for simple linear model
        # hint: use np.ones(), np.append(), and np.power()
        # part 7: modify to create matrix for polynomial model
        Phi = X
        if self.m_ == 1:
            Phi = np.ones((n, 2))
            for i in range(n):
                Phi[i, 1] = X[i]
        else:
            Phi = np.ones((n, self.m_ + 1))
            p = np.arange(0, self.m_ + 1)
            for index, row in enumerate(Phi):
                row = np.repeat(X[index], self.m_ + 1)
                row = np.power(row, p)
                Phi[index] = row
        ### ========== TODO: END ========== ###

        return Phi


    def fit_SGD(self, X, y, eta=0.01,
                eps=1e-10, tmax=1000000, verbose=False):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares stochastic gradient descent.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size (also known as alpha)
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes

        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0:
            raise Exception('SGD with regularization not implemented')

        if verbose:
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()

        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration

        # SGD loop
        for t in range(tmax):
            ### ========== TODO: START ========== ###
            # part 6: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None:
                eta = None
            else:
                eta = eta_input
            ### ========== TODO: END ========== ###

            # iterate through examples
            for i in range(n):
                ### ========== TODO: START ========== ###
                phi = X[i,:]
                y_pred = np.dot(self.coef_, phi)
                # part 4: update theta (self.coef_) using one step of SGD
                # hint: you can simultaneously update all theta via vector math
                self.coef_ = self.coef_ - eta * (y_pred - y[i]) * X[i] 
                # track error
                # TODO: predict y using updated theta
                err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)
                ### ========== TODO: END ========== ###

            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) < eps:
                break

            # debugging
            if verbose:
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec

        print('number of iterations: %d' % (t+1))

        return self


    def fit(self, X, y):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            self    -- an instance of self
        """

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO: START ========== ###
        # part 5: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        # hint: X.T is the transpose of X
        # be sure to update self.coef_ with your solution
        # part 10: add the regularization term
        XXT = np.linalg.pinv(np.dot(X.T, X))
        self.coef_ = np.dot(XXT, np.dot(X.T,y))
        ### ========== TODO: END ========== ###


    def predict(self, X):
        """
        Predict output for X.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features

        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None:
            raise Exception('Model not initialized. Perform a fit first.')

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO: START ========== ###
        # part 3: predict y (hint: X times theta)
        y = np.dot(X, self.coef_)
        ### ========== TODO: END ========== ###

        return y


    def cost(self, X, y):
        """
        Calculates the objective function.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO: START ========== ###
        # part 4: get predicted y, then compute J(theta)
        X = self.generate_polynomial_features(X)
        y_vector  = np.dot(X, self.coef_)
        cost = (np.dot((y - y_vector).transpose(), (y - y_vector))) * 0.5
        ### ========== TODO: END ========== ###
        return cost


    def rms_error(self, X, y):
        """
        Calculates the root mean square error.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO: START ========== ###
        # part 8: compute RMSE
        n,d = X.shape
        error = np.sqrt((2 * self.cost(X, y))/n)
        ### ========== TODO: END ========== ###
        return error


    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs):
        """Plot regression line."""
        if 'color' not in kwargs:
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = '-'

        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


######################################################################
# main
######################################################################

def main():
    # toy data
    X = np.array([2]).reshape((1,1))          # shape (n,d) = (1L,1L)
    y = np.array([3]).reshape((1,))           # shape (n,) = (1L,)
    coef = np.array([4,5]).reshape((2,))      # shape (d+1,) = (2L,), 1 extra for bias

    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')

    ### ========== TODO: START ========== ###
    # part 1: plot train and test data
    print('Visualizing data...')
    plt.scatter(train_data.X, train_data.y, label = "Train")
    plt.scatter(test_data.X, test_data.y, label = "Test")
    plt.xlabel("X, features")
    plt.ylabel("Y, labels")
    plt.title("Features vs Labels for train and test data")
    plt.legend(loc='upper right')
    plt.show()
    ### ========== TODO: END ========== ###


    ### ========== TODO: START ========== ###
    # parts 2-6: main code for linear regression
    print('Investigating linear regression...')

    # model
    model = PolynomialRegression()

    # test part 2 -- soln: [[1 2]]
    print(model.generate_polynomial_features(X))
    print()

    # test part 3 -- soln: [14]
    model.coef_ = coef
    print(model.predict(X))
    print()

    # test part 4, bullet 1 -- soln: 60.5
    print(model.cost(X, y))
    print()

    # test part 4, bullets 2-3
    # for eta = 0.01, soln: theta = [2.44; -2.82], iterations = 616  
    model.fit_SGD(train_data.X, train_data.y, 0.01)
    print('sgd solution: %s' % str(model.coef_))
    print()

    # test part 5 -- soln: theta = [2.45; -2.82]
    model.fit(train_data.X, train_data.y)
    print('closed_form solution: %s' % str(model.coef_))
    print()

    # non-test code (YOUR CODE HERE)

    ### ========== TODO: END ========== ###
    start = time.time()
    model.fit_SGD(train_data.X, train_data.y, 0.015)
    end = time.time()
    print('sgd solution: %s' % str(model.coef_))
    print(end - start)

    ### ========== TODO: START ========== ###
    # parts 7-9: main code for polynomial regression
    print('Investigating polynomial regression...')

    # toy data
    m = 2                                     # polynomial degree
    coefm = np.array([4,5,6]).reshape((3,))   # shape (3L,), 1 bias + 3 coeffs

    # test part 7 -- soln: [[1 2 4]]
    model = PolynomialRegression(m)
    print(model.generate_polynomial_features(X))

    # test part 8 -- soln: 35.0
    model.coef_ = coefm
    print(model.rms_error(X, y))

    # non-test code (YOUR CODE HERE)

    ### ========== TODO: END ========== ###


    ### ========== TODO: START ========== ###
    # parts 10-11: main code for regularized regression
    print('Investigating regularized regression...')

    # test part 10 -- soln should be close to [3 0 0]
    # note: your solution may be slightly different
    #       due to limitations in floating point representation
    model = PolynomialRegression(m=2, reg_param=1e-5)
    model.fit(X, y)
    print(model.coef_)

    # non-test code (YOUR CODE HERE)

    ### ========== TODO: END ========== ###

    print('Done!')


if __name__ == "__main__":
    main()
