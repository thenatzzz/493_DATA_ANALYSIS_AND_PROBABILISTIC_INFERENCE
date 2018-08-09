import numpy as np
from scipy.optimize import minimize
import math
from numpy import linalg
import random

# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:,:-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:,-1])[:, None]
    X = (Xraw-Xraw.mean(axis=0))/np.std(Xraw, axis=0)
    y = (yraw-yraw.mean(axis=0))/np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4] # take every fourth observation for the test set
    train_ind = list(set(ind)-set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]

    return X_train, y_train, X_test, y_test

# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    sample = np.zeros((mean.shape[0], )) # This is only a placeholder
    # Task 2:S
    # TODO: Implement a draw from a multivariate Gaussian here
    dimension = cov.shape[0]

    X = np.random.multivariate_normal(np.zeros(dimension),np.eye(dimension))
    sample = np.dot(np.linalg.cholesky(cov),X) + mean

    # Return drawn sample
    return sample

# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
# x = np.random.multivariate_normal((1,2),[[1,0],[0,1]],(1,10))
#
# print(x)
# print(x.shape)

class RadialBasisFunction():
    def __init__(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################

    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X=X_aug

        n = X.shape[0]
        covMat = np.zeros((n,n))

        # Task 1:
        # TODO: Implement the covariance matrix here

        # k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2))
        for index_row in range(covMat.shape[0]):
            for index_col in range(covMat.shape[0]):
                exp_part1 = (-1)*((linalg.norm(X[index_row]-X[index_col]))**2)
                exp_part2 = (1/(2*(self.length_scale**2)))
                result = self.sigma2_f * math.exp(exp_part1*exp_part2)
                covMat[index_row][index_col] = result
        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrixm
        return covMat
# print(pow(4,2))

class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        return K

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        # Task 3:
        # TODO: compute the mean and covariance of the prediction
        prior_mean = 0
        sigma2_n = 0

        kernel_matrix= self.k.covMatrix(self.X, Xa)

        kernel_xs_x = np.copy(kernel_matrix[self.X.shape[0]:,:self.X.shape[0]])
        inverse_kernel_x = np.linalg.inv(self.K)
        mean_fa = prior_mean + np.dot(np.dot(kernel_xs_x.T, \
                                        inverse_kernel_x+sigma2_n),self.y)


        kernel_xs_xs = np.copy(kernel_matrix[self.X.shape[0]:, self.X.shape[0]:])
        kernel_x_xs = np.copy(kernel_matrix[:self.X.shape[0], self.X.shape[0]:])
        #Need to delete excess noise
        cov_fa_1 = kernel_xs_xs-(self.k.sigma2_n*np.identity(kernel_xs_xs.shape[0]))
        cov_fa_2 =  np.dot(np.dot(kernel_xs_x.T,inverse_kernel_x),kernel_xs_x)
        cov_fa = cov_fa_1 - cov_fa_2
        # Return the mean and covariance
        return mean_fa, cov_fa

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        mll = 0
        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y
        inverse_k = np.linalg.inv(self.K)

        first_term = (1.0/2)*np.dot(np.dot(self.y.T,inverse_k),self.y)
        second_term = (1.0/2)*np.linalg.slogdet(self.K)[1]
        constant  = (self.n/2.0)*np.log(2.0*np.pi)

        mll = first_term + second_term + constant
        # Return mll
        return mll[0]

    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def calculateGradLogMagLikelihood(self,alpha,k_alpha_inv, grad_k_alpha):
        result  = -0.5*np.trace(np.dot(np.dot(alpha,alpha.T)-k_alpha_inv,grad_k_alpha))
        return result

    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        grad_ln_sigma_f = grad_ln_length_scale = grad_ln_sigma_n = 0
        # Task 5:
        # TODO: calculate the gradients of the negative log marginal likelihood
        # wrt. the hyperparameters
        k_alpha_inv = np.linalg.inv(self.K)
        alpha = np.dot(k_alpha_inv,self.y)

        grad_k_ln_alpha_f = np.zeros((self.n,self.n))
        grad_k_ln_length_scale = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                term_1 = -0.5*(np.linalg.norm(self.X[i]-self.X[j])**2)/(self.k.length_scale**2)
                term_2 = self.k.sigma2_f*np.exp(term_1)

                grad_k_ln_alpha_f[i][j] = 2.0*term_2
                grad_k_ln_length_scale[i][j] = (-2.0)*term_1*term_2
        grad_k_ln_alpha_n = 2*self.k.sigma2_n * np.identity(self.n)

        grad_ln_sigma_f= self.calculateGradLogMagLikelihood(alpha,k_alpha_inv,grad_k_ln_alpha_f)
        grad_ln_length_scale= self.calculateGradLogMagLikelihood(alpha,k_alpha_inv,grad_k_ln_length_scale)
        grad_ln_sigma_n= self.calculateGradLogMagLikelihood(alpha,k_alpha_inv,grad_k_ln_alpha_n)

        # Combine gradients
        gradients = np.array([grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])

        # Return the gradients
        return gradients


    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        mse = 0
        # Task 7:
        # TODO: Implement the MSE between ya and fbar
        mse = np.sum((ya-fbar)**2.0)/ya.shape[0]

        # Return mse
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya
        predictive_var = np.diagonal(cov + self.k.sigma2_n).reshape((cov.shape[0],1))

        term_1 = 0.5 * np.log(2*math.pi*predictive_var)
        term_2 = ((ya-fbar)**2)/(2.0*predictive_var)
        msll = np.sum(term_1+term_2)/ya.shape[0]

        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method ='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
        return res.x

if __name__ == '__main__':

    np.random.seed(42)

    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################
