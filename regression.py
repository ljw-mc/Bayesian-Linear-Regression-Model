import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    X, X_T = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

    mu = np.array([0,0]) # zero mean given in Q
    cov = np.array([[beta, 0], [0, beta]])
    data_points = np.column_stack((X.ravel(), X_T.ravel()))

    Prior = util.density_Gaussian(mu, cov, data_points).reshape(100, 100)

    plt.figure()
    plt.plot([-0.1], [-0.5], marker='x')
    plt.annotate('true value of a', xy=(-0.1, -0.5), xytext=(-0.8, 0.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=9, ha='center', va='center')
    plt.contour(X, X_T, Prior)
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.title(f'Prior distribution of vector a = [a0, a1]^T, beta={beta}')
    plt.savefig("P(a)")
    # plt.show()
    
    return None
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    such that (x, z) forms a data point
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    X = np.column_stack((np.ones_like(x), x))
    cov_a = np.array(np.eye(2) / beta)
    S = np.array(np.linalg.inv(cov_a + (1/sigma2)*X.T@X))

    mu = ((1/sigma2)*S@X.T@z).flatten()


    X, X_T = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    data_points = np.column_stack((X.ravel(), X_T.ravel()))
    density = util.density_Gaussian(mu, S, data_points).reshape(100, 100)

    plt.figure()
    plt.plot([-0.1], [-0.5], marker='x')
    plt.annotate('true value of a', xy=(-0.1, -0.5), xytext=(-0.8, 0.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=9, ha='center', va='center')
    plt.contour(X, X_T, density)
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title(f'A Posteriori N={len(x)}')
    plt.savefig(f'A Posteriori N={len(x)}')
    plt.show()
    
    
    return mu, S

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    # Calculate the predictive mean and variance
    x_pred = np.column_stack((np.ones_like(x), x))
    new_mu = x_pred @ mu
    new_cov = sigma2 + x_pred @ Cov @ x_pred.T



    plt.scatter(x_train, z_train, label="training data")
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.errorbar(x, new_mu, yerr=np.sqrt(np.diag(new_cov)), fmt = "rx", label="predictions w/ error bars")
    plt.legend()
    plt.title(f'Prediction with N={len(x_train)}')
    plt.savefig(f'Prediction with N={len(x_train)}')
    plt.show()
    
    
    return None

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 1
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)

    ns  = 5
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)

    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
