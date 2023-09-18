import math
from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from torch import nn
from torch.nn.modules import upsampling
from torch.functional import F
from torch.optim import Adam
from torch.distributions.multivariate_normal import  MultivariateNormal
def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)
    nll=(target*torch.log(mu) + (1-target)*torch.log(1-mu)).sum(1)
    print(nll.shape)
    # log_likelihood_bernoulli
    return nll
    


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)
    """a = (z - mu) ** 2
    log_norm_constant = -0.5 * torch.log(torch.Tensor(2 * math.pi))
    log_p = -0.5 * (logvar + a / logvar.exp())
    log_p = (log_p + log_norm_constant)"""
    """c=[]
    for m in mu:
        normal_dist= (1/(torch.sqrt(2*math.pi*logvar**2))* torch.exp(-(1/2)*(z.subtract(m))/logvar**2)).sum(1)
        c.append(normal_dist)
    t=torch.Tensor(np.array(c))
    print(t.shape)"""
    #t=np.sum(-(np.log(np.sqrt(self.sigma_sq)) + (1 / 2) * np.log(2 * np.pi)) - (test_data - self.mu) ** 2/ (2 * self.sigma_sq), axis=1)
    #(0.5 * torch.pow((z - mu) / logvar.exp(), 2) + logvar + 0.5 * np.log(2 * np.pi))
    #t=(logvar + (1 / 2) * np.log(2 * np.pi) - (z - mu) ** 2/ (2 * logvar)).sum(1)
    #print(batch_size)
    #t=(1/(logvar * np.sqrt(2*np.pi)) * np.exp(-1/2*(z - mu)**2/logvar**2)).sum(1)
    #print(t.shape)
    #p = (1./torch.sqrt(2.0*np.pi*logvar) * torch.exp(- ((z-mu)**2) / (2.0 * logvar))).mean(1)
    cov1 = torch.stack([torch.diag(sigma) for sigma in torch.exp(logvar)])
    p = MultivariateNormal(mu, cov1)
    d=p.log_prob(z)
    print(d.shape)

    return d

def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    #print(y)
    batch_size = y.size(0)
    sample_size = y.size(1)
    ai, inds = torch.max(y,dim=1)    
    #print(ai.shape)
    #print(y-ai.unsqueeze(1))
    exp=torch.exp(y-ai.unsqueeze(1))
    #print(exp.shape)
    #print(ai)
    return torch.log((exp.mean(dim=1)))+ ai

def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.

    - accepts stacks of means, but only one S0 and S1

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    sigma_1=logvar_p
    sigma_2=logvar_q
    mu1=mu_p
    mu2=mu_q
    print(mu1)
    print(sigma_1)
    #sigma_diag_1 = np.eye(sigma_1.shape[0]) * sigma_1
	#sigma_diag_2 = np.eye(sigma_2.shape[0]) * sigma_2
    
    cov1 = torch.stack([torch.diag(sigma) for sigma in torch.exp(sigma_1)])
    p = MultivariateNormal(mu1, cov1)
    cov2 = torch.stack([torch.diag(sigma) for sigma in torch.exp(sigma_2)])
    q = MultivariateNormal(mu2, cov2)
    



    kl=torch.distributions.kl.kl_divergence(q,p)
    print(kl)


    return kl


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)



    sigma_1=logvar_p
    sigma_2=logvar_q
    mu1=mu_p
    mu2=mu_q


    cov1 = torch.stack([torch.diag(sigma) for sigma in torch.exp(sigma_1)])
    p = MultivariateNormal(mu1, cov1)
    cov2 = torch.stack([torch.diag(sigma) for sigma in torch.exp(sigma_2)])
    q = MultivariateNormal(mu2, cov2)
    num_samples=torch.Tensor(np.array([num_samples]))
    q.rsample((num_samples,))
    
    # kld
    return q