"""
Apply Gradient descent for the inexact linear solver
"""
import numpy as np
from scipy import optimize

def linear_inexact_solve(A,B,v,beta):
    """ On every iteration
    denote w_t = v
    and w_t+1 = u
    and initialisation x_0 = beta*v
    """
    args = (A,B,v)
    new_w = optimize.fmin_cg(cost, beta*v, fprime=grad, args=args, disp=False)
    return new_w

def cost(u, *args):
    """
    Compute the cost for
    the cost function
    ||u-B^-1Av||_B
    """
    A,B,v = args
    cost_value = 0.5*u.T@B@u - u.T@A@v
    return cost_value

def grad(u, *args):
    """
    Compute gradient for the cost function
    f(u) = ||u-B^-1Av||_B
    as del(f(u)) = Bu - Av
    """
    A,B,v = args
    gradient = B@u - A@v
    return gradient
