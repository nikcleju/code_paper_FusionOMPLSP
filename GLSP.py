"""
GLSP.py

Provides Greedy Least Squares Pursuit (GLSP)
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

from pyCSalgos.base import SparseSolver, ERCcheckMixin

import numpy
import scipy

class GreedyLeastSquares(ERCcheckMixin, SparseSolver):
    """
    Performs sparse coding via Greedy Least Squares (GLS)
    """

    # All parameters related to the algorithm itself are given here.
    # The data and dictionary are given to the run() method
    def __init__(self, stopval):

        # parameter check
        if stopval < 0:
            raise ValueError("stopping value is negative")

        if stopval < 1:
            self.stopcrit = StopCriterion.TOL
        else:
            self.stopcrit = StopCriterion.FIXED
        self.stopval = stopval

    def __str__(self):
        return "GLSP ("+str(self.stopval)+")"

    def solve(self, data, dictionary, realdict=None):
        return _greedy_least_squares(data, dictionary, self.stopval)

    def checkERC(self, acqumatrix, dictoper, support):

        raise NotImplementedError('checkERC() not implemented for solver GLSP')

        '''
        D = numpy.dot(acqumatrix, dictoper)

        # Should normalize here the dictionary or not?
        for i in range(D.shape[1]):
            D[:,i] = D[:,i] / numpy.linalg.norm((D[:,i],2))

        dict_size = dictoper.shape[1]
        k = support.shape[0]
        num_data = support.shape[1]
        results = numpy.zeros(num_data, dtype=bool)

        for i in range(num_data):
            T = support[:,i]
            Tc = numpy.setdiff1d(range(dict_size), T)
            A = numpy.dot(D[:,Tc].T, numpy.linalg.pinv(D[:,T].T))
            assert(A.shape == (dict_size-k, k))

            linf = numpy.max(numpy.sum(numpy.abs(A),1))
            if linf < 1:
                results[i] = True
            else:
                results[i] = False
        return results
        '''

class StopCriterion:
    """
    Stopping criterion type:
        StopCriterion.FIXED:    fixed number of iterations
        StopCriterion.TOL:      until approximation error is below tolerance
    """
    FIXED = 1
    TOL   = 2


def _greedy_least_squares(data, dictionary, stopval):
    """
    Greedy Least Squares Pursuit algorihtm
    :param data: 2D array containing the data to decompose, columnwise
    :param dictionary: dictionary containing the atoms, columnwise
    :param stopval: stopping criterion
    :return: coefficients
    """

    # parameter check
    if stopval < 0:
        raise ValueError("stopping value is negative")
    if stopval > dictionary.shape[0]:
        raise ValueError("stopping value > signal size")
    if stopval > dictionary.shape[1]:
        raise ValueError("stopping value > dictionary size")

    if len(data.shape) == 1:
        data = numpy.atleast_2d(data)
        if data.shape[0] < data.shape[1]:
            data = numpy.transpose(data)
    coef = numpy.zeros((dictionary.shape[1], data.shape[1]))

    # Prepare the pseudoinverse of the dictionary, used for all signals
    Dpinv_original = scipy.linalg.pinv(dictionary)    # tall matrix
    
    for i in range(data.shape[1]):
        
        # Solve GLSP for single vector data[i]
        y = data[:,i]
        N = dictionary.shape[1]
        T = numpy.array((), dtype=int)      # Set of selected atoms
        Tc = numpy.arange(N, dtype =int)    # Set of remaining atoms
        gamma = scipy.linalg.lstsq(dictionary,y)[0]
        Dpinv = Dpinv_original.copy()       # make fresh copy, Dpinv will be destroyed in the process
        
        niter = 0;
        finished = False
        while not finished:
            
            # 1.Select new atom
            # Make a copy of gamma and zero the chosen atoms, 
            # so the selected atom preserves the original numbering
            gammatemp = gamma.copy();
            gammatemp[T] = 0    # skip already chosen elements
            sel = numpy.argmax(numpy.abs(gammatemp))
            # Update sets
            T = numpy.append(T,sel)
            Tc = numpy.setdiff1d(Tc, [sel], assume_unique=False)
            
            #print 'GLSP step %d: selected atom %d'%(niter, sel)
            
            # 2. Update gamma
            # Compute new null space direction and make column vector 2D
            newNS = numpy.dot(Dpinv, dictionary[:,sel])
            newNS = numpy.atleast_2d(newNS)
            if newNS.shape[0] < newNS.shape[1]:
                newNS = newNS.T
            # Update gamma
            newNSProj = numpy.dot(newNS, newNS.T) / numpy.linalg.norm(newNS, 2)**2
            gamma = gamma - numpy.dot(newNSProj, gamma)
            #gamma = gamma - numpy.dot(newNS, numpy.dot(newNS.T, gamma)) / numpy.dot(newNS.T, newNS)
            #gamma = numpy.squeeze(gamma) # make vector again, for easier access to entries later
            # Update Dpinv (QR orthogonalization)
            Dpinv = Dpinv - numpy.dot(newNSProj, Dpinv)
            
            # Check against whole NS calculation
            #gamma1 = solveGammaWholeNS(y, dictionary, T)
            # Check against system solve GAP-style
            #gamma2 = solveGammaGAPSystem(y, dictionary, T)
            
            niter = niter + 1            
            
            # 3. Check termination conditions
            if (stopval < 1) and (numpy.linalg.norm(gamma[Tc],2) < stopval):
                finished = True
            elif (stopval > 1) and (niter == stopval):
                finished = True
         
        #print 'GLSP final: T = %s'%(T)
        
        # Recompute gamma on support T, since our algorithm only preserved values in Tc
        gamma[T] = scipy.linalg.lstsq(dictionary[:,T], y)[0]
        gamma[Tc] = 0
        coef[:,i] = gamma
    
    return coef

def solveGammaWholeNS(y, D, T):
    Dpinv = scipy.linalg.pinv(D).T
    gammaorig = numpy.dot(Dpinv.T, y)        
    if T.size != 0:
        Pc = numpy.dot(D[:,T], scipy.linalg.pinv(D[:,T]))
        Dpinv2 = numpy.dot(Pc, Dpinv)
        newNS = Dpinv2[:T.size,:].T
        gamma = gammaorig - numpy.dot(newNS, numpy.dot(numpy.linalg.pinv(newNS), gammaorig))
    else:
        #Dpinv2 = Dpinv
        gamma = gammaorig
    return gamma

def solveGammaGAPSystem(y, D, T):
    Dpinv = scipy.linalg.pinv(D).T
    gammaorig = numpy.dot(Dpinv.T, y)        
    N = D.shape[1]
    I = numpy.delete(numpy.eye(N), T,0)
    augmat = numpy.concatenate((1e6*D, I))
    augy = numpy.concatenate((1e6*y, numpy.zeros(N - T.size)))
    gamma = scipy.linalg.lstsq(augmat, augy)[0]
    return gamma
