"""
Provides OMP_GLSP algorithm class
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

from pyCSalgos.base import SparseSolver, ERCcheckMixin

import numpy
import scipy

class OMP_GLSP(SparseSolver, ERCcheckMixin):
    """
    Performs sparse coding via OMP_GLSP
    """

    # All parameters related to the algorithm itself are given here.
    # The data and dictionary are given to the run() method
    def __init__(self, stopval, fusionparam):

        # parameter check
        if stopval < 0:
            raise ValueError("stopping value is negative")

        if stopval < 1:
            self.stopcrit = StopCriterion.TOL
        else:
            self.stopcrit = StopCriterion.FIXED
        self.stopval = stopval
        self.fusionparam = fusionparam

    def __str__(self):
        return "OMP_GLSP ("+str(self.stopval)+","+str(self.fusionparam)+")"

    def solve(self, data, dictionary, realdict=None):
        return _omp_glsp(data, dictionary, self.stopval, self.fusionparam)

    def checkERC(self, acqumatrix, dictoper, support):

        raise NotImplementedError('checkERC() not implemented for solver OMP_GLSP')

class StopCriterion:
    """
    Stopping criterion type:
        StopCriterion.FIXED:    fixed number of iterations
        StopCriterion.TOL:      until approximation error is below tolerance
    """
    FIXED = 1
    TOL   = 2


def _omp_glsp(data, dictionary, stopval, fusionparam):
    """
    Hybrid combination of Greedy Least Squares Pursuit and Orthogonal Matching Pursuit algorithms
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
        
        # Solve for each vector data[i]

        # Prepare stuff
        y = data[:,i, None]  # Ensures 2D column vector, see https://stackoverflow.com/a/32764336
        N = dictionary.shape[1]
        T = numpy.array((), dtype=int)      # Set of selected atoms
        Tc = numpy.arange(N, dtype =int)    # Set of remaining atoms
        gamma = numpy.zeros((dictionary.shape[1],1)) # Initial gamma is all-zero
        
        # Iterations
        niter = 0
        finished = False
        while not finished:

            # Do one step of OMP_GLSP
            Tnew, gamma, residualnorm = Step_OMPGLSP_minL2res(y, dictionary, T, fusionparam)

            # Check if finished
            niter = niter + 1
            if (stopval < 1) and (residualnorm < stopval):
                finished = True
            elif (stopval > 1) and (niter == stopval):
                finished = True
            
            if not finished:
                # Update sets
                T = numpy.append(T, Tnew)
                Tc = numpy.setdiff1d(Tc, [Tnew], assume_unique=False)
            
            #print 'GLSP step %d: selected atom %d'%(niter, sel)
            
        #print 'GLSP final: T = %s'%(T)
        
        # Recompute gamma on support T, since our algorithm only preserved values in Tc
        gamma[T] = scipy.linalg.lstsq(dictionary[:,T], y)[0]
        gamma[Tc] = 0
        coef[:,i, None] = gamma
   
    return coef

def UpdateGLSP_WholeNS(y, D, T):
    """
    GLSP update step, non-efficient, redoes all projections
    """
    
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

def UpdateGLSPGAPSystem(y, D, T):
    Dpinv = scipy.linalg.pinv(D).T
    gammaorig = numpy.dot(Dpinv.T, y)        
    N = D.shape[1]
    I = numpy.delete(numpy.eye(N), T,0)
    augmat = numpy.concatenate((1e6*D, I))
    augy = numpy.concatenate((1e6*y, numpy.zeros(N - T.size)))
    gamma = scipy.linalg.lstsq(augmat, augy)[0]
    return gamma
    
#==================
# New:
#==================
def SelectAtomGLSP(gamma, T):
	"""
	Selects a new atom according to GLSP rule. 
	Input:
	 - gamma: the current (previous) estimate of solution
	 - T: the current support
    Return:
     - sel: index of atom to choose
	"""
	
	# Make a copy of gamma and zero the chosen atoms, 
	# so the selected atom preserves the original numbering
	gammatemp = gamma.copy()
	gammatemp[T] = 0    # skip already chosen elements
	sel = numpy.argmax(numpy.abs(gammatemp))
	return sel


def SelectAtomOMP(gamma, y, D):
	"""
	Selects a new atom according to OMP rule. 
	Input:
	 - gamma: the current (previous) estimate of solution
	 - D: the dictionary or its pseudoinverse
	 - y: the measurements vectpr
    Return:
     - sel: index of atom to choose     
	"""
	# Compute residual
	residual = y - numpy.dot(D, gamma)
	# Choose max of D' * residual
	sel = numpy.argmax(numpy.abs(numpy.dot(D.T, residual)))
	return sel
	
def UpdateOMP_WholeProj(y, D, T):
    """
    Updates the estimate solution via OMP style, non-efficient, whole projection
    """
    if T.size == 0:
        # Empty support, return zeros
        return numpy.zeros((D.shape[1],1))
    else:
        # Compute the support
        DT = D[:,T]
        DTpinv = scipy.linalg.pinv(DT)
        # Compute projection coefficients
        coeff = numpy.dot(DTpinv, y)
        # Compute gamma
        gamma = numpy.zeros((D.shape[1],1))
        gamma[T] = coeff
        return gamma
	
	
def Step_OMPGLSP_minL2res(y, D, T, fusionparam):
    """
	Performs one step of OMP_GLSP. 

    One step of the algorithm starts from a candidate support T^{k} and returns the new candidate support T^{k+1}. It consists of the following stages:
     1. With the input support T^{k}, compute both OMP and GLSP candidate solutions.
     2. Decide which solution one to use, based on a certain criterion.
     3. WIth the selected solution, compute the new candidate atom.
    The candidate solution gamma is just a by-product of the support. 
    It is only used to determine the next atom in the support.

    Inputs:
     y = the measurement vector
     D = the dictionary
     T = the candidate support at the current iteration
    Returns:
     Tnew = new selected atom index, to be included in the new support 
     gamma = the current solution for the input support T, computed either with  GLSP or OMP
     residualnorm = norm of residual (either OMP or GLSP), used for termination  condition
	"""
    
    # 1. Compute solutions with both algorithms
    gammaOMP  = UpdateOMP_WholeProj(y, D, T)
    gammaGLSP = UpdateGLSP_WholeNS(y, D, T)

    # Compute residuals, will be passed at output, needed for stopping criterion
    # OMP
    residualOMP = y - numpy.dot(D, gammaOMP)
    resnormOMP  = numpy.sqrt(numpy.sum(numpy.dot(residualOMP.T, residualOMP)))
    # GLSP
    gammacosupp = gammaGLSP.copy()
    gammacosupp[T] = 0 # ignore coefficients from support
    resnormGLSP = numpy.sqrt(numpy.sum(numpy.dot(gammacosupp.T, gammacosupp)))


    # 2. Choose one approach 
    #criterion = 'weightedsumnorm'  # Possible: 'minL2res', 'largestGapPerc'
                                  #      'weightedsumnorm'
    fusionmode = fusionparam['mode']

    if fusionmode == 'minL2res':
        # Criterion: Minimum L2 residual 
        # Select method with smallest residual
        useGLSP = (resnormGLSP < resnormOMP)
    
    elif fusionmode == 'largestGapPerc':
        # Criterion: Largest gap between 1st and 2nd candidate, in percentage
        # OMP
        residualOMP = y - numpy.dot(D, gammaOMP)
        innerprods = numpy.dot(D.T, residualOMP)
        innerprodsSort = numpy.sort(numpy.abs(innerprods), 0)
        OMPorder = numpy.argsort(numpy.abs(innerprods), 0)
        candidate1 = innerprodsSort[-1]
        candidate2 = innerprodsSort[-2]
        ratioOMP = candidate2/candidate1  # smaller than 1, smaller is better
        # GLSP
        gammacosupp = gammaGLSP.copy()
        gammacosupp[T] = 0 # ignore coefficients from resnormOMPsupport
        gammacosuppSort = numpy.sort(numpy.abs(gammacosupp), 0)
        GLSPorder = numpy.argsort(numpy.abs(gammacosupp), 0)
        candidate1 = gammacosuppSort[-1]
        candidate2 = gammacosuppSort[-2]
        ratioGLSP = candidate2/candidate1  # smaller than 1, smaller is better
        # Select method with smallest residual
        useGLSP = (ratioGLSP < ratioOMP)
        #if (OMPorder[-1] == GLSPorder[-1]):
        #    print('Iteration %d: agree'%(T.size+1))
        #else:
        #    print('Iteration %d: disagree'%(T.size+1))


    elif fusionmode == 'weightedsumnorm':
        fusionlambda = fusionparam['lambda']
        # OMP
        residualOMP = y - numpy.dot(D, gammaOMP)
        innerprods = numpy.dot(D.T, residualOMP)
        OMPnormalized = innerprods / numpy.sqrt(numpy.dot(innerprods.T, innerprods))
        # GLSP
        gammacosupp = gammaGLSP.copy()
        gammacosupp[T] = 0 # ignore coefficients from support
        GLSPnormalized = gammacosupp / numpy.sqrt(numpy.dot(gammacosupp.T, gammacosupp))
        # Weighted sum
        OMPGLSPweightedsum = fusionlambda*numpy.abs(OMPnormalized) + (1-fusionlambda)*numpy.abs(GLSPnormalized)
        # Choose max
        Tnew = numpy.argsort(OMPGLSPweightedsum, 0)[-1]
        # Compute outputs
        # DEBUG HACK
        gamma = gammaOMP
        residualnorm = resnormOMP
        return Tnew, gamma, residualnorm

    elif fusionmode == 'signedweightedsumnorm':
        fusionlambda = fusionparam['lambda']
        # OMP
        residualOMP = y - numpy.dot(D, gammaOMP)
        innerprods = numpy.dot(D.T, residualOMP)
        OMPnormalized = innerprods / numpy.sqrt(numpy.dot(innerprods.T, innerprods))
        # GLSP
        gammacosupp = gammaGLSP.copy()
        gammacosupp[T] = 0 # ignore coefficients from support
        GLSPnormalized = gammacosupp / numpy.sqrt(numpy.dot(gammacosupp.T, gammacosupp))
        # Weighted sum
        OMPGLSPweightedsum = fusionlambda*numpy.abs(OMPnormalized) + (1-fusionlambda)*numpy.abs(GLSPnormalized)
        # Choose max
        Tnew = numpy.argsort(OMPGLSPweightedsum, 0)[-1]
        # Compute outputs
        # DEBUG HACK
        gamma = gammaOMP
        residualnorm = resnormOMP
        return Tnew, gamma, residualnorm



    # 3. Go with the chosen approach
    if useGLSP:
		# Go with GLSP
        gamma = gammaGLSP
        Tnew = SelectAtomGLSP(gamma, T)
        residualnorm = resnormGLSP
    else:
		# Go with OMP
        gamma = gammaOMP
        Tnew  = SelectAtomOMP(gamma, y, D)
        residualnorm = resnormOMP
    return Tnew, gamma, residualnorm
		
