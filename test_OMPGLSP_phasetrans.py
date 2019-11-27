__author__ = 'Nic'

#import mkl
#mkl.set

import numpy
import scipy
import datetime

import sys, socket
hostname = socket.gethostname()
if hostname == 'caraiman':
    pyCSalgos_path = '/home/nic/code/pyCSalgos'
elif hostname == 'nclejupchp':
    pyCSalgos_path = '/home/ncleju/Work/code/pyCSalgos'

sys.path.insert(0,pyCSalgos_path)
sys.path.insert(0,'..')
#sys.path.insert(0,'/home/ncleju/code/pyCSalgos')
#sys.path.append('D:\\Facultate\\Code\\pyCSalgos')
#sys.path.append('D:\\Facultate\\Code\\code_analysisbysynthesis')

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#from pyCSalgos import AnalysisPhaseTransition
#from pyCSalgos import UnconstrainedAnalysisPursuit
#from pyCSalgos import AnalysisBySynthesis
#from pyCSalgos import OrthogonalMatchingPursuit
from OMP_GLSP import OMP_GLSP

from GLSP import GreedyLeastSquares
from pyCSalgos import SynthesisPhaseTransition
from pyCSalgos import OrthogonalMatchingPursuit


def run():
    #=======================================================
    # PARAMETERS
    #=======================================================
    ### Square 200, exact, 2 signals each
    # runname = 'fig_exact_OMPGLSP_test1_square'  # Name 
    # signal_size, dict_size = 200, 200           # Signal dimensions
    # deltas = numpy.arange(0.1, 1, 0.05)         # Phase transition grid
    # rhos = numpy.arange(0.1, 1, 0.05)
    # snr_db =  numpy.Inf                         # SNR ratio
    # numdata = 2                                 # Number of signals to average 
    # solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, 'largestGapPerc')]
    # #solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR')]
    # solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    # success_thresh = 1e-6                       # Threshold for considering successful recovery
    # save_folder = 'save'                        # Where to save


    ### Square 200, exact, 2 signals each
    # runname = 'fig_exact_OMPGLSP_test5_square'  # Name 
    # signal_size, dict_size = 100, 150           # Signal dimensions
    # deltas = numpy.arange(0.1, 1, 0.1)         # Phase transition grid
    # rhos = numpy.arange(0.1, 1, 0.1)
    # snr_db =  numpy.Inf                         # SNR ratio
    # numdata = 10                                 # Number of signals to average 
    # solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, 'weightedsumnorm')]
    # #solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR')]
    # solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    # success_thresh = 1e-6                       # Threshold for considering successful recovery
    # save_folder = 'save'                        # Where to save



    #deltas = numpy.array([0.5, 0.9])
    #rhos = numpy.array([0.2, 0.5])
    
    #    ### 100x100, exact, 2 signals each, 0.1 steps
    #    runname = 'fig_exact_OMPGLSP_test2'  # Name 
    #    signal_size, dict_size = 100, 100           # Signal dimensions
    #    deltas = numpy.arange(0.1, 1, 0.1)          # Phase transition grid
    #    rhos = numpy.arange(0.1, 1, 0.1)
    #    snr_db =  numpy.Inf                         # SNR ratio
    #    numdata = 2                                 # Number of signals to average 
    #    solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, 'largestGapPerc')]
    #    solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    #    success_thresh = 1e-9                       # Threshold for considering successful recovery
    #    save_folder = 'save'              #    ### 100x100, exact, 2 signals each, 0.1 steps
    #    runname = 'fig_exact_OMPGLSP_test2'  # Name 
    #    signal_size, dict_size = 100, 100           # Signal dimensions
    #    deltas = numpy.arange(0.1, 1, 0.1)          # Phase transition grid
    #    rhos = numpy.arange(0.1, 1, 0.1)
    #    snr_db =  numpy.Inf                         # SNR ratio
    #    numdata = 2                                 # Number of signals to average 
    #    solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, 'largestGapPerc')]
    #    solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    #    success_thresh = 1e-9                       # Threshold for considering successful recovery
    #    save_folder = 'save'                        # Where to save
    #    
    #    ### 100x130, exact, 2 signals each, 0.1 steps
    #    runname = 'fig_exact_OMPGLSP_test3'  # Name 
    #    signal_size, dict_size = 100, 130           # Signal dimensions
    #    deltas = numpy.arange(0.1, 1, 0.1)          # Phase transition grid
    #    rhos = numpy.arange(0.1, 1, 0.1)
    #    snr_db =  numpy.Inf                         # SNR ratio
    #    numdata = 2                                 # Number of signals to average 
    #    solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, 'largestGapPerc')]
    #    solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    #    success_thresh = 1e-9                       # Threshold for considering successful recovery
    #    save_folder = 'save'                        # Where to save
    #
    #    ### 100x130, approx, 2 signals each, 0.1 steps
    #    runname = 'fig_exact_OMPGLSP_test4'  # Name 
    #    signal_size, dict_size = 100, 130           # Signal dimensions
    #    deltas = numpy.arange(0.1, 1, 0.1)          # Phase transition grid
    #    rhos = numpy.arange(0.1, 1, 0.1)
    #    snr_db =  40                         # SNR ratio
    #    numdata = 5                                 # Number of signals to average 
    #    solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, 'largestGapPerc')]
    #    solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    #    success_thresh = None                       # Threshold for considering successful recovery
    #    save_folder = 'save'                        # Where to save
          # Where to save
    #    
    #    ### 100x130, exact, 2 signals each, 0.1 steps
    #    runname = 'fig_exact_OMPGLSP_test3'  # Name 
    #    signal_size, dict_size = 100, 130           # Signal dimensions
    #    deltas = numpy.arange(0.1, 1, 0.1)          # Phase transition grid
    #    rhos = numpy.arange(0.1, 1, 0.1)
    #    snr_db =  numpy.Inf                         # SNR ratio
    #    numdata = 2                                 # Number of signals to average 
    #    solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, 'largestGapPerc')]
    #    solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    #    success_thresh = 1e-9                       # Threshold for considering successful recovery
    #    save_folder = 'save'                        # Where to save
    #
    #    ### 100x130, approx, 2 signals each, 0.1 steps
    #    runname = 'fig_exact_OMPGLSP_test4'  # Name 
    #    signal_size, dict_size = 100, 130           # Signal dimensions
    #    deltas = numpy.arange(0.1, 1, 0.1)          # Phase transition grid
    #    rhos = numpy.arange(0.1, 1, 0.1)
    #    snr_db =  40                         # SNR ratio
    #    numdata = 5                                 # Number of signals to average 
    #    solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, 'largestGapPerc')]
    #    solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    #    success_thresh = None                       # Threshold for considering successful recovery
    #    save_folder = 'save'                        # Where to save


    # ######################################################
    # # Ill-conditioned dictionaries

    # ### Square 200, exact, 2 signals each
    # runname = 'fig_exact_OMPGLSP_UnnormDict100'  # Name 
    # signal_size, dict_size = 50, 100           # Signal dimensions
    # deltas = numpy.arange(0.1, 1, 0.1)         # Phase transition grid
    # rhos = numpy.arange(0.1, 1, 0.1)
    # snr_db =  numpy.Inf                         # SNR ratio
    # #snr_db =  60                         # SNR ratio
    # numdata = 30                                 # Number of signals to average 
    # solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, 'weightedsumnorm'), OMP_GLSP(1e-6, 'largestGapPerc')]
    # #solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR')]
    # solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    # success_thresh = 1e-6                       # Threshold for considering successful recovery
    # save_folder = 'save'                        # Where to save
    # numpy.random.seed(1234)
    # dictionary = numpy.random.randn(signal_size, dict_size)
    # [U,S,Vt] = numpy.linalg.svd(dictionary, full_matrices=False)
    # A = 5
    # RCconst = 0.07
    # S = 5 * numpy.exp(-RCconst * numpy.arange(Vt.shape[0]))
    # plt.plot(S)
    # plt.show()
    # plt.savefig('save/' + runname + '_spectrum' + '.' + 'pdf', bbox_inches='tight')
    # plt.savefig('save/' + runname + '_spectrum' + '.' + 'png', bbox_inches='tight')
    # dictionary = U @ numpy.diag(S) @ Vt
    # for i in range(dictionary.shape[1]):
    #     #dictionary[:,i] = S[i] * dictionary[:,i] / numpy.sqrt(numpy.sum( dictionary[:,i]**2 ))  # normalize columns
    #     dictionary[:,i] = dictionary[:,i] / numpy.sqrt(numpy.sum( dictionary[:,i]**2 ))  # normalize columns
    # acqumatrix     = "randn"


    # ######################################################
    # # Special low-spark dictionaries
    # runname = 'fig_exact_OMPGLSP_LowSpark100x150'  # Name 
    # signal_size, dict_size = 100, 150           # Signal dimensions
    # deltas = numpy.arange(0.1, 1, 0.1)         # Phase transition grid
    # rhos = numpy.arange(0.1, 1, 0.1)
    # snr_db =  numpy.Inf                         # SNR ratio
    # #snr_db =  60                         # SNR ratio
    # numdata = 2                                 # Number of signals to average 
    # solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, 'weightedsumnorm'), OMP_GLSP(1e-6, 'largestGapPerc')]
    # #solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR')]
    # solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    # success_thresh = 1e-6                       # Threshold for considering successful recovery
    # save_folder = 'save'                        # Where to save
    # numpy.random.seed(1234)
    # #dictionary = numpy.random.randn(signal_size, dict_size)
    # #[U,S,Vt] = numpy.linalg.svd(dictionary, full_matrices=False)
    # #A = 5
    # #RCconst = 0.07
    # #S = 5 * numpy.exp(-RCconst * numpy.arange(Vt.shape[0]))
    # #plt.plot(S)
    # #plt.show()
    # #plt.savefig('save/' + runname + '_spectrum' + '.' + 'pdf', bbox_inches='tight')
    # #plt.savefig('save/' + runname + '_spectrum' + '.' + 'png', bbox_inches='tight')
    # #dictionary = U @ numpy.diag(S) @ Vt
    # #for i in range(dictionary.shape[1]):
    # #    dictionary[:,i] = dictionary[:,i] / numpy.sqrt(numpy.sum( dictionary[:,i]**2 ))  # normalize columns
    # dictionary = numpy.random.randn(signal_size, signal_size)
    # extra = -  5 * dictionary[:,::2] - 5 * dictionary[:,1::2]
    # dictionary = numpy.hstack((dictionary, extra))
    # #for i in range(dictionary.shape[1]):
    # #    dictionary[:,i] = dictionary[:,i] / numpy.sqrt(numpy.sum( dictionary[:,i]**2 ))  # normalize columns
    # acqumatrix     = "randn"

    
    
    # ######################################################
    # # Noisy measurements
    # ## Square 200, exact, 2 signals each
    # runname = 'fig_exact_OMPGLSP_SquareNoisy'  # Name 
    # signal_size, dict_size = 200, 200           # Signal dimensions
    # deltas = numpy.arange(0.1, 1, 0.05)         # Phase transition grid
    # rhos = numpy.arange(0.1, 1, 0.05)
    # snr_db =  40                         # SNR ratio
    # numdata = 5                                 # Number of signals to average 
    # #solvers = [GreedyLeastSquares(1e-2), OrthogonalMatchingPursuit(1e-2, 'sparsify_QR'), OMP_GLSP(1e-2, {'mode':'weightedsumnorm', 'lambda':0.5}), OMP_GLSP(1e-2, {'mode':'signedweightedsumnorm', 'lambda':0.5}), OMP_GLSP(1e-2, {'mode':'largestGapPerc'})]
    # solvers = [GreedyLeastSquares(1e-2), OrthogonalMatchingPursuit(1e-2, 'sparsify_QR'), OMP_GLSP(1e-2, {'mode':'signedweightedsumnorm', 'lambda':0.5})]
    # solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    # success_thresh = 1e-2                       # Threshold for considering successful recovery
    # save_folder = 'save'                        # Where to save
    # dictionary     = "randn"
    # acqumatrix     = "randn"

    # ######################################################
    # # Unnormalized dictionaries

    # ### Square 60, exact, 2 signals each
    # runname = 'fig_exact_OMPGLSP_UnnormDict60'  # Name 
    # signal_size, dict_size = 60, 60           # Signal dimensions
    # deltas = numpy.arange(0.1, 1, 0.1)         # Phase transition grid
    # rhos = numpy.arange(0.1, 1, 0.1)
    # snr_db =  numpy.Inf                         # SNR ratio
    # #snr_db =  60                         # SNR ratio
    # numdata = 2                                 # Number of signals to average 
    # solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, {'mode':'weightedsumnorm', 'lambda':0.5}), OMP_GLSP(1e-6, {'mode':'signedweightedsumnorm', 'lambda':0.5}), OMP_GLSP(1e-6, {'mode':'largestGapPerc'})]
    # solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    # success_thresh = 1e-6                       # Threshold for considering successful recovery
    # save_folder = 'save'                        # Where to save
    # numpy.random.seed(4321)
    # dictionary = numpy.random.randn(signal_size, dict_size)
    # A = 5
    # RCconst = 0.2
    # S = 5 * numpy.exp(-RCconst * numpy.arange(dict_size))
    # plt.plot(S)
    # plt.show()
    # plt.savefig('save/' + runname + '_spectrum' + '.' + 'pdf', bbox_inches='tight')
    # plt.savefig('save/' + runname + '_spectrum' + '.' + 'png', bbox_inches='tight')
    # for i in range(dictionary.shape[1]):
    #     dictionary[:,i] = S[i] * dictionary[:,i] / numpy.sqrt(numpy.sum( dictionary[:,i]**2 ))  # normalize columns
    # acqumatrix     = "randn"

    ######################################################
    # Bad LSP dictionary

    ### Square 60, exact, 2 signals each
    runname = 'fig_exact_OMPGLSP_BadLSP60'  # Name 
    signal_size, dict_size = 60, 600           # Signal dimensions
    deltas = numpy.arange(0.1, 1, 0.1)         # Phase transition grid
    rhos = numpy.arange(0.1, 1, 0.1)
    snr_db =  numpy.Inf                         # SNR ratio
    #snr_db =  60                         # SNR ratio
    numdata = 20                                 # Number of signals to average 
    solvers = [GreedyLeastSquares(1e-6), OrthogonalMatchingPursuit(1e-6, 'sparsify_QR'), OMP_GLSP(1e-6, {'mode':'weightedsumnorm', 'lambda':0.5}), OMP_GLSP(1e-6, {'mode':'signedweightedsumnorm', 'lambda':0.5}), OMP_GLSP(1e-6, {'mode':'largestGapPerc'})]
    solvers_names = ['GLS','OMP', 'OMP_GLSP']   # Names used for figure files
    success_thresh = 1e-6                       # Threshold for considering successful recovery
    save_folder = 'save'                        # Where to save
    numpy.random.seed(4321)
    #dictionary, nullbasis = make_bad_LSP_dictionary(signal_size, dict_size)
    dictionary = "randn"
    acqumatrix = "randn"

    #========================================================

    time_start = datetime.datetime.now()
    print(time_start.strftime("%Y-%m-%d-%H:%M:%S:%f") + " --- Started running %s..."%(runname))

    # Make filenames for the images to save
    file_prefix = save_folder + '/' + runname        
    if len(solvers_names) > 1:
        figs_filename = [file_prefix + s for s in solvers_names]
    else:
        figs_filename = file_prefix
    # Make filename for the data file to save
    data_filename = file_prefix

    #pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, numdata, numpy.inf,  oper_type="tightframe")
    #snr_db = numpy.inf
    pt = SynthesisPhaseTransition(signal_size, dict_size, deltas, rhos, numdata, snr_db,  dictionary=dictionary, acqumatrix="randn")
    pt.set_solvers(solvers)
    pt.run(processes=1, random_state=123)
    #pt.run(random_state=123)
    
    #pt.savedata(data_filename)
    pt.savedescription(data_filename)
    pt.plot(subplot=True, solve=True, check=False, thresh=success_thresh, show=False,
            basename=data_filename, saveexts=['png', 'pdf'])
    #pt.plot_global_error(shape=((len(alphas),len(betas))), thresh=1e-6, show=False,
    #                     basename=file_prefix+'_global', saveexts=['png', 'pdf'], textfilename=file_prefix+'_global.txt')  # old comment: this order because C order (?)


    time_end = datetime.datetime.now()
    print(time_end.strftime("%Y-%m-%d-%H:%M:%S:%f") + " --- Ended. Elapsed: " + \
          str((time_end - time_start).seconds) + " seconds")


def make_bad_LSP_dictionary(n, N):
    nullbasis = numpy.random.randn(N-n, N)
    A = 5
    RCconst = 0.05
    S = 5 * numpy.exp(-RCconst * numpy.arange(N))
    plt.plot(S)
    plt.show()
    plt.savefig('save/' + 'make_bad_LSP_dictionary' + '_spectrum' + '.' + 'pdf', bbox_inches='tight')
    plt.savefig('save/' + 'make_bad_LSP_dictionary' + '_spectrum' + '.' + 'png', bbox_inches='tight')
    for i in range(nullbasis.shape[1]):
        nullbasis[:,i] = S[i] * nullbasis[:,i] / numpy.sqrt(numpy.sum( nullbasis[:,i]**2 ))  # normalize columns

    nullbasis2 = scipy.linalg.orth(nullbasis.T).T

    #U, S, Vt = numpy.linalg.svd(nullbasis2, full_matrices=True
    dictionary = scipy.linalg.null_space(nullbasis2).T

    return dictionary, nullbasis2



if __name__ == "__main__":
    #import cProfile
    #cProfile.run('run()', 'profile')

    run()
