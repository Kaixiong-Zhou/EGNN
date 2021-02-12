# Simplified MI computation code from https://github.com/ravidziv/IDNNs
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()
bin_space = 0.1

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

def calc_condtion_entropy(layerdata_i, p_i):
    # Condition entropy of hidden activity given x
    p_t_given_x, _ = get_unique_probs(layerdata_i)
    p_t_given_x = np.asarray(p_t_given_x, dtype=np.float32).T
    H2X = p_i * (-np.sum(p_t_given_x * np.log2(p_t_given_x)))
    return H2X

def mi_bin(layerdata, num_of_bins, p_input, unique_inverse_input, p_label, unique_inverse_label, mode='H'):

    if mode == 'X':
        # num_of_bins = np.floor((np.amax(layerdata) - np.amin(layerdata)) / bin_space)
        # bins = np.linspace(np.amin(layerdata), np.amax(layerdata), num_of_bins, dtype='float32')
        # layerdata = layerdata / np.sqrt(np.sum(np.square(layerdata), axis=1, keepdims=True))
        bins = np.linspace(0., 1., num_of_bins, dtype='float32')
        digitized = bins[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins) - 1].reshape(len(layerdata), -1)
    else:
        layerdata = layerdata / (np.amax(layerdata) - np.amin(layerdata))
        # bins = np.linspace(np.amin(layerdata), np.amax(layerdata), num_of_bins, dtype='float32')
        # layerdata = layerdata / np.sqrt(np.sum(np.square(layerdata), axis=1, keepdims=True))
        bins = np.linspace(0., 1., num_of_bins, dtype='float32')
        digitized = bins[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins, right=True)].reshape(len(layerdata), -1)

    p_ts, _ = get_unique_probs(digitized)
    H_LAYER = -np.sum(p_ts * np.log2(p_ts))
    # compute the mutual information between input data and hidden layer
    H2X_array = np.array(
        Parallel(n_jobs=NUM_CORES)(delayed(calc_condtion_entropy)(digitized[unique_inverse_input == i, :], p_input[i])
                                   for i in range(p_input.shape[0])))
    MI_HX = H_LAYER - np.sum(H2X_array)
    # compute the mutual information between label data and hidden layer
    H2Y_array = np.array(
        Parallel(n_jobs=NUM_CORES)(delayed(calc_condtion_entropy)(digitized[unique_inverse_label == i, :], p_label[i])
                                   for i in range(p_label.shape[0])))
    MI_HY = H_LAYER - np.sum(H2Y_array)
    return MI_HX, MI_HY


def mi_XY_bin(inputdata, labeldata, num_of_bins):
    # inputdata = inputdata / np.sqrt(np.sum(np.square(inputdata), axis=1, keepdims=True))
    bins = np.linspace(0., 1., num_of_bins, dtype='float32')

    # bins = np.linspace(np.amin(inputdata), np.amax(inputdata), num_of_bins, dtype='float32')
    digitized = bins[np.digitize(np.squeeze(inputdata.reshape(1, -1)), bins) - 1].reshape(len(inputdata), -1)
    p_input, unique_inverse_input = get_unique_probs(digitized)
    p_input = np.asarray(p_input).T

    p_label, unique_inverse_label = get_unique_probs(labeldata)
    p_label = np.asarray(p_label).T
    MI_XX, MI_XY =  mi_bin(inputdata, num_of_bins, p_input, unique_inverse_input, p_label, unique_inverse_label, mode='X')

    return MI_XX, MI_XY, p_input, unique_inverse_input, p_label, unique_inverse_label


# def mi_HY_bin(labelixs, layerdata, binsize):
#     # This is even further simplified, where we use np.floor instead of digitize
#     def get_h(d):
#         digitized = np.floor(d / binsize).astype('int')
#         p_ts, _ = get_unique_probs(digitized)
#         return -np.sum(p_ts * np.log(p_ts))
#
#     H_LAYER = get_h(layerdata)
#     H_LAYER_GIVEN_OUTPUT = 0
#     for label, ixs in labelixs.items():
#         H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(layerdata[ixs, :])
#     return H_LAYER, H_LAYER - H_LAYER_GIVEN_OUTPUT


