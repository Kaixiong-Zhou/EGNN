import numpy as np

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = np.sum(np.square(X), axis=1, keepdims=True)
    dists = x2 + x2.T - 2 * np.matmul(X, X.T)
    return dists

def entropy_estimator_kl(x, var, probs_node=None):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = float(x.shape[1]), float(x.shape[0])
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims / 2.0) * np.log(2 * np.pi * var)
    if probs_node is None:
        lprobs = np.log(np.sum(np.exp(-dists2), axis=1)) - np.log(N) - normconst
        h = -np.mean(lprobs)
    else:
        lprobs = np.log(np.matmul(np.exp(-dists2), probs_node)) - normconst
        h = -np.sum(lprobs * probs_node)

    return dims/2 + h

def entropy_estimator_bd(x, var, probs_node=None):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = float(x.shape[1]), float(x.shape[0])
    val = entropy_estimator_kl(x,4*var, probs_node)
    return val + np.log(0.25)*dims/2

def kde_condentropy(x, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = x.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)


def mi_kde(h, inputdata, labeldata, num_classes, var, mode='upper', entro_which='first', probs_node=None):
    nats2bits = float(1.0 / np.log(2))
    h_norm = np.sum(np.square(h), axis=1, keepdims=True)
    h_norm[h_norm == 0.] = 1e-3
    h = h / np.sqrt(h_norm)
    input_norm = np.sum(np.square(inputdata), axis=1, keepdims=True)
    input_norm[input_norm == 0.] = 1e-3
    inputdata = inputdata / np.sqrt(input_norm)

    if labeldata is not None:
        if len(labeldata.shape) > 1:
            labeldata = np.squeeze(labeldata, axis=1)

    # compute the entropy of hidden activity
    if mode == 'upper':
        entropy_h = entropy_estimator_kl(h, var, probs_node)
    else:
        entropy_h = entropy_estimator_bd(h, var, probs_node)

    entropy_input = 0.
    if entro_which in ['second', 'both']:
        entropy_input = entropy_estimator_kl(inputdata, var, probs_node) if mode == 'upper' else \
            entropy_estimator_bd(inputdata, var, probs_node)


    # compute the entropy of hidden activity given input
    # entropy_h_input = kde_condentropy(h, var)
    entropy_h_input = 0.
    if entro_which in ['first', 'both']:

        indices = np.argmax(inputdata, axis=1)
        indices = np.expand_dims(indices, axis=1)
        p_input, unique_inverse_input = get_unique_probs(indices)
        p_input = np.asarray(p_input).T
        for i in range(len(p_input)):
            labelixs = unique_inverse_input==i
            if probs_node is not None:
                probs_idx = probs_node[labelixs, :] / np.sum(probs_node[labelixs, :])
            else:
                probs_idx = None
            if mode == 'upper':
                entropy_h_input += p_input[i] * entropy_estimator_kl(h[labelixs, :], var, probs_idx)
            else:
                entropy_h_input += p_input[i] * entropy_estimator_bd(h[labelixs, :], var, probs_idx)

    # entropy_input_h = kde_condentropy(inputdata, var)
    entropy_input_h = 0.
    if entro_which in ['second', 'both']:
        indices = np.argmax(h, axis=1)
        indices = np.expand_dims(indices, axis=1)
        p_h, unique_inverse_h = get_unique_probs(indices)
        p_h = np.asarray(p_h).T
        for i in range(len(p_h)):
            labelixs = unique_inverse_h==i
            if probs_node is not None:
                probs_idx = probs_node[labelixs, :] / np.sum(probs_node[labelixs, :])
            else:
                probs_idx = None
            if mode == 'upper':
                entropy_input_h += p_h[i] * entropy_estimator_kl(inputdata[labelixs, :], var, probs_idx)
            else:
                entropy_input_h += p_h[i] * entropy_estimator_bd(inputdata[labelixs, :], var, probs_idx)


    # compute the entropy of hidden activity given label
    entropy_h_label = float(0.)
    if labeldata is not None:
        for i in range(num_classes):
            labelixs = labeldata == i
            prob = float(sum(labelixs)) / labeldata.shape[0]
            if mode == 'upper':
                entropy_h_label += prob * entropy_estimator_kl(h[labelixs, :], var)
            else:
                entropy_h_label += prob * entropy_estimator_bd(h[labelixs, :], var)


    # print((entropy_h-entropy_h_input), (entropy_input - entropy_input_h))
    if entro_which == 'both':
        MI_HX = ((entropy_h - entropy_h_input) + (entropy_input - entropy_input_h)) * 0.5
    elif entro_which == 'first':
        MI_HX = entropy_h - entropy_h_input
    else:
        MI_HX = entropy_input - entropy_input_h
    return nats2bits*MI_HX, nats2bits*(entropy_h-entropy_h_label)





