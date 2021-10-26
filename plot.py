import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec




def plot_EGNN_bounds():
    filedir = f'./logs/Cora'
    filename = 'EGNN' + f'energy_layer64_15_40_20000000_-10.npy'
    # filedir = f'./logs/Pubmed'
    # filename = 'EGNN' + f'energy_layer64_11_40_20000000_-10.npy'
    filename = os.path.join(filedir, filename)
    EGNN_energy = np.mean(np.load(filename), axis=0)
    EGNN_layer_energy = EGNN_energy[2:]

    EGNN_upper = [EGNN_energy[0] * 0.4] * len(EGNN_layer_energy)
    EGNN_lower = [EGNN_energy[0] * 0.11] + list(EGNN_layer_energy[0:-1] * 0.11)

    filename = 'GCN' + f'energy_layer64_12_100_5000000_-10.npy'
    filename = os.path.join(filedir, filename)
    GCN_energy = np.mean(np.load(filename), axis=0)
    GCN_layer_energy = GCN_energy[-64:]

    filename = 'GCNII' + f'energy_layer64_12_100_5000000_-10.npy'
    filename = os.path.join(filedir, filename)
    GCN_energy = np.mean(np.load(filename), axis=0)
    GCNII_layer_energy = GCN_energy[-64:]



    x_range = list(range(1, 65, 1))
    y_energy = [EGNN_layer_energy, GCNII_layer_energy, GCN_layer_energy, EGNN_upper, EGNN_lower]
    line_style = ['-', '-', '--', '-.', ':']
    color = ['r', 'm', 'k', 'b', 'g']
    fig = plt.figure()
    legends = ['EGNN', 'GCNII', 'GCN', 'Upper', 'Lower']

    for i in range(5):
        y = y_energy[i]
        plt.plot(x_range, y, ls=line_style[i], c=color[i])

    plt.legend(legends, loc='upper right', bbox_to_anchor=(1., 0.9), fontsize=15)
    plt.xlabel('Layers', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    plt.tight_layout()


    filedir = f'./figs/Cora'
    # filedir = f'./figs/Pubmed'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    filename = 'EGNN' + f'_Cora_layers64_bounds.pdf'
    filename = os.path.join(filedir, filename)
    plt.savefig(filename)

def plot_hyper():
    x = []
    # x.append([1.5, 1, 0.5, 0, -0.5, -1, -1.5])
    x.append([1, 0.5, 0, -0.5, -1])
    x.append([0, 0.1, 0.3, 0.5, 1, 1.5, 2])
    x.append([0., 0.1,  0.3,  0.5,  0.75, 0.95])
    x.append([0.2, 0.4, 0.6, 0.8, 1.])

    y = []
    # y.append([26.4, 27.4, 22.9, 85.2, 85.6, 85.5, 85.5])
    # y.append([27.2, 64.8, 83.1, 84.2, 84.9, 85.2, 85.3])
    # y.append([12.9, 85.5, 85.7, 85.3, 84.1, 71.5])
    # y.append([85.5, 85.5, 85.4, 85.6, 85.7])

    # y.append([38, 38.4, 41.4, 80, 80.2, 80, 80])
    # y.append([50.4, 79.9, 80.1, 80.1, 79.9, 80, 80])
    # y.append([44.1, 79.9, 79.9, 79.6, 78.5, 75.7])
    # y.append([79.9, 79.9, 79.8, 80, 80.1])

    # y.append([11.1, 11.7, 18.8, 93.1, 93.2, 93.2, 93.3])
    # y.append([89.4, 92.9, 93.1, 93.3, 93.2, 93.3, 93.3])
    # y.append([79.7, 93.3, 93.1, 92.8, 92.2, 90.5])
    # y.append([93.2, 93.2, 93.2, 93.2, 93.3])

    y.append([72.2, 72.5, 72.6, 72.7, 72.7])
    y.append([72.7, 71.6, 71.8, 71.2, 71.8, 71.7, 71.8])
    y.append([62.4, 71.9, 72, 72.3, 72.7, 69.5])
    y.append([72.7, 72.9, 72.6, 72.7, 72.7])


    colors = ['r', 'b', 'k', 'm']
    patterns = ('/', '//', '-', '\\')
    x_labels = ['$b$', '$\gamma$', '$c_{\min}$', '$c_{\max}$']
    figure_all = plt.figure(figsize=(8, 2))
    spec = gridspec.GridSpec(ncols=4, nrows=1, figure=figure_all,
                             wspace=0.05, left=0.05, right=0.99, top=0.9,bottom=0.105)
    for i in range(4):
        ax = figure_all.add_subplot(spec[0, i])
        if i == 0:
            width = 0.23
        elif i == 1:
            width = 0.15
        else:
            width = 0.1
        plt.bar(x[i], y[i], color='white', edgecolor=colors[i], width=width, hatch=patterns[i])
        ax.set_title(x_labels[i], fontsize=10)
        plt.xticks(fontsize=10)
        if i == 0:
            plt.yticks(fontsize=10)
        else:
            plt.yticks([])

    # spec.tight_layout(figure_all)

    filedir = f'./figs/Ogbn-arxiv'
    # filedir = f'./figs/Pubmed'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    filename = 'EGNN' + f'_Ogbn-arxiv_layers32_hypers.pdf'
    filename = os.path.join(filedir, filename)
    plt.savefig(filename)

def plot_hyper_NAIS():
    x = [0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    y = []
    y.append([0.7071, 0.708, 0.7083, 0.7098, 0.7083, 0.7111, 0.7073])
    y.append([0.7111, 0.7078, 0.7108, 0.7081, 0.7068, 0.7056, 0.7058])
    y.append([0.7089, 0.7088, 0.7093, 0.7111, 0.7098, 0.7071, 0.7031])

    colors = ['r', 'b', 'k', 'm']
    patterns = ('/', '//', '\\', '-')
    x_labels = ['$\lambda_1$', '$\lambda_2$', '$\lambda_3$']
    figure_all = plt.figure(figsize=(6, 1.5))
    spec = gridspec.GridSpec(ncols=3, nrows=1, figure=figure_all,
                             wspace=0.05, left=0.08, right=0.99, top=0.85, bottom=0.13)

    for i in range(3):
        ax = figure_all.add_subplot(spec[0, i])
        plt.bar(x, y[i], color='white', edgecolor=colors[i], width=0.1, hatch=patterns[i])
        ax.set_title(x_labels[i], fontsize=10)
        ax.set_xscale("log")
        plt.xticks(fontsize=10)
        if i == 0:
            plt.yticks(fontsize=10)
        else:
            plt.yticks([])

    filedir = './figs/NAIS'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    filename = 'NAIS_GNN_hypers.pdf'
    filename = os.path.join(filedir, filename)
    plt.savefig(filename)


if __name__ == "__main__":
    # plot_EGNN_bounds()
    plot_hyper()
    # plot_hyper_NAIS()


# def avg(MI_array, interval=50):
#     len_list = MI_array.shape[0]
#     len_avg = int(np.ceil(len_list / interval))
#     MI_return = []
#     for i in range(len_avg):
#         MI_avg = np.mean(MI_array[i*interval:(i+1)*interval])
#         MI_return.append(MI_avg)
#     return np.array(MI_return)
#
# def plot_MI(paths, print_layers, print_epochs, X, savefile):
#     interval = 100
#     MI = []
#     for i, path in enumerate(paths):
#         log = torch.load(path)
#         MI_seeds = np.mean(np.array(log[X][0:10]), axis=0)
#         MI.append(MI_seeds)
#
#     if X in ['MI_XiX', 'MI_XiY', 'MI_NiXi', 'MI_NiY', 'MI_PNiY', 'MI_NNiY', 'MI_PNiNNi']:
#         MI = np.array(MI) / np.max(MI)
#         xlable = 'Mutual information'
#     elif X in ['grad_mean']:
#         grad_var = []
#         for i, path in enumerate(paths):
#             log = torch.load(path)
#             grad_var.append(log['grad_var'][-1])
#         MI = np.array(MI) / np.array(grad_var)
#         xlable = 'Gradient SNR'
#     elif X in ['MI_PNiXi', 'MI_NNiXi']:
#         MI_NNiXi = []
#         for i, path in enumerate(paths):
#             log = torch.load(path)
#             MI_NNiXi.append(log['MI_NNiXi'][-1])
#         MI_NNiXi = np.array(MI_NNiXi)
#         MI_NNiXi[MI_NNiXi <= 0] = 1e-8
#         MI = np.array(MI) / np.array(MI_NNiXi)
#         xlable = 'Mutual Information Ratio'
#     elif X in ['MI_NicomXi', 'MI_NicomY',  'MI_NihopXi', 'MI_NihopY']:
#         MI = np.array(MI) / np.max(MI)
#         xlable = 'Mutual information'
#         MI = MI[:, :, 1]
#         print(MI.shape)
#     elif X in ['dis', 'dis_hop', 'dis_com']:
#         xlable = 'Distance'
#         MI = np.array(MI) / np.max(MI)
#         MI = MI[:, :, 4]
#     elif X in ['MI_Xi1Xi']:
#         xlable = 'MI ratio'
#         MI_neighbor = []
#         for i, path in enumerate(paths):
#             log = torch.load(path)
#             MI_seeds = np.mean(np.array(log['MI_Ni1Xi'][0:10]), axis=0)
#             MI_neighbor.append(MI_seeds)
#         MI = np.array(MI) / (np.array(MI_neighbor) + 1e-8)
#         MI /= np.max(MI)
#     else:
#         pass
#
#     x_range = range(interval, print_epochs[-1]+2, interval)
#
#     legends = []
#     fig = plt.figure()
#     markers = ['s', 'o', 'v', '^', '1', '2', '+']
#     for i in range(len(paths)):
#         MI_i = MI[i]
#         MI_i = avg(MI_i, interval)
#         plt.plot(x_range, MI_i, marker=markers[i], markersize=5)
#         legends.append(f'Layer {print_layers[i]}')
#
#     plt.legend(legends, loc='upper right', fontsize=10)
#     plt.xlabel('Epochs')
#     plt.ylabel(xlable)
#     plt.savefig(savefile)
#
# def plot_MI_hops(paths, print_layers, print_epochs, X, Y, savefile):
#     interval = 100
#     MI_X = []
#     MI_Y = []
#     for i, path in enumerate(paths):
#         log = torch.load(path)
#         MI_seeds = np.mean(np.array(log[X][0:10]), axis=0)
#         MI_X.append(MI_seeds)
#         MI_seeds = np.mean(np.array(log[Y][0:10]), axis=0)
#         MI_Y.append(MI_seeds)
#
#     MI_X = np.array(MI_X) / np.max(MI_X)
#
#     hops = [1, 2, 3, 4, 5]
#     MI_X_layer1 = []
#     MI_X_ratios = []
#     legends = []
#     fig = plt.figure()
#     markers = ['s', 'o', 'v', '^', '1', '2', '+']
#     for hop in range(5):
#         tmp = avg(MI_X[0, :, hop], interval)
#         MI_X_layer1.append(np.mean(tmp[-5:]))
#
#     for i in range(len(paths)):
#         # MI_X_max = avg(MI_X[i, :, 0], interval)
#         MI_X_i = []
#         for hop in range(5):
#             MI_X_hop = avg(MI_X[i, :, hop], interval)
#             MI_X_i.append(np.mean(MI_X_hop[-5:]) / MI_X_layer1[hop])
#
#         plt.plot(hops, MI_X_i, marker=markers[i], markersize=5)
#         legends.append(f'Layer {print_layers[i]}')
#
#     plt.legend(legends, loc='upper left', fontsize=10)
#     plt.xlabel('Hops')
#     plt.ylabel('MI ratio')
#     plt.savefig(savefile)
#
#
#
# def plot_inforplane_lastLayer(paths, print_layers, print_epochs, X, Y, savefile):
#     interval = 100
#     plot_num = int((print_epochs[-1]+1)/interval)
#     sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=plot_num))
#     sm._A = []
#     fig = plt.figure()
#
#     for i, path in enumerate(paths):
#         log = torch.load(path)
#         MI_X = np.array(log[X])
#         MI_Y = np.array(log[Y])
#
#         MI_X = MI_X[:, -1][print_epochs]
#         MI_Y = MI_Y[:, -1][print_epochs]
#
#         MI_X = avg(MI_X, interval)
#         MI_Y = avg(MI_Y, interval)
#         # MI_X = discount(MI_X, 0.2)
#         # MI_Y = MI_Y[:, -1][print_epochs]
#         plt.scatter(MI_X, MI_Y, s=20, facecolors=[sm.to_rgba(epoch) for epoch in range(plot_num)], edgecolor='none', zorder=2)
#         plt.plot(MI_X, MI_Y, label=f'Layer {print_layers[i]}')
#     plt.xlabel('I(H; X)')
#     plt.ylabel('I(H; Y)')
#     plt.title('Information Plane')
#     # plt.ylim([np.amin(MI_X), np.max(MI_X)])
#     # plt.xlim([np.amin(MI_Y), np.min(MI_Y)])
#     cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
#     plt.colorbar(sm, label='Epoch', cax=cbaxes)
#     plt.tight_layout()
#     plt.savefig(savefile)
#
#
# def plot_acc_compare(paths_list, print_layers, savefile):
#     fig, ax1 = plt.subplots()
#
#     num_compare = len(paths_list)
#     colors = ['tab:red', 'tab:blue']
#     markers = ['s', 'o', 'v', '^', '1', '2', '+']
#     legends = ['Norm', 'GroupNorm']
#     for j in range(num_compare):
#         acc_means = []
#         acc_stds = []
#         for i, path in enumerate(paths_list[j]):
#             log = torch.load(path)
#             acc_array = np.array(log['acc_test'][-10:])
#             acc_seeds = np.max(acc_array, axis=1)
#             acc_means.append(np.mean(acc_seeds))
#             acc_stds.append(np.std(acc_seeds))
#         a, b, c = ax1.errorbar(print_layers, acc_means, yerr=acc_stds, fmt='-', marker=markers[j], markersize=5,
#                                uplims=True, lolims=True, capsize=0, color=colors[j])
#         b[0].set_marker('_')
#         b[0].set_markersize(5)
#         b[1].set_marker('_')
#         b[1].set_markersize(5)
#     ax1.set_xlabel('Layers')
#     ax1.set_ylabel('Accuracy')
#     plt.legend(legends, loc='upper left', fontsize=10)
#     plt.savefig(savefile)
#
# def plot_acc(paths, print_layers, savefile):
#     # fig, ax1 = plt.figure()
#     fig, ax1 = plt.subplots()
#     acc_means = []
#     acc_stds = []
#     MI_means = []
#     MI_stds = []
#     for i, path in enumerate(paths):
#         log = torch.load(path)
#         acc_array = np.array(log['acc_test'])
#         acc_seeds = np.mean(acc_array[:, -100:], axis=1)
#         acc_means.append(np.mean(acc_seeds))
#         acc_stds.append(np.std(acc_seeds))
#
#         MI_array = np.array(log['MI_XiY'][:-2])
#         MI_seeds = np.mean(MI_array[:, -100:], axis=1)
#         MI_means.append(np.mean(MI_seeds))
#         MI_stds.append(np.std(MI_seeds))
#     # plt.plot(print_layers, accs, label=f'Layer {print_layers[i]}')
#     color = 'tab:red'
#     a, b, c = ax1.errorbar(print_layers, acc_means, yerr=acc_stds, fmt='-', marker='s', markersize=5,
#                            uplims=True, lolims=True, capsize=0, color=color)
#     b[0].set_marker('_')
#     b[0].set_markersize(5)
#     b[1].set_marker('_')
#     b[1].set_markersize(5)
#     ax1.set_xlabel('Layers')
#     ax1.set_ylabel('Accuracy', color=color)
#     ax1.tick_params(axis='y', labelcolor=color)
#
#     ax2 = ax1.twinx()
#     color = 'tab:blue'
#     MI_means /= np.max(MI_means)
#     MI_stds /= np.max(MI_means)
#     a, b, c = ax2.errorbar(print_layers, MI_means, yerr=MI_stds, fmt='-', marker='o', markersize=5,
#                            uplims=True, lolims=True, capsize=0, color=color)
#     b[0].set_marker('_')
#     b[0].set_markersize(5)
#     b[1].set_marker('_')
#     b[1].set_markersize(5)
#     ax2.set_ylabel('Mutual information')
#     ax2.tick_params(axis='y', labelcolor=color)
#
#
#     # plt.xlabel('Layers')
#     # plt.ylabel('Accuracy')
#     plt.savefig(savefile)
#
# def plot_hypers(trnr):
#     pass






