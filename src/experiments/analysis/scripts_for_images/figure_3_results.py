
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join


results_path = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\code\latent_ood_in_world_models\data\results'

data_types = ['translate', 'rotate']
model_types = ['axial_point_network_full', 'axial_point_network_lines', 'cnn', 'mlp_nn', 'transformer']
model_names = ['Axial Pointer', 'Axial Pointer Linear', 'CNN', 'MLP', 'Transformer']


all_results = {}
for data_type in data_types:
    all_results[data_type] = {0: [], 1: [], 2: []}
    for model in model_types:
        f = join(results_path, data_type, f'log_{data_type}_{model}.csv')
        raw_results = pd.read_csv(f, usecols=[1, 4], skiprows=[0, 0], names=['error', 'distance'])
        for d in [0, 1, 2]:
            all_results[data_type][d].append(raw_results['error'][raw_results['distance'] == d].to_numpy())

'''
del all_results['rotate'][0][1]
del all_results['rotate'][1][1]
del all_results['rotate'][2][1]
'''

fig = plt.figure(constrained_layout=True)
fig.suptitle('Results')
subfigs = fig.subfigures(nrows=2, ncols=1)

for data_type, subfig in zip(data_types, subfigs):
    subfig.suptitle(data_type)

    axs = subfig.subplots(nrows=1, ncols=3)
    for d, ax in zip([0, 1, 2], axs):
        data = np.transpose(all_results[data_type][d])
        ax.plot(data)
        ax.set_title(f'Distance {d}')

fig.text(0.5, 0.04, 'epochs', ha='center', va='center')
fig.text(0.06, 0.5, '% error', ha='center', va='center', rotation='vertical')






