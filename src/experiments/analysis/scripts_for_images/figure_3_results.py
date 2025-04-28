
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


fig = plt.plot()


