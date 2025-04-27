

import pandas as pd
import numpy as np

f = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\analysis\results\all_learning_curves_for_paper.xlsx'
models = ['Axial Point Lines', 'Axial Point Full', 'CNN', 'Transformer', 'MLP']
datasets = ['Translate0', 'Translate1', 'Translate2', 'Rotate0', 'Rotate1', 'Rotate2']

raw_results = {}
for model in models:
    raw_results[model] = pd.read_excel(f, sheet_name=model, usecols='B,C,D,F,G,H', skiprows=[0, 1, 2, 3], names=datasets)


results_per_dataset = {}
for dataset in datasets:
    results_per_dataset[dataset] = pd.DataFrame(columns=models)
    for model in models:
        results_per_dataset[dataset] = results_per_dataset[dataset].assign(**{model: raw_results[model][dataset]})


all_axes = {}
for dataset in datasets:
    all_axes[dataset] = results_per_dataset[dataset].iloc[:200].plot(title=f'{dataset[:-1]} Distance {dataset[-1]}',
                                                                     legend=True)

