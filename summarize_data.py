import os

import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='paper', style='whitegrid', palette='colorblind', font='serif', font_scale=1.5, rc=None)

# Load the data
# f = "results_magnitude=0.4_ERROR=implicit_missing_values.csv"
files = os.listdir("./data/outputs")
dataset_files = {}
dataset_times = {}
dataset_batch_sizes = {}
for f in files:
    data = pd.read_csv(f'data/outputs/{f}')
    splitted_f = f.split('_')
    dataset = splitted_f[1]
    batch_size = int(splitted_f[2].split('.')[0])
    if dataset not in dataset_files:
        dataset_files[dataset] = [f]
    else:
        dataset_files[dataset].append(f)
    pred, true = data['predictions'], data['ground_truth']
    train_time, test_time, total_time, batch_time = data['train_time'], data['test_time'], data['total_time'], data['batch_time']


    if dataset not in dataset_times:
        dataset_times[dataset] = [total_time]
    else:
        dataset_times[dataset].append(total_time)

    if dataset not in dataset_batch_sizes:
        dataset_batch_sizes[dataset] = [batch_size]
    else:
        dataset_batch_sizes[dataset].append(batch_size)
    # calculate roc
    roc = float(roc_auc_score(true, pred))
    (tn, fp), (fn, tp) = confusion_matrix(true, pred)
    total = tn + fp + fn + tp

    # save results
    results = pd.DataFrame({
        'roc': [round(roc, 2)],
        'tn': [round(tn/total, 2)],
        'fp': [round(fp/total, 2)],
        'fn': [round(fn/total, 2)],
        'tp': [round(tp/total, 2)],
        'avg_train_time': [train_time.mean()],
        'avg_test_time': [test_time.mean()],
        'std_train_time': [train_time.std()],
        'std_test_time': [test_time.std()],
        'min_train_time': [train_time.min()],
        'min_test_time': [test_time.min()],
        'max_train_time': [train_time.max()],
        'max_test_time': [test_time.max()],
    })
    results.to_csv(f'results/summary_{f}', index=False)
    # Plot confusion matrix
    conf_matrix = confusion_matrix(true, pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d',cmap='YlGnBu',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix, BATCH SIZE={batch_size}')
    plt.savefig(f'results/confusion_matrix_plot_{f}.png')
    plt.close()

    # Generate plot for train time
    plt.figure(figsize=(10, 5))
    plt.plot(train_time, label='Train Time')
    plt.xlabel('Batch')
    plt.ylabel('Time (seconds)')
    plt.title(f'Train Time per Batch, BATCH SIZE={batch_size}')
    plt.legend()
    plt.savefig(f'results/train_time_plot_{f}.png')
    plt.close()

    # Generate plot for test time
    plt.figure(figsize=(10, 5))
    plt.plot(test_time, label='Test Time')
    plt.xlabel('Batch')
    plt.ylabel('Time (seconds)')
    plt.title(f'Test Time per Batch, BATCH SIZE={batch_size}')
    plt.legend()
    plt.savefig(f'results/test_time_plot_{f}.png')
    plt.close()

    # Generate plot for total time
    plt.figure(figsize=(10, 5))
    plt.plot(total_time, label='Total Time')
    plt.plot(batch_time, label='Batch Time')
    plt.xlabel('Batch')
    plt.ylabel('Time (seconds)')
    plt.title(f'Total Time per Batch, BATCH SIZE={batch_size}')
    plt.legend()
    plt.savefig(f'results/total_time_plot_{f}.png')
    plt.close()

for dataset in dataset_times:
    times = dataset_times[dataset]
    batch_sizes = dataset_batch_sizes[dataset]
    plt.figure(figsize=(10, 5))
    for i in range(len(times)):
        plt.plot(times[i], label=f'BATCH SIZE={batch_sizes[i]}')

    plt.xlabel('Batch')
    plt.ylabel('Time (seconds)')
    plt.title(f'Total Time per Batch, {dataset}')
    plt.legend()
    plt.savefig(f'results/total_time_plot_{dataset}.png')

for dataset, files in dataset_files.items():
    def read_file(f):
        df = pd.read_csv(f'results/summary_{f}')
        batch_size = int(f.split('_')[2].split('.')[0])
        df['batch_size'] = batch_size
        return df
    df = pd.concat([read_file(f) for f in files], ignore_index=True)
    df.to_csv(f'results/summary_{dataset}.csv', index=False)
    new_df = df[['roc', 'tn', 'fp', 'fn', 'tp', 'batch_size']]
    new_df.to_latex(f'results/summary_{dataset}.tex', index=False)