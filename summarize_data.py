import os

import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt

# Load the data
# f = "results_magnitude=0.4_ERROR=implicit_missing_values.csv"
for f in os.listdir("./data/outputs"):
    data = pd.read_csv(f'data/outputs/{f}')
    pred, true = data['predictions'], data['ground_truth']
    train_time, test_time = data['train_time'], data['test_time']

    # calculate roc
    roc = float(roc_auc_score(true, pred))
    (tn, fp), (fn, tp) = confusion_matrix(true, pred)

    # save results
    results = pd.DataFrame({
        'roc': [roc],
        'tn': [tn],
        'fp': [fp],
        'fn': [fn],
        'tp': [tp],
        'avg_train_time': [train_time.mean()],
        'avg_test_time': [test_time.mean()],
        'std_train_time': [train_time.std()],
        'std_test_time': [test_time.std()],
        'min_train_time': [train_time.min()],
        'min_test_time': [test_time.min()],
        'max_train_time': [train_time.max()],
        'max_test_time': [test_time.max()],
    })
    results.to_csv(f'results/summary_{f}.csv', index=False)

    # Generate plot for train time
    plt.figure(figsize=(10, 5))
    plt.plot(train_time, label='Train Time')
    plt.xlabel('Batch')
    plt.ylabel('Time (seconds)')
    plt.title('Train Time per Batch')
    plt.legend()
    plt.savefig(f'results/train_time_plot_{f}.png')
    plt.close()

    # Generate plot for test time
    plt.figure(figsize=(10, 5))
    plt.plot(test_time, label='Test Time')
    plt.xlabel('Batch')
    plt.ylabel('Time (seconds)')
    plt.title('Test Time per Batch')
    plt.legend()
    plt.savefig(f'results/test_time_plot_{f}.png')
    plt.close()