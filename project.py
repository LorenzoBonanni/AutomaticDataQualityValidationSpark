import argparse
import os
import random
import time

import pandas as pd
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql import SparkSession
from tqdm import trange

from anomaly_utils import perturb_batch
from data_utils import get_df_splits, compute_statistics

spark = SparkSession.builder \
    .appName("DataQualityValidation") \
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.9") \
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
    .master("local[4]") \
    .getOrCreate()
from synapse.ml.isolationforest import IsolationForest

TRAIN_BATCHES = 10
CONTAMINATION = 0.01  # Fraction of anomalies in the dataset
MAGNITUDE = 0.4

# Train Isolation Forest model
def train_isolation_forest(final_data):
    isolation_forest = IsolationForest() \
        .setBootstrap(False) \
        .setMaxSamples(int(0.2 * final_data.count())) \
        .setFeaturesCol("scaled_features") \
        .setPredictionCol("prediction") \
        .setScoreCol("outlierScore") \
        .setContamination(CONTAMINATION) \
        .setContaminationError(0.01 * CONTAMINATION) \
        .setRandomSeed(1)
    return isolation_forest.fit(final_data)


def prepare_training_data(statistics, df_splits, train_batches, i):
    """Update statistics and prepare training data."""
    data, schema = compute_statistics(df_splits[train_batches + i].drop("timestamp"))
    statistics.append(data)
    return spark.createDataFrame(statistics, schema=schema)


def prepare_test_batch(test_batch, anomaly, magnitude):
    """Prepare and perturb the test batch."""
    chosen_anomaly = random.choice(anomaly)
    perturbed_rdd, target = perturb_batch(test_batch, chosen_anomaly, magnitude)
    return perturbed_rdd, target


def calculate_batch_time(test_batch):
    """Calculate the time elapsed for the test batch."""
    initial_timestamp = test_batch.head()['timestamp']
    final_timestamp = test_batch.tail(1)[0]['timestamp']
    return (final_timestamp - initial_timestamp).seconds


def prepare_and_scale_features(data, scaler_model=None):
    """Prepare and normalize feature vectors for training and test data."""
    assembler = VectorAssembler(inputCols=data.columns, outputCol="features")

    # Transform training data
    feature_vectors = assembler.transform(data)
    if scaler_model is None:
        scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
        scaler_model = scaler.fit(feature_vectors)

    scaled_data = scaler_model.transform(feature_vectors).select("scaled_features")
    return scaled_data, scaler_model


def train_and_predict(isolation_forest_model, scaled_test_data):
    """Train the model and make predictions on the test data."""
    return isolation_forest_model.transform(scaled_test_data).collect()[0]["prediction"]


def run_experiment(df_splits, train_batches=TRAIN_BATCHES, magnitude=MAGNITUDE, anomaly=None, args=None):
    random.seed(1)
    results = {
        'ground_truth': [],
        'predictions': [],
        'train_time': [],
        'test_time': [],
        'total_time': [],
        'batch_time': []
    }

    # Compute initial statistics for training batches
    statistics = [compute_statistics(batch.drop("timestamp"))[0] for batch in df_splits[:train_batches]]

    for i in trange(len(df_splits) - train_batches - 1):
        initial_time = time.time()

        # Update statistics and prepare training data
        training_data = prepare_training_data(statistics, df_splits, train_batches, i)
        statistics_time = time.time() - initial_time

        # Calculate batch time
        test_batch = df_splits[train_batches + i + 1]
        results['batch_time'].append(calculate_batch_time(test_batch))
        test_batch = test_batch.drop("timestamp")
        # Prepare and perturb the test batch
        perturbed_rdd, target = prepare_test_batch(test_batch, anomaly, magnitude)
        results['ground_truth'].append(target)
        test_data = spark.createDataFrame(perturbed_rdd, test_batch.schema)

        # Prepare and scale features using the pre-fitted scaler
        scaled_training_data, scaler_model = prepare_and_scale_features(training_data, None)

        # Train model
        initial_time = time.time()
        isolation_forest_model = train_isolation_forest(scaled_training_data)
        train_time = time.time() - initial_time + statistics_time
        results['train_time'].append(train_time)

        # Detect anomalies in new data
        initial_time = time.time()
        data, schema = compute_statistics(test_data)
        test_data_stats = spark.createDataFrame([data], schema=schema)
        scaled_test_data, _ = prepare_and_scale_features(test_data_stats, scaler_model)
        prediction = train_and_predict(isolation_forest_model, scaled_test_data)
        results['predictions'].append(prediction)
        test_time = time.time() - initial_time
        results['test_time'].append(test_time)
        results['total_time'].append(train_time + test_time)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = f'data/outputs/results_{args.dataset}_{args.batch_size}.csv' if args else 'data/outputs/results.csv'
    results_df.to_csv(output_path, index=False)


def seed_everything(param):
    random.seed(param)
    os.environ['PYTHONHASHSEED'] = str(param)


def main():
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    seed_everything(1)
    parser = argparse.ArgumentParser(description="Run data quality validation experiment")
    parser.add_argument('--dataset', type=str, required=True, choices=['household', 'metropt3', 'onlineretail'], help='Dataset to use for the experiment')
    parser.add_argument('--batch_size', type=int, required=False, help='Batch size for the experiment', default=100)
    arguments = parser.parse_args()
    df_splits = get_df_splits(spark, arguments)
    spark.sparkContext.setLogLevel("ERROR")
    run_experiment(df_splits, anomaly=list(range(6)), args=arguments)
    spark.stop()


if __name__ == '__main__':
    main()
