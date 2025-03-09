
import random
import time

import pandas as pd
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, stddev, max, min, approx_count_distinct
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from tqdm import trange

spark = SparkSession.builder \
        .appName("DataQualityValidation") \
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.9") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .master("local[4]") \
        .getOrCreate()
from synapse.ml.isolationforest import IsolationForest

# Constants
DATA_PATHS = {
    'metroPT3': 'data/inputs/MetroPT3(AirCompressor).csv',
    'onlineRetail': 'data/inputs/Online Retail.xlsx'
}
ANOMALY_TYPES = ["explicit_missing_values", "implicit_missing_values", "numeric_anomalies", "swaped_numeric_fields", "swapped_textual_fields", "typos"]
# BATCH_SIZE = 1000
TRAIN_BATCHES = 10
CONTAMINATION = 0.01  # Fraction of anomalies in the dataset
# ANOMALY = [0,1,2,3]
MAGNITUDE = 0.4


# Load and preprocess household data
def load_household_data(spark):
    path = 'data/inputs/household_power_consumption.csv'
    schema = StructType([
        StructField('Date', StringType(), True),
        StructField('Time', StringType(), True),
        StructField('Global_active_power', DoubleType(), True),
        StructField('Global_reactive_power', DoubleType(), True),
        StructField('Voltage', DoubleType(), True),
        StructField('Global_intensity', DoubleType(), True),
        StructField('Sub_metering_1', DoubleType(), True),
        StructField('Sub_metering_2', DoubleType(), True),
        StructField('Sub_metering_3', DoubleType(), True)
    ])
    df = spark.read.csv(path, schema=schema, sep=';', header=True)
    # df = df.withColumn(
    #     "Date",
    #     to_date(col("Date"), "dd/MM/yyyy")
    # )
    # df = df.withColumn(
    #     "Time",
    #     to_timestamp(col("Time"), "HH:mm:ss")
    # )
    # df = df.withColumn(
    #     "timestamp",
    #     to_timestamp(concat_ws(" ", col("Date"), col("Time")), "dd/MM/yyyy HH:mm:ss")
    # )
    # df = df.drop("Date", "Time")
    return df

# Compute descriptive statistics for each partition
def compute_statistics(df):
    total_rows = df.count()
    columns = [col_name for col_name in df.columns if col_name not in ["Date", "Time"]]
    agg_exprs = []

    for column in columns:
        agg_exprs.append((count(col(column).isNotNull()) / total_rows).alias(f"{column}_completeness"))
        agg_exprs.append(approx_count_distinct(col(column)).alias(f"{column}_approx_distinct_count"))
        agg_exprs.append(max(col(column)).alias(f"{column}_max"))
        agg_exprs.append(mean(col(column)).alias(f"{column}_mean"))
        agg_exprs.append(min(col(column)).alias(f"{column}_min"))
        agg_exprs.append(stddev(col(column)).alias(f"{column}_stddev"))

    agg_result = df.agg(*agg_exprs).collect()[0]
    return agg_result


def generate_explicit_missing_values(df, indexes):
    # Convert the DataFrame to an RDD and zip with index
    rdd_with_index = df.rdd.zipWithIndex()
    cols = df.columns

    # Function to modify rows at specified indexes
    def modify_row(row):
        row_data, row_index = row
        if row_index in indexes:
            # Randomly select columns to replace with NULL
            columns_to_null = random.sample(cols, random.randint(1, len(cols)))
            # Create a new row with NULL values for selected columns
            new_row = [None if col_name in columns_to_null else value for col_name, value in zip(cols, row_data)]
            return new_row
        else:
            return row_data

    # Apply the modification to the RDD
    modified_rdd = rdd_with_index.map(modify_row)

    return modified_rdd

# Perturb batch data
def generate_implicit_missing_values(df, rows_indexes):
    # Convert the DataFrame to an RDD and zip with index
    rdd_with_index = df.rdd.zipWithIndex()
    cols = df.columns
    col_types = {col_name: col_type for col_name, col_type in df.dtypes}

    # Function to modify rows at specified indexes
    def modify_row(row):
        row_data, row_index = row
        if row_index in rows_indexes:
            # Randomly select columns to replace with NULL
            columns_to_null = random.sample(cols, random.randint(1, len(cols)))
            # Create a new row with NULL values for selected columns
            value_to_replace = {
                "string": "NONE",
                "integer": 99999,
                "double": 99999.0
            }
            new_row = [value_to_replace[col_types[col_name]] if col_name in columns_to_null else value for col_name, value in zip(cols, row_data)]
            return new_row
        else:
            return row_data

    # Apply the modification to the RDD
    modified_rdd = rdd_with_index.map(modify_row)

    return modified_rdd


def generate_numeric_anomaly(df, rows_indexes):
    # Convert the DataFrame to an RDD and zip with index
    rdd_with_index = df.rdd.zipWithIndex()
    cols = df.columns
    col_types = {col_name: col_type for col_name, col_type in df.dtypes}
    agg_exprs = []
    for column in cols:
        if col_types[column] == "string":
            continue
        agg_exprs.append(mean(col(column)).alias(f"{column}_mean"))
        agg_exprs.append(stddev(col(column)).alias(f"{column}_stddev"))

    agg_result = df.agg(*agg_exprs).collect()[0].asDict()

    # Function to modify rows at specified indexes
    def modify_row(row):
        row_data, row_index = row
        if row_index in rows_indexes:
            # Randomly select columns to replace with NULL
            columns_to_modify = random.sample(cols, random.randint(1, len(cols)))
            # Create a new row with NULL values for selected columns
            get_value = lambda cname: random.gauss(agg_result[f"{cname}_mean"], agg_result[f"{cname}_stddev"]*random.randint(2, 5))
            new_row = [get_value(col_name) if col_name and col_types[col_name] != "string" in columns_to_modify else value for
                       col_name, value in zip(cols, row_data)]
            return new_row
        else:
            return row_data

    # Apply the modification to the RDD
    modified_rdd = rdd_with_index.map(modify_row)

    return modified_rdd


def generate_swapped_numeric_fields_anomaly(df, rows_indexes):
    # Convert the DataFrame to an RDD and zip with index
    rdd_with_index = df.rdd.zipWithIndex()
    cols = df.columns
    col_types = {col_name: col_type for col_name, col_type in df.dtypes}

    # Function to modify rows at specified indexes
    def modify_row(row):
        row_data, row_index = row
        if row_index in rows_indexes:
            col1, col2 = random.sample([c for c in cols if col_types[c] != "string"], 2)
            new_row = [row_data[cols.index(col2)] if col_name == col1 else row_data[cols.index(col1)] if col_name == col2 else value for col_name, value in zip(cols, row_data)]
            return new_row
        else:
            return row_data

    # Apply the modification to the RDD
    modified_rdd = rdd_with_index.map(modify_row)

    return modified_rdd


def perturb_batch(df, anomaly, magnitude):
    """
    Introduces synthetic anomalies into a batch of data. The types of anomalies include:
    - Explicit missing values: Replaces a fraction of values with NULL.
    - Implicit missing values: Replaces values with placeholders like 'NONE' or 99999.
    - Numeric anomalies: Introduces Gaussian noise to numeric values.
    - Swapped numeric fields: Swaps values between two numeric columns.
    - Swapped textual fields: Swaps values between two textual columns.
    - Typos: Introduces random typos in textual attributes based on keyboard proximity.
    """

    if random.random() < magnitude:
        target = 1
        rows_indexes = random.sample(range(0, df.count()), random.randint(1, int(0.1 * df.count())))
        if ANOMALY_TYPES[anomaly] == "explicit_missing_values":
            perturbed_rdd = generate_explicit_missing_values(df, rows_indexes)
        elif ANOMALY_TYPES[anomaly] == "implicit_missing_values":
            perturbed_rdd = generate_implicit_missing_values(df, rows_indexes)
        elif ANOMALY_TYPES[anomaly] == "numeric_anomalies":
            perturbed_rdd = generate_numeric_anomaly(df, rows_indexes)
        elif ANOMALY_TYPES[anomaly] == "swaped_numeric_fields":
            perturbed_rdd = generate_swapped_numeric_fields_anomaly(df, rows_indexes)
        elif ANOMALY_TYPES[anomaly] == "swapped_textual_fields":
            pass
        else:
            raise ValueError("Invalid anomaly type")
    else:
        perturbed_rdd = df.rdd
        target = 0

    return perturbed_rdd, target

# Train Isolation Forest model
def train_isolation_forest(final_data, n_samples=0.1):
    isolation_forest = IsolationForest() \
        .setBootstrap(False) \
        .setMaxSamples(int(n_samples*final_data.count())) \
        .setFeaturesCol("scaled_features") \
        .setPredictionCol("prediction") \
        .setScoreCol("outlierScore") \
        .setContamination(CONTAMINATION) \
        .setContaminationError(0.01 * CONTAMINATION) \
        .setRandomSeed(1)
    return isolation_forest.fit(final_data)

def split_household_data(df):
    dates = df.select(col("Date")).distinct().collect()
    df_splits = [df.filter(col("Date") == row["Date"]) for row in dates]
    return df_splits[:100]

def run_experiment(df_splits, train_batches=TRAIN_BATCHES, magnitude=MAGNITUDE, anomaly=None, n_samples=0.1):
    random.seed(1)
    gt = []
    pred = []
    train_time = []
    test_time = []
    for i in trange(len(df_splits)-train_batches):
        initial_time = time.time()
        training_data = spark.createDataFrame([compute_statistics(batch) for batch in df_splits[:train_batches+i]])
        statistics_time = time.time() - initial_time
        test_batch = df_splits[train_batches+i]
        chosen_anomaly = random.choice(anomaly)
        perturbed_rdd, target = perturb_batch(test_batch, chosen_anomaly, magnitude)
        gt.append(target)
        test_data = spark.createDataFrame(perturbed_rdd, test_batch.schema)

        # Prepare and normalize feature vectors
        initial_time = time.time()
        assembler = VectorAssembler(inputCols=[column for column in training_data.columns], outputCol="features")
        scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
        feature_vectors = assembler.transform(training_data)
        scaler_model = scaler.fit(feature_vectors)
        scaled_data =  scaler_model.transform(feature_vectors)
        final_data = scaled_data.select("scaled_features")

        # Train model
        isolation_forest_model = train_isolation_forest(final_data, n_samples=n_samples)
        train_time.append(time.time() - initial_time + statistics_time)

        # Detect anomalies in new data
        initial_time = time.time()
        test_data = spark.createDataFrame([compute_statistics(test_data)])
        feature_vectors = assembler.transform(test_data)
        scaled_data = scaler_model.transform(feature_vectors)
        final_data = scaled_data.select("scaled_features")
        predictions = isolation_forest_model.transform(final_data)
        pred.append(predictions.collect()[0]["prediction"])
        test_time.append(time.time() - initial_time)

    results_df = pd.DataFrame({'ground_truth': gt, 'predictions': pred, 'train_time': train_time, 'test_time': test_time})
    error_string = ','.join([ANOMALY_TYPES[a] for a in anomaly])
    results_df.to_csv(f'data/outputs/results_magnitude={magnitude}_ERROR={error_string}.csv', index=False)

def main():
    # Load household data
    df = load_household_data(spark)
    # Split df into batches
    df_splits = split_household_data(df)
    spark.sparkContext.setLogLevel("ERROR")
    for n_sample in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        run_experiment(df_splits, anomaly=[0, 1, 2, 3], n_samples=n_sample)
    spark.stop()

if __name__ == '__main__':
    main()