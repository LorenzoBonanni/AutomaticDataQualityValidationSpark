import math
from itertools import groupby
from math import log, sqrt

from pyspark.sql.functions import col, count, mean, stddev, max, min, approx_count_distinct, to_timestamp, \
    concat_ws
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType, TimestampType
from pyspark.sql import Row

def peculiarity_index( df, column):
    trigrams_count = (
        df.select(column).filter(col(column).isNotNull())
        .rdd.map(lambda row: list(row[0]))  # Convert each string to a character list (remove spaces)
        .map(lambda chars: ([chars[i] + chars[i + 1] + chars[i + 2] for i in range(len(chars) - 2)] if len(
            chars) > 2 else []))  # Generate trigrams per string
        .map(lambda trigrams: [(trigram, 1) for trigram in trigrams])  # Convert each trigram into (trigram, 1)
        .map(lambda trigram_list: dict(
            (trigram, sum(num for _, num in group)) for trigram, group in
            groupby(sorted(trigram_list, key=lambda x: x[0]), key=lambda x: x[0])
        ))  # Sort bigram list and count occurrences
    )

    bigram_count = (
        df.select(column).filter(col(column).isNotNull())
        .rdd.map(lambda row: list(row[0]))  # Convert each string to a character list
        .map(lambda chars: (
            [chars[i] + chars[i + 1] for i in range(len(chars) - 1)] if len(chars) > 1 else []
        ))  # Generate bigrams per string
        .map(lambda bigrams: [(bigram, 1) for bigram in bigrams])  # Convert each bigram into (bigram, 1)
        .map(lambda bigram_list: dict(
            (bigram, sum(num for _, num in group)) for bigram, group in
            groupby(sorted(bigram_list, key=lambda x: x[0]), key=lambda x: x[0])
        ))  # Sort bigram list and count occurrences
    ).collect()

    idx_trigrams_count = trigrams_count.zipWithIndex()


    pecuniarity_index = (idx_trigrams_count
        .map(
            lambda x: [
                (trigram,
                 0.5 * (log(bigram_count[x[1]].get(trigram[:2], 1e-6)) + log(bigram_count[x[1]].get(trigram[1:], 1e-6))) - log(num))
                for trigram, num in x[0].items()
            ]
        )
        .map(lambda lst: [(trigram, val ** 2) for trigram, val in lst])
        .map( lambda lst: sqrt(sum(val for _, val in lst) / (len(lst)+1e-6)))
    ).max()
    return pecuniarity_index


def split_data(df, args):
    """
    Split the DataFrame into batches of a specified size.

    Parameters:
    df (DataFrame): The input DataFrame to be split.
    args (Namespace): Arguments containing the batch size.

    Returns:
    list: A list of DataFrame batches.
    """
    BATCH_SIZE = args.batch_size
    df_splits = [df.offset(i * BATCH_SIZE).limit(BATCH_SIZE) for i in
                 range((df.count() + BATCH_SIZE - 1) // BATCH_SIZE)]
    return df_splits[:200]


def load_online_retail_data(spark):
    path = 'data/inputs/Online Retail.csv'
    schema = StructType([
        StructField('InvoiceNo', IntegerType(), True),
        StructField('StockCode', StringType(), True),
        StructField('Description', StringType(), True),
        StructField('Quantity', IntegerType(), True),
        StructField('InvoiceDate', StringType(), True),
        StructField('UnitPrice', DoubleType(), True),
        StructField('CustomerID', StringType(), True),
        StructField('Country', StringType(), True)
    ])
    df = spark.read.csv(path, sep=',', header=True, schema=schema)
    df = df.withColumn(
        "timestamp",
        to_timestamp(col("InvoiceDate"), "dd/MM/yyyy HH:mm")
    )
    df = df.drop("InvoiceDate", "CustomerID")
    return df


def load_household_data(spark):
    """
    Load and preprocess household data.

    Parameters:
    spark (SparkSession): The Spark session.

    Returns:
    DataFrame: The preprocessed household data.
    """
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
    df = df.withColumn(
        "timestamp",
        to_timestamp(concat_ws(" ", col("Date"), col("Time")), "dd/MM/yyyy HH:mm:ss")
    )
    df = df.drop("Date", "Time")
    return df


def load_metropt3_data(spark):
    """
    Load and preprocess MetroPT3 data.

    Parameters:
    spark (SparkSession): The Spark session.

    Returns:
    DataFrame: The preprocessed MetroPT3 data.
    """
    path = 'data/inputs/MetroPT3(AirCompressor).csv'
    schema = StructType([
        StructField('index', IntegerType(), True),
        StructField('timestamp', TimestampType(), True),
        StructField('TP2', DoubleType(), True),
        StructField('TP3', DoubleType(), True),
        StructField('H1', DoubleType(), True),
        StructField('DV_pressure', DoubleType(), True),
        StructField('Reservoirs', DoubleType(), True),
        StructField('Oil_temperature', DoubleType(), True),
        StructField('Motor_current', DoubleType(), True),
        StructField('COMP', DoubleType(), True),
        StructField('DV_eletric', DoubleType(), True),
        StructField('Towers', DoubleType(), True),
        StructField('MPG', DoubleType(), True),
        StructField('LPS', DoubleType(), True),
        StructField('Pressure_switch', DoubleType(), True),
        StructField('Oil_level', DoubleType(), True),
        StructField('Caudal_impulses', DoubleType(), True)
    ])
    df = spark.read.csv(path, sep=',', header=True, schema=schema)
    df = df.drop("index")

    return df


def get_df_splits(spark, args):
    """
    Load the dataset and split it into batches.

    Parameters:
    spark (SparkSession): The Spark session.
    args (Namespace): Arguments containing the dataset name and batch size.

    Returns:
    list: A list of DataFrame batches.

    Raises:
    ValueError: If the dataset name is invalid.
    """
    if args.dataset == 'household':
        # Load household data
        df = load_household_data(spark)
        # Split df into batches
        return split_data(df, args)
    elif args.dataset == 'metropt3':
        # Load metropt3 data
        df = load_metropt3_data(spark)
        # Split df into batches
        return split_data(df, args)
    elif args.dataset == 'onlineretail':
        # Load online retail data
        df = load_online_retail_data(spark)
        # Split df into batches
        return split_data(df, args)
    else:
        raise ValueError("Invalid dataset")


# Compute descriptive statistics for each partition
def compute_statistics(df):
    """
    Compute descriptive statistics for each partition of the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    Row: A Row object containing the computed statistics.
    """
    total_rows = df.count()
    columns = [col_name for col_name in df.columns if col_name not in ['timestamp']]
    agg_exprs = []
    pec_idx = {}
    statistics_schema = []

    for column in columns:
        agg_exprs.append((count(col(column).isNotNull()) / total_rows).alias(f"{column}_completeness"))
        statistics_schema.append(StructField(f"{column}_completeness", DoubleType(), True))
        agg_exprs.append((approx_count_distinct(col(column))/total_rows).alias(f"{column}_approx_distinct_count"))
        statistics_schema.append(StructField(f"{column}_approx_distinct_count", DoubleType(), True))
        if isinstance(df.schema[column].dataType, (DoubleType, IntegerType)):
            agg_exprs.append(max(col(column)).cast('float').alias(f"{column}_max"))
            statistics_schema.append(StructField(f"{column}_max", DoubleType(), True))
            agg_exprs.append(mean(col(column)).cast('float').alias(f"{column}_mean"))
            statistics_schema.append(StructField(f"{column}_mean", DoubleType(), True))
            agg_exprs.append(min(col(column)).cast('float').alias(f"{column}_min"))
            statistics_schema.append(StructField(f"{column}_min", DoubleType(), True))
            agg_exprs.append(stddev(col(column)).alias(f"{column}_stddev"))
            statistics_schema.append(StructField(f"{column}_stddev", DoubleType(), True))
        else:
            pec_idx[f"{column}_pcidx"] = peculiarity_index(df, column)
            statistics_schema.append(StructField(f"{column}_pcidx", DoubleType(), True))

    agg_result = df.agg(*agg_exprs).collect()[0]
    if pec_idx:
        agg_result = Row(**(agg_result.asDict() | pec_idx))
    return agg_result, StructType(statistics_schema)
