import math
from itertools import groupby
from math import log, sqrt

from pyspark.sql.functions import col, count, mean, stddev, max, min, approx_count_distinct, to_timestamp, \
    concat_ws
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType, TimestampType
from pyspark.sql import Row

def peculiarity_index( df, column):
    def compute_pc_idx(row):
        string = row[0]
        char_string = list(string)
        trigrams = [char_string[i] + char_string[i + 1] + char_string[i + 2] for i in range(len(char_string) - 2)]
        tri_count = {}
        bi_count = {}
        for trigram in trigrams:
            bigram1 = trigram[:2]
            bigram2 = trigram[1:]
            tri_count[trigram] = tri_count.get(trigram, 0) + 1
            bi_count[bigram1] = bi_count.get(bigram1, 0) + 1
            bi_count[bigram2] = bi_count.get(bigram2, 0) + 1

        t = [
            (0.5 * (log(bi_count.get(trigram[:2], 1e-6)) + log(bi_count.get(trigram[1:], 1e-6))) - log(num)) ** 2
            for trigram, num in tri_count.items()
        ]

        return sqrt(sum(t) / (len(t) + 1e-6))

    pc_idx = df.select(column).filter(col(column).isNotNull()).rdd.map(compute_pc_idx).max()

    return float(pc_idx)

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
        StructField('InvoiceNo', StringType(), True),
        StructField('StockCode', StringType(), True),
        StructField('Description', StringType(), True),
        StructField('Quantity', DoubleType(), True),
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
    df = df.withColumn("Quantity", col("Quantity").cast("double"))
    df = df.withColumn("InvoiceNo", col("InvoiceNo").cast("double"))
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
            agg_exprs.append(max(col(column)).cast(DoubleType()).alias(f"{column}_max"))
            statistics_schema.append(StructField(f"{column}_max", DoubleType(), True))
            agg_exprs.append(mean(col(column)).cast(DoubleType()).alias(f"{column}_mean"))
            statistics_schema.append(StructField(f"{column}_mean", DoubleType(), True))
            agg_exprs.append(min(col(column)).cast(DoubleType()).alias(f"{column}_min"))
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
