import random
from math import ceil

from pyspark.sql.functions import mean, col, stddev

ANOMALY_TYPES = ["explicit_missing_values", "implicit_missing_values", "numeric_anomalies", "swaped_numeric_fields", "swapped_textual_fields", "typos"]


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
        if col_types[column] != "double":
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
        rows_indexes = random.sample(range(0, df.count()), random.randint(1, ceil(0.1 * df.count())))
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