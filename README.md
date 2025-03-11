# Automated Data Quality Validation

This project reproduces the results of the paper *Automating Data Quality Validation for Dynamic Data Ingestion* using PySpark and the Isolation Forest algorithm. The goal is to validate data quality in dynamic environments by detecting anomalies in large-scale datasets.

## Prerequisites

Before running the code, ensure the following software is installed on your system:

- **Java 11**
- **Python 3.12**
- **Apache Spark 3.5.5**

## Installation

1. **Install Python Libraries**:  
   Install all required Python libraries by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Spark**:  
   Ensure that Spark is properly configured and that the `SPARK_HOME` environment variable is set.

3. **Prepare Data**:  
   The data is divided into two directories: `data/inputs` and `data/outputs`. The input files are provided in the `inputs.tar.xz` archive. To prepare the data, follow these steps:
   ```bash
   tar -xf inputs.tar.xz
   mv inputs/* data/inputs/
   ```
   This will uncompress the input files and place them in the `data/inputs` directory.

## Running the Code

### Reproduce All Results
To reproduce all the results from the project, simply run the `run_all.sh` script. This script contains all the necessary configurations to execute the experiments:
```bash
./run_all.sh
```

### Run Specific Configurations
If you want to run a specific configuration, use the `project.py` script with the appropriate arguments. To see the available arguments, run:
```bash
python project.py -h
```

Example usage:
```bash
python project.py --dataset metropt3 --batch_size 1000
```

## Project Structure

- **`project.py`**: Main script for running the data quality validation pipeline.
- **`run_all.sh`**: Bash script to reproduce all experiments.
- **`requirements.txt`**: List of Python dependencies.
- **`data/`**: Directory containing the datasets used in the experiments.
  - **`inputs/`**: Contains the input datasets.
  - **`outputs/`**: Stores the results of the experiments.
- **`results/`**: Directory where the results of the experiments are saved.

## Datasets

The following datasets were used in this project:
- **Metro PT-3**: Air Production Unit (APU) readings from a metro train.
- **Individual Household Electric Power Consumption**: Electric power consumption measurements from a single household.
- **Online Retail**: Transactional data from a UK-based online retail company.
## Acknowledgments

This project is based on the paper *Automating Data Quality Validation for Dynamic Data Ingestion* by Redyuk et al. (2021).