import os
import pandas as pd

from config import src_dir


def generate_mean_column(
        input_csv: str,
        output_csv: str,
        target_column: str,
        interval: int = 48
):
    """
    Reads a CSV file, computes the mean of the target column over the specified interval,
    and writes a new CSV file with the computed mean column.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
        target_column (str): Column name in the input CSV for which the mean is computed.
        interval (int): Number of timesteps to compute the mean (default is 48 for 1-hour mean).
    """
    # Read the original CSV file
    print(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv, index_col='Date', parse_dates=['Date'])

    # Compute the mean over the specified interval
    print(f"Computing mean over the next {interval} timesteps...")
    mean_48 = []
    for i in range(len(df)):
        if i + interval < len(df):
            mean_val = df[target_column][i + 1:i + 1 + interval].mean()  # Compute mean for next 48 timesteps
        else:
            mean_val = df[target_column][i:].mean()  # Use remaining timesteps at the end of the data
        mean_48.append(mean_val)

        # Create a new DataFrame with the computed mean column
    mean_df = pd.DataFrame({'Date': df.index, 'mean_48': mean_48})

    # Save to a new CSV file
    print(f"Saving mean column to {output_csv}...")
    mean_df.to_csv(output_csv, index=False)
    print("Done!")


# Example usage
if __name__ == "__main__":
    # Paths to input and output CSV files
    input_csv = os.path.join(src_dir, 'data', 'fine_res', 'on_2016_1.25min.csv')
    output_csv = os.path.join(src_dir, 'data', 'fine_res', 'on_2016_1.25min_mean_demand.csv')
    target_column = "demand"  # Column name for the target variable in the input CSV
    interval = 48

    # Generate the mean column and save to a new CSV
    generate_mean_column(input_csv, output_csv, target_column)