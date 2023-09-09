#imports

import pandas as pd
import numpy as np
import os

from scipy.stats import shapiro, spearmanr


def area_vs_value_tst(df):
    # Hypothesis 1: Property Size Impact
    r, p = spearmanr(df['area'], df['value'])

    if p < 0.05:
        print(f"Result: There is a significant monotonic relationship between property size and value (p-value={p}).")
    else:
        print(f"Result: There is no significant monotonic relationship between property size and value (p-value={p}).")


def perform_spearmanr_test(df, col_name):
    """
    Perform a Spearman rank correlation test between the specified column and 'value' column in the DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame containing the data.
    - col_name (str): The name of the column to test for correlation with 'value'.

    Returns:
    None: Prints the result of the Spearman rank correlation test.
    """
    r, p = spearmanr(df[col_name], df['value'])
    
    if p < 0.05:
        print(f"Result: There is a significant monotonic relationship between {col_name} and value (p-value={p}).")
    else:
        print(f"Result: There is no significant monotonic relationship between {col_name} and value (p-value={p}).")
