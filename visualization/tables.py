"""CSV table-writing helpers."""
import pandas as pd
import numpy as np
import os


def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def matrix_table(name, matrix):
    rows=[]
    for i,row in enumerate(np.asarray(matrix)):
        rows.append([name, i, *(float(v) for v in row)])
    return rows
