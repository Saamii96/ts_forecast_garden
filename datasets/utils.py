import numpy as np
import pandas as pd
import psutil
import os


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce the memory usage of a pandas dataframe.
    
    Args:
        df: pandas dataframe to reduce size
        verbose: whether to print the memory usage reduction
    
    Returns:
        pd.DataFrame: the reduced dataframe
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



def get_memory_usage() -> float:
    """
    Get the memory usage of the current process.
    
    Returns:
        float: the memory usage in GB
    """
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 



def sizeof_fmt(num: float, suffix: str = 'B') -> str    :
    """
    Format the memory usage in a human readable format.
    
    Args:
        num: the memory usage in bytes
        suffix: the suffix to add to the memory usage
    Returns:
        str: the memory usage in a human readable format
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)



def merge_by_concat(df1: pd.DataFrame, df2: pd.DataFrame, merge_on: list) -> pd.DataFrame:
    """
    Merge two dataframes by concatenating the merge_on columns.
    
    Args:
        df1: first dataframe
        df2: second dataframe
        merge_on: list of columns to merge on
    Returns:
        pd.DataFrame: the merged dataframe
    """
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)

    return df1