from datasetsforecast.m5 import M5
import pandas as pd
import logging
from pathlib import Path
import os
import numpy as np
import sys
from typing import Optional, Tuple
sys.path.append('./../')

from datasets.utils import reduce_mem_usage, get_memory_usage, sizeof_fmt, merge_by_concat


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class M5Dataset:
    """
    M5 Forecasting Competition Dataset handler.
    
    This class provides methods to load, process, and manage the M5 dataset
    from the datasetsforecast library.
    """
    
    def __init__(self, data_dir: str, target: str = 'sales', end_train: int = 1913):
        """
        Initialize M5Dataset.
        
        Args:
            data_dir: Directory where the M5 dataset will be stored
            target: Target column name (default: 'sales')
            end_train: Last training day (default: 1913)
        """
        self.data_dir = Path(data_dir)
        self.target = target
        self.end_train = end_train
        self.main_index = ['id', 'd']
        
        # Dataset paths
        self.datasets_dir = self.data_dir / 'm5' / 'datasets'
        self.processed_dir = self.data_dir / 'm5' / 'processed'
        
        # Cached data
        self._Y_df: Optional[pd.DataFrame] = None
        self._X_df: Optional[pd.DataFrame] = None
        self._S_df: Optional[pd.DataFrame] = None
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
    
    def load_dataset(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """
        Load M5 dataset using datasetsforecast library from Nixtla.
        
        Returns:
            Tuple of (Y_df, X_df, S_df) - target series, exogenous variables, static data
        """    
        logger.info(f"Loading M5 dataset to: {self.data_dir}")
        Y_df, X_df, S_df = M5.load(directory=str(self.data_dir))
        
        # Cache the data
        self._Y_df = Y_df
        self._X_df = X_df
        self._S_df = S_df
        
        logger.info(f"Target series shape: {Y_df.shape}")
        logger.info(f"Exogenous variables shape: {X_df.shape if X_df is not None else 'None'}")
        logger.info(f"Static data shape: {S_df.shape}")
        
        logger.info(f"Target series columns: {Y_df.columns.tolist()}")
        logger.info("First few rows of target series:")
        print(Y_df.head())
        
        return Y_df, X_df, S_df
    
    def is_dataset_loaded(self) -> bool:
        """Check if the M5 dataset is already loaded."""
        return (self.datasets_dir.exists() and 
                any(self.datasets_dir.iterdir()) and
                self._Y_df is not None)
    
    def create_grid_1(self) -> pd.DataFrame:
        """
        Create M5 grid 1.

        Returns:
            pd.DataFrame: the processed grid
        """
        train_df = pd.read_csv(self.datasets_dir / 'sales_train_validation.csv')
        train_df['id'] = train_df.item_id + '_' + train_df.store_id + '_validation'

        prices_df = pd.read_csv(self.datasets_dir / 'sell_prices.csv')

        calendar_df = pd.read_csv(self.datasets_dir / 'calendar.csv')
        calendar_df['d'] = np.arange(calendar_df.shape[0]) + 1
        calendar_df['d'] = 'd_' + calendar_df['d'].astype('str')
        calendar_df['d'] = calendar_df['d'].astype('category')

        logger.info('Create Grid')

        index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
        grid_df = pd.melt(train_df, 
                        id_vars = index_columns, 
                        var_name = 'd', 
                        value_name = self.target)

        logger.info(f'Train rows: {len(train_df)}, {len(grid_df)}')

        add_grid = pd.DataFrame()
        for i in range(1, 29):
            temp_df = train_df[index_columns]
            temp_df = temp_df.drop_duplicates()
            temp_df['d'] = 'd_'+ str(self.end_train+i)
            temp_df[self.target] = np.nan
            add_grid = pd.concat([add_grid,temp_df])

        grid_df = pd.concat([grid_df,add_grid])
        grid_df = grid_df.reset_index(drop=True)
        del temp_df, add_grid, train_df

        logger.info(f'Original grid_df: {sizeof_fmt(grid_df.memory_usage(index=True).sum())}')

        for col in index_columns:
            grid_df[col] = grid_df[col].astype('category')

        logger.info(f'Reduced grid_df: {sizeof_fmt(grid_df.memory_usage(index=True).sum())}')

        logger.info('Release week')
        release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
        release_df.columns = ['store_id','item_id','release']

        grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])
        del release_df

        grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])
        grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]
        grid_df = grid_df.reset_index(drop=True)

        logger.info(f'Original grid_df: {sizeof_fmt(grid_df.memory_usage(index=True).sum())}')

        grid_df['release'] = grid_df['release'] - grid_df['release'].min()
        grid_df['release'] = grid_df['release'].astype(np.int16)

        logger.info(f'Reduced grid_df: {sizeof_fmt(grid_df.memory_usage(index=True).sum())}')

        logger.info('Save Part 1')
        grid_df.to_pickle(self.processed_dir / 'grid_part_1.pkl')
        logger.info('Size:', grid_df.shape)

        return grid_df




if __name__ == "__main__":
    # Get the project root directory (one level up from datasets folder)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'

    # Create M5Dataset instance
    m5_dataset = M5Dataset(data_dir=str(data_dir))

    logger.info("Loading M5 dataset...")

    # Load dataset if not already loaded
    if not m5_dataset.is_dataset_loaded():
        m5_dataset.load_dataset()
    
    logger.info("M5 dataset loaded successfully!")

    # Create and save processed data
    # grid_1 = m5_dataset.create_grid_1()
