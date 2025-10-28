from datasetsforecast.long_horizon import LongHorizon
import pandas as pd
import logging
from pathlib import Path
import os
import numpy as np
import sys
import shutil
from typing import Optional, Tuple
sys.path.append('./../')

from datasets.utils import reduce_mem_usage, get_memory_usage, sizeof_fmt, merge_by_concat


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ETTh1Dataset:
    """
    ETTh1 Dataset handler for Electricity Transformer Temperature.
    
    This class provides methods to load, process, and manage ETTh1 dataset
    from the datasetsforecast library. The ETTh1 dataset contains 7 variables
    collected at 15-minute intervals over two years, including the target variable
    "Oil Temperature (OT)" and six power load features.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize ETTh1Dataset.
        
        Args:
            data_dir: Directory where the dataset will be stored
            name: Dataset name (default: 'ETTm2')
            freq: Frequency of the time series (default: '15T' for 15 minutes)
            n_ts: Number of time series (default: 7)
            test_size: Test set size (default: 11520)
            val_size: Validation set size (default: 11520)
            horizons: Forecasting horizons (default: (96, 192, 336, 720))
        """
        self.data_dir = Path(data_dir)
        self.name = 'ETTh1'
        
        # Dataset paths - clean structure like M5
        self.datasets_dir = self.data_dir / self.name / 'datasets'
        self.processed_dir = self.data_dir / self.name / 'processed'
        
        # LongHorizon creates a nested structure we'll reorganize
        self.longhorizon_dir = self.data_dir / self.name / 'longhorizon'
        
        # Cached data
        self._Y_df: Optional[pd.DataFrame] = None
        self._X_df: Optional[pd.DataFrame] = None
        self._S_df: Optional[pd.DataFrame] = None
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """
        Load long horizon dataset using datasetsforecast library.
        
        Returns:
            Tuple of (Y_df, X_df, S_df) - target series, exogenous variables, static data
        """    
        logger.info(f"Loading {self.name} dataset to: {self.data_dir}")
        Y_df, X_df, S_df = LongHorizon.load(directory=str(self.data_dir / self.name), group=self.name)
        
        # Cache the data
        self._Y_df = Y_df
        self._X_df = X_df
        self._S_df = S_df
        
        # Reorganize files from longhorizon/datasets/ETTh1 to datasets/ETTh1
        self._reorganize_files()
        
        logger.info(f"Target series shape: {Y_df.shape}")
        logger.info(f"Exogenous variables shape: {X_df.shape if X_df is not None else 'None'}")
        logger.info(f"Static data shape: {S_df.shape if S_df is not None else 'None'}")
        
        logger.info(f"Target series columns: {Y_df.columns.tolist()}")
        logger.info("First few rows of target series:")
        print(Y_df.head())
        
        return Y_df, X_df, S_df
    
    def _reorganize_files(self):
        """Reorganize files from longhorizon structure to clean structure."""
        source_dir = self.longhorizon_dir / 'datasets' / self.name
        target_dir = self.datasets_dir
        
        if source_dir.exists() and not target_dir.exists():
            logger.info(f"Reorganizing files from {source_dir} to {target_dir}")
            
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy only the main dataset files, exclude S folder
            for item in source_dir.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    logger.info(f"Copying {item.name}")
                    shutil.copy2(item, target_dir)
                elif item.is_dir() and item.name != 'S':
                    logger.info(f"Copying directory {item.name}")
                    shutil.copytree(item, target_dir / item.name)
                elif item.name == 'S':
                    logger.info(f"Skipping duplicate S folder")
            
            logger.info(f"Files reorganized. Cleaning up longhorizon directory...")
            shutil.rmtree(self.longhorizon_dir, ignore_errors=True)
            logger.info("Cleanup complete.")
        elif target_dir.exists():
            logger.info(f"Dataset already organized in {target_dir}")
    
    def is_dataset_loaded(self) -> bool:
        """Check if the long horizon dataset is already loaded."""
        return (self.datasets_dir.exists() and 
                any(self.datasets_dir.iterdir()) and
                self._Y_df is not None)


if __name__ == "__main__":
    # Get the project root directory (one level up from datasets folder)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'

    # Create ETTh1Dataset instance
    long_horizon_dataset = ETTh1Dataset(data_dir=str(data_dir))

    logger.info("Loading long horizon dataset...")

    # Load dataset if not already loaded
    if not long_horizon_dataset.is_dataset_loaded():
        long_horizon_dataset.load_dataset()
    
    logger.info("Long horizon dataset loaded successfully!")