from datasetsforecast.m5 import M5
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_m5_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load M5 dataset using datasetsforecast library from Nixtla.
    
    Returns:
        tuple: (Y_df, X_df, S_df) containing target series, exogenous variables, and static data
    """
    # Get the project root directory (one level up from datasets folder)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    # Ensure the data directory exists
    data_dir.mkdir(exist_ok=True)
    
    logger.info(f"Loading M5 dataset to: {data_dir}")
    Y_df, X_df, S_df = M5.load(directory=str(data_dir))
    
    logger.info(f"Target series shape: {Y_df.shape}")
    logger.info(f"Exogenous variables shape: {X_df.shape if X_df is not None else 'None'}")
    logger.info(f"Static data shape: {S_df.shape}")
    
    logger.info("\nTarget series columns:", Y_df.columns.tolist())
    logger.info("First few rows of target series:")
    print(Y_df.head())
    
    return Y_df, X_df, S_df


if __name__ == "__main__":
    logger.info("Loading M5 dataset...")
    Y_df, X_df, S_df = load_m5_dataset()
    logger.info("M5 dataset loaded successfully!")