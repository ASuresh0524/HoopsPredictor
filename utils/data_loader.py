import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import Optional, List, Dict, Union
import yaml
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class to handle loading and preprocessing of NBA data from SQLite databases.
    
    Attributes:
        team_db_path (str): Path to the team-level SQLite database
        player_db_path (str): Path to the player-level SQLite database
        config (dict): Configuration dictionary loaded from config.yaml
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the DataLoader with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.team_db_path = self.config['data']['team_data_path']
        self.player_db_path = self.config['data']['player_data_path']
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
    def get_table_names(self, database: str = 'team') -> List[str]:
        """
        Get all table names from specified database.
        
        Args:
            database (str): Either 'team' or 'player'
            
        Returns:
            List[str]: List of table names
        """
        conn = sqlite3.connect(self.team_db_path if database == 'team' else self.player_db_path)
        return pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()
    
    def load_team_data(self, start_date, end_date):
        """
        Load team data for a specific date range.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Team statistics for the specified date range
        """
        try:
            conn = sqlite3.connect(self.team_db_path)
            
            # Get all table names (dates) in the database
            dates_query = "SELECT name FROM sqlite_master WHERE type='table'"
            dates = pd.read_sql_query(dates_query, conn)['name'].tolist()
            
            # Filter dates within the specified range
            valid_dates = [
                date for date in dates 
                if start_date <= date <= end_date
            ]
            
            if not valid_dates:
                self.logger.warning(f"No data found between {start_date} and {end_date}")
                return pd.DataFrame()
            
            # Load and concatenate data from each date
            dfs = []
            for date in valid_dates:
                query = f'SELECT *, "{date}" as Date FROM "{date}"'
                df = pd.read_sql_query(query, conn)
                dfs.append(df)
            
            team_data = pd.concat(dfs, ignore_index=True)
            conn.close()
            
            return team_data
            
        except Exception as e:
            self.logger.error(f"Error loading team data: {str(e)}")
            raise

    def load_game_data(self, start_date, end_date):
        """
        Load game data for a specific date range.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Game data including matchups and outcomes
        """
        try:
            conn = sqlite3.connect(self.player_db_path)
            
            query = f"""
            SELECT * FROM dataset_2012-24 
            WHERE Date >= '{start_date}' 
            AND Date <= '{end_date}'
            """
            
            game_data = pd.read_sql_query(query, conn)
            conn.close()
            
            return game_data
            
        except Exception as e:
            self.logger.error(f"Error loading game data: {str(e)}")
            raise

    def get_train_val_test_split(self):
        """
        Get training, validation and test datasets based on config dates.
        
        Returns:
            tuple: (train_data, val_data, test_data) for both team and game data
        """
        # Load training data
        train_team = self.load_team_data(
            self.config['data']['train_start_date'],
            self.config['data']['train_end_date']
        )
        train_games = self.load_game_data(
            self.config['data']['train_start_date'],
            self.config['data']['train_end_date']
        )
        
        # Load validation data
        val_team = self.load_team_data(
            self.config['data']['val_start_date'],
            self.config['data']['val_end_date']
        )
        val_games = self.load_game_data(
            self.config['data']['val_start_date'],
            self.config['data']['val_end_date']
        )
        
        # Load test data
        test_team = self.load_team_data(
            self.config['data']['test_start_date'],
            self.config['data']['test_end_date']
        )
        test_games = self.load_game_data(
            self.config['data']['test_start_date'],
            self.config['data']['test_end_date']
        )
        
        return {
            'team': (train_team, val_team, test_team),
            'games': (train_games, val_games, test_games)
        }

    def get_latest_team_stats(self, date=None):
        """
        Get the most recent team statistics.
        
        Args:
            date (str, optional): Specific date to get stats for. If None, gets most recent.
            
        Returns:
            pd.DataFrame: Most recent team statistics
        """
        try:
            conn = sqlite3.connect(self.team_db_path)
            
            # Get all table names (dates)
            dates_query = "SELECT name FROM sqlite_master WHERE type='table'"
            dates = pd.read_sql_query(dates_query, conn)['name'].tolist()
            
            if date:
                if date not in dates:
                    self.logger.warning(f"No data found for date {date}")
                    return pd.DataFrame()
                target_date = date
            else:
                target_date = max(dates)
            
            query = f'SELECT * FROM "{target_date}"'
            latest_stats = pd.read_sql_query(query, conn)
            conn.close()
            
            return latest_stats
            
        except Exception as e:
            self.logger.error(f"Error getting latest team stats: {str(e)}")
            raise
        
    def preprocess_team_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess team-level data with standard cleaning operations.
        
        Args:
            df (pd.DataFrame): Raw team data
            
        Returns:
            pd.DataFrame: Preprocessed team data
        """
        # Ensure datetime columns are properly formatted
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
            
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Add season identifier if not present
        if 'season' not in df.columns and 'game_date' in df.columns:
            df['season'] = df['game_date'].dt.year.where(
                df['game_date'].dt.month < 8,
                df['game_date'].dt.year + 1
            )
            
        return df
        
    def preprocess_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess player-level data with standard cleaning operations.
        
        Args:
            df (pd.DataFrame): Raw player data
            
        Returns:
            pd.DataFrame: Preprocessed player data
        """
        # Similar datetime handling
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
            
        # Handle missing values differently for player stats
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                # Use player's average for the season if available
                df[col] = df.groupby(['player_id', 'season'])[col].transform(
                    lambda x: x.fillna(x.mean())
                )
                # If still null, use overall average
                df[col] = df[col].fillna(df[col].mean())
                
        # Add games played counter
        if 'player_id' in df.columns and 'game_date' in df.columns:
            df['games_played'] = df.groupby('player_id').cumcount() + 1
            
        return df 