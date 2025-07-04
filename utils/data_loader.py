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
    
    def load_team_data(self, start_date, end_date=None):
        """
        Load team data from SQLite database for a given date range.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Team statistics for the specified date range
        """
        if end_date is None:
            end_date = start_date
            
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dfs = []
        current = start
        
        while current <= end:
            table_name = current.strftime('%Y-%m-%d')
            
            try:
                with sqlite3.connect(self.team_db_path) as conn:
                    query = f'SELECT * FROM "{table_name}"'
                    df = pd.read_sql_query(query, conn)
                    df['Date'] = table_name
                    dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Could not load data for {table_name}: {str(e)}")
            
            current += timedelta(days=1)
        
        if not dfs:
            raise ValueError(f"No data found between {start_date} and {end_date}")
        
        team_data = pd.concat(dfs, ignore_index=True)
        return team_data

    def load_player_data(self, start_date=None, end_date=None):
        """
        Load player data from SQLite database for a given date range.
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Player statistics for the specified date range
        """
        with sqlite3.connect(self.player_db_path) as conn:
            # Get the most recent table
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                conn
            )
            table_name = tables['name'].max()
            
            query = f'SELECT * FROM "{table_name}"'
            if start_date and end_date:
                query += f' WHERE Date BETWEEN "{start_date}" AND "{end_date}"'
            elif start_date:
                query += f' WHERE Date >= "{start_date}"'
            elif end_date:
                query += f' WHERE Date <= "{end_date}"'
            
            df = pd.read_sql_query(query, conn)
            
            # Rename duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            
            return df

    def load_game_data(self, start_date, end_date=None):
        """
        Load game data by combining team data into matchups.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Game matchup data for the specified date range
        """
        team_data = self.load_team_data(start_date, end_date)
        
        # Create game matchups
        games = []
        dates = team_data['Date'].unique()
        
        for date in dates:
            date_games = team_data[team_data['Date'] == date]
            teams = date_games['TEAM_NAME'].unique()
            
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    home_team = teams[i]
                    away_team = teams[i + 1]
                    
                    home_stats = date_games[date_games['TEAM_NAME'] == home_team].iloc[0]
                    away_stats = date_games[date_games['TEAM_NAME'] == away_team].iloc[0]
                    
                    game = {
                        'Date': date,
                        'HOME_TEAM': home_team,
                        'AWAY_TEAM': away_team,
                        'HOME_PTS': home_stats['PTS'],
                        'AWAY_PTS': away_stats['PTS'],
                        'HOME_FG_PCT': home_stats['FG_PCT'],
                        'AWAY_FG_PCT': away_stats['FG_PCT'],
                        'HOME_FG3_PCT': home_stats['FG3_PCT'],
                        'AWAY_FG3_PCT': away_stats['FG3_PCT'],
                        'HOME_FT_PCT': home_stats['FT_PCT'],
                        'AWAY_FT_PCT': away_stats['FT_PCT'],
                        'HOME_REB': home_stats['REB'],
                        'AWAY_REB': away_stats['REB'],
                        'HOME_AST': home_stats['AST'],
                        'AWAY_AST': away_stats['AST'],
                        'HOME_STL': home_stats['STL'],
                        'AWAY_STL': away_stats['STL'],
                        'HOME_BLK': home_stats['BLK'],
                        'AWAY_BLK': away_stats['BLK'],
                        'HOME_TOV': home_stats['TOV'],
                        'AWAY_TOV': away_stats['TOV']
                    }
                    games.append(game)
        
        return pd.DataFrame(games)

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
            
            query = f'SELECT *, "{target_date}" as Date FROM "{target_date}"'
            latest_stats = pd.read_sql_query(query, conn)
            latest_stats['Date'] = pd.to_datetime(latest_stats['Date'])
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