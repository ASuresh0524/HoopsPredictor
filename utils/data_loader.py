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
                    
                    # Calculate point differential and win/loss
                    point_differential = home_stats['PTS'] - away_stats['PTS']
                    home_win = int(point_differential > 0)
                    
                    game = {
                        'game_id': f"{date}_{home_team}_{away_team}",
                        'game_date': date,
                        'home_team_id': home_stats['TEAM_ID'],
                        'away_team_id': away_stats['TEAM_ID'],
                        'team_name': home_team,
                        'team_name.1': away_team,
                        'pts': home_stats['PTS'],
                        'pts.1': away_stats['PTS'],
                        'fg_pct': home_stats['FG_PCT'],
                        'fg_pct.1': away_stats['FG_PCT'],
                        'fg3_pct': home_stats['FG3_PCT'],
                        'fg3_pct.1': away_stats['FG3_PCT'],
                        'ft_pct': home_stats['FT_PCT'],
                        'ft_pct.1': away_stats['FT_PCT'],
                        'reb': home_stats['REB'],
                        'reb.1': away_stats['REB'],
                        'ast': home_stats['AST'],
                        'ast.1': away_stats['AST'],
                        'stl': home_stats['STL'],
                        'stl.1': away_stats['STL'],
                        'blk': home_stats['BLK'],
                        'blk.1': away_stats['BLK'],
                        'tov': home_stats['TOV'],
                        'tov.1': away_stats['TOV'],
                        'point_differential': point_differential,
                        'won': home_win,
                        'is_home': 1
                    }
                    games.append(game)
                    
                    # Add reverse matchup for away team perspective
                    away_game = game.copy()
                    away_game.update({
                        'game_id': f"{date}_{away_team}_{home_team}",
                        'home_team_id': away_stats['TEAM_ID'],
                        'away_team_id': home_stats['TEAM_ID'],
                        'team_name': away_team,
                        'team_name.1': home_team,
                        'pts': away_stats['PTS'],
                        'pts.1': home_stats['PTS'],
                        'fg_pct': away_stats['FG_PCT'],
                        'fg_pct.1': home_stats['FG_PCT'],
                        'fg3_pct': away_stats['FG3_PCT'],
                        'fg3_pct.1': home_stats['FG3_PCT'],
                        'ft_pct': away_stats['FT_PCT'],
                        'ft_pct.1': home_stats['FT_PCT'],
                        'reb': away_stats['REB'],
                        'reb.1': home_stats['REB'],
                        'ast': away_stats['AST'],
                        'ast.1': home_stats['AST'],
                        'stl': away_stats['STL'],
                        'stl.1': home_stats['STL'],
                        'blk': away_stats['BLK'],
                        'blk.1': home_stats['BLK'],
                        'tov': away_stats['TOV'],
                        'tov.1': home_stats['TOV'],
                        'point_differential': -point_differential,
                        'won': 1 - home_win,
                        'is_home': 0
                    })
                    games.append(away_game)
        
        if not games:
            raise ValueError(f"No games found between {start_date} and {end_date}")
            
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
        Preprocess team data by cleaning and standardizing column names.
        
        Args:
            df (pd.DataFrame): Raw team data
            
        Returns:
            pd.DataFrame: Preprocessed team data
        """
        df = df.copy()
        
        # Convert date to datetime and rename to game_date
        df['game_date'] = pd.to_datetime(df['Date'])
        df = df.drop('Date', axis=1)
        
        # Rename columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Rename team ID column
        if 'team_id' not in df.columns and 'team_name' in df.columns:
            # Create a mapping of team names to IDs if needed
            team_names = df['team_name'].unique()
            team_id_map = {name: str(i+1).zfill(3) for i, name in enumerate(sorted(team_names))}
            df['team_id'] = df['team_name'].map(team_id_map)
        
        # Ensure required columns exist
        required_cols = [
            'team_id', 'team_name', 'pts', 'fg_pct', 'fg3_pct', 'ft_pct',
            'reb', 'ast', 'stl', 'blk', 'tov', 'game_date'
        ]
        
        # Create mapping for any missing columns
        col_mapping = {
            'pts': 'pts',
            'fg_pct': 'fg_pct',
            'fg3_pct': 'fg3_pct',
            'ft_pct': 'ft_pct',
            'reb': 'reb',
            'ast': 'ast',
            'stl': 'stl',
            'blk': 'blk',
            'tov': 'tov',
            'team_id': 'team_id',
            'team_name': 'team_name'
        }
        
        # Apply column mapping
        df = df.rename(columns=col_mapping)
        
        # Check for missing required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Calculate additional metrics
        if 'off_rating' not in df.columns:
            df['off_rating'] = df['pts'] / (df['fg_pct'] + df['fg3_pct'] + df['ft_pct'])
            
        if 'def_rating' not in df.columns:
            df['def_rating'] = df['pts'] / (df['reb'] + df['stl'] + df['blk'])
            
        # Sort by date and team
        df = df.sort_values(['game_date', 'team_name'])
        
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