import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import yaml
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    A class to handle feature engineering for both team and player data.
    
    Attributes:
        config (dict): Configuration dictionary loaded from config.yaml
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the FeatureEngineer with configuration settings.
        
        Args:
            config_path (str): Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.rolling_windows = self.config['feature_engineering']['rolling_windows']
        self.momentum_windows = self.config['feature_engineering']['momentum_windows']
        self.head2head_lookback = self.config['feature_engineering']['head2head_lookback_days']
        
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)

    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str], context: str) -> None:
        """
        Validate that a DataFrame is not empty and has required columns.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            required_columns (List[str]): List of required column names
            context (str): Context for error messages
            
        Raises:
            ValueError: If DataFrame is empty or missing required columns
        """
        if df.empty:
            raise ValueError(f"Empty DataFrame provided for {context}")
            
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for {context}: {missing_cols}")

    def calculate_rolling_averages(self, df: pd.DataFrame, 
                                 columns: List[str],
                                 group_by: str,
                                 windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate rolling averages for specified columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to calculate rolling averages for
            group_by (str): Column to group by (e.g., 'team_id' or 'player_id')
            windows (List[int], optional): List of window sizes
            
        Returns:
            pd.DataFrame: DataFrame with added rolling average columns
        """
        # Validate input DataFrame
        required_cols = [group_by, 'game_date'] + columns
        self.validate_dataframe(df, required_cols, "rolling averages calculation")
        
        if windows is None:
            windows = self.rolling_windows
            
        df = df.sort_values('game_date')
        result_df = df.copy()
        
        try:
            for window in windows:
                for col in columns:
                    result_df[f'{col}_rolling_{window}'] = df.groupby(group_by)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
        except Exception as e:
            self.logger.error(f"Error calculating rolling averages: {str(e)}")
            raise
                
        return result_df
        
    def calculate_team_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team momentum indicators.
        
        Args:
            df (pd.DataFrame): Team game data
            
        Returns:
            pd.DataFrame: DataFrame with momentum features
        """
        required_cols = ['team_id', 'won', 'point_differential', 'pts', 'game_date']
        self.validate_dataframe(df, required_cols, "team momentum calculation")
        
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        try:
            for window in self.momentum_windows:
                # Win streak (positive) or losing streak (negative)
                df[f'win_streak_{window}'] = df.groupby('team_id')['won'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).sum()
                )
                
                # Point differential momentum
                df[f'point_diff_momentum_{window}'] = df.groupby('team_id')['point_differential'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Offensive rating trend
                df[f'offensive_momentum_{window}'] = df.groupby('team_id')['pts'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Defensive rating trend
                df[f'defensive_momentum_{window}'] = df.groupby('team_id')['pts'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                ).shift(1)  # Shift to avoid data leakage
        except Exception as e:
            self.logger.error(f"Error calculating team momentum: {str(e)}")
            raise
            
        return df
        
    def calculate_head2head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate head-to-head statistics between teams.
        
        Args:
            df (pd.DataFrame): Team game data with both teams
            
        Returns:
            pd.DataFrame: DataFrame with head-to-head features
        """
        required_cols = ['team_id', 'team_id.1', 'game_date', 'won', 'point_differential']
        self.validate_dataframe(df, required_cols, "head-to-head features calculation")
        
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Initialize features
        df['days_since_last_matchup'] = np.nan
        df['h2h_wins_home'] = 0
        df['h2h_wins_away'] = 0
        df['avg_point_diff'] = 0
        
        try:
            for idx, row in df.iterrows():
                date = row['game_date']
                home_team = row['team_id']
                away_team = row['team_id.1']
                
                # Get previous matchups within lookback period
                lookback_date = date - timedelta(days=self.head2head_lookback)
                prev_matchups = df[
                    (df['game_date'] < date) & 
                    (df['game_date'] >= lookback_date) &
                    (
                        ((df['team_id'] == home_team) & (df['team_id.1'] == away_team)) |
                        ((df['team_id'] == away_team) & (df['team_id.1'] == home_team))
                    )
                ]
                
                if not prev_matchups.empty:
                    # Days since last matchup
                    df.loc[idx, 'days_since_last_matchup'] = (
                        date - prev_matchups['game_date'].max()
                    ).days
                    
                    # Head-to-head wins
                    home_wins = len(prev_matchups[
                        (prev_matchups['team_id'] == home_team) & 
                        (prev_matchups['won'] == 1)
                    ])
                    away_wins = len(prev_matchups) - home_wins
                    
                    df.loc[idx, 'h2h_wins_home'] = home_wins
                    df.loc[idx, 'h2h_wins_away'] = away_wins
                    
                    # Average point differential
                    df.loc[idx, 'avg_point_diff'] = prev_matchups['point_differential'].mean()
        except Exception as e:
            self.logger.error(f"Error calculating head-to-head features: {str(e)}")
            raise
        
        return df
        
    def add_rest_days_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rest days between games for teams/players.
        
        Args:
            df (pd.DataFrame): Game data
            
        Returns:
            pd.DataFrame: DataFrame with rest days feature
        """
        if not self.config['feature_engineering']['rest_days_impact']:
            return df
            
        required_cols = ['game_date']
        self.validate_dataframe(df, required_cols, "rest days calculation")
        
        df = df.copy()
        df = df.sort_values('game_date')
        
        try:
            # Calculate days since last game
            for entity in ['team_id', 'player_id']:
                if entity in df.columns:
                    df[f'{entity}_rest_days'] = df.groupby(entity)['game_date'].diff().dt.days
                    
            # Fill first game of season/career with median rest
            rest_cols = [col for col in df.columns if col.endswith('_rest_days')]
            for col in rest_cols:
                median_rest = df[col].median()
                df[col] = df[col].fillna(median_rest)
        except Exception as e:
            self.logger.error(f"Error calculating rest days: {str(e)}")
            raise
            
        return df
        
    def add_home_court_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add home court advantage related features.
        
        Args:
            df (pd.DataFrame): Game data
            
        Returns:
            pd.DataFrame: DataFrame with home court features
        """
        if not self.config['feature_engineering']['home_court_advantage']:
            return df
            
        required_cols = ['team_id', 'is_home', 'won']
        self.validate_dataframe(df, required_cols, "home court features calculation")
        
        df = df.copy()
        
        try:
            # Calculate home/away win percentages
            df['home_win_pct'] = df.groupby('team_id').apply(
                lambda x: x[x['is_home']]['won'].expanding().mean()
            ).reset_index(level=0, drop=True)
            
            df['away_win_pct'] = df.groupby('team_id').apply(
                lambda x: x[~x['is_home']]['won'].expanding().mean()
            ).reset_index(level=0, drop=True)
            
            # Calculate point differential at home/away
            if 'point_differential' in df.columns:
                df['home_point_diff'] = df.groupby('team_id').apply(
                    lambda x: x[x['is_home']]['point_differential'].expanding().mean()
                ).reset_index(level=0, drop=True)
                
                df['away_point_diff'] = df.groupby('team_id').apply(
                    lambda x: x[~x['is_home']]['point_differential'].expanding().mean()
                ).reset_index(level=0, drop=True)
        except Exception as e:
            self.logger.error(f"Error calculating home court features: {str(e)}")
            raise
            
        return df

    def calculate_rolling_stats(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics for team performance metrics.
        
        Args:
            df (pd.DataFrame): Team statistics dataframe
            windows (List[int], optional): List of window sizes for rolling calculations
            
        Returns:
            pd.DataFrame: DataFrame with additional rolling statistics columns
        """
        if windows is None:
            windows = [3, 5, 10]
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by team and date
        df = df.sort_values(['TEAM_NAME', 'Date'])
        
        # Basic stats to calculate rolling averages for
        stats = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        
        for window in windows:
            for stat in stats:
                # Calculate rolling mean
                df[f'{stat}_rolling_{window}'] = df.groupby('TEAM_NAME')[stat].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                # Calculate rolling standard deviation
                df[f'{stat}_rolling_std_{window}'] = df.groupby('TEAM_NAME')[stat].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
        
        # Calculate win/loss streak
        df['WIN'] = (df['W'] > df['W'].shift(1)).astype(int)
        df['STREAK'] = df.groupby('TEAM_NAME')['WIN'].transform(
            lambda x: x.groupby((x != x.shift(1)).cumsum()).cumsum()
        )
        
        # Calculate offensive and defensive ratings
        df['POSS'] = df['FGA'] - df['OREB'] + df['TOV'] + (0.4 * df['FTA'])
        df['OFF_RATING'] = (df['PTS'] / df['POSS']) * 100
        df['DEF_RATING'] = (df['PTS'].shift(1) / df['POSS'].shift(1)) * 100
        
        # Calculate four factors
        df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
        df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
        df['OREB_PCT'] = df['OREB'] / (df['OREB'] + df['DREB'].shift(1))
        df['FT_RATE'] = df['FTA'] / df['FGA']
        
        # Calculate pace
        df['PACE'] = 48 * ((df['POSS'] + df['POSS'].shift(1)) / (2 * (df['MIN'] / 5)))
        
        # Calculate rest days
        df['REST_DAYS'] = df.groupby('TEAM_NAME')['Date'].diff().dt.days.fillna(0)
        
        return df

    def create_matchup_features(
        self,
        home_team_stats: pd.DataFrame,
        away_team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create features for team matchups by comparing team statistics.
        
        Args:
            home_team_stats (pd.DataFrame): Home team statistics
            away_team_stats (pd.DataFrame): Away team statistics
            
        Returns:
            pd.DataFrame: DataFrame with matchup features
        """
        # Validate input DataFrames
        self.validate_dataframe(home_team_stats, ['team_id'], "home team stats")
        self.validate_dataframe(away_team_stats, ['team_id'], "away team stats")
        
        try:
            # Ensure both DataFrames have the same columns
            common_cols = list(set(home_team_stats.columns) & set(away_team_stats.columns))
            if not common_cols:
                raise ValueError("No common columns found between home and away team stats")
                
            # Create matchup DataFrame
            matchup_features = pd.DataFrame()
            
            # Calculate differentials for all numeric columns
            numeric_cols = home_team_stats.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in common_cols and col != 'team_id':
                    matchup_features[f'{col}_diff'] = (
                        home_team_stats[col].values - away_team_stats[col].values
                    )
                    matchup_features[f'{col}_ratio'] = (
                        home_team_stats[col].values / away_team_stats[col].values
                    )
            
            # Add team IDs
            matchup_features['home_team_id'] = home_team_stats['team_id']
            matchup_features['away_team_id'] = away_team_stats['team_id']
            
            return matchup_features
            
        except Exception as e:
            self.logger.error(f"Error creating matchup features: {str(e)}")
            raise

    def prepare_game_features(
        self,
        games_df: pd.DataFrame,
        team_stats_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare features for game prediction by combining game data with team statistics.
        
        Args:
            games_df (pd.DataFrame): Game data
            team_stats_df (pd.DataFrame): Team statistics
            
        Returns:
            pd.DataFrame: DataFrame with prepared features
        """
        # Validate input DataFrames
        required_game_cols = ['game_id', 'home_team_id', 'away_team_id', 'game_date']
        required_stats_cols = ['team_id', 'game_date']
        self.validate_dataframe(games_df, required_game_cols, "games data")
        self.validate_dataframe(team_stats_df, required_stats_cols, "team stats")
        
        try:
            games_df = games_df.copy()
            team_stats_df = team_stats_df.copy()
            
            # Ensure datetime format
            games_df['game_date'] = pd.to_datetime(games_df['game_date'])
            team_stats_df['game_date'] = pd.to_datetime(team_stats_df['game_date'])
            
            # Get the latest stats before each game
            features_list = []
            
            for _, game in games_df.iterrows():
                game_date = game['game_date']
                home_team = game['home_team_id']
                away_team = game['away_team_id']
                
                # Get latest stats before the game
                home_stats = team_stats_df[
                    (team_stats_df['team_id'] == home_team) &
                    (team_stats_df['game_date'] < game_date)
                ].sort_values('game_date').iloc[-1:] if not team_stats_df.empty else pd.DataFrame()
                
                away_stats = team_stats_df[
                    (team_stats_df['team_id'] == away_team) &
                    (team_stats_df['game_date'] < game_date)
                ].sort_values('game_date').iloc[-1:] if not team_stats_df.empty else pd.DataFrame()
                
                if not (home_stats.empty or away_stats.empty):
                    # Create matchup features
                    matchup_features = self.create_matchup_features(home_stats, away_stats)
                    matchup_features['game_id'] = game['game_id']
                    features_list.append(matchup_features)
            
            if not features_list:
                raise ValueError("No valid matchups found for feature preparation")
                
            # Combine all features
            final_features = pd.concat(features_list, ignore_index=True)
            
            # Merge with original game data
            result = games_df.merge(final_features, on='game_id', how='inner')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error preparing game features: {str(e)}")
            raise

    def prepare_features(
        self, 
        game_data: pd.DataFrame, 
        team_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main method to prepare all features for model training and prediction.
        
        Args:
            game_data (pd.DataFrame): Game data
            team_data (pd.DataFrame): Team statistics data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed game features and team features
        """
        try:
            # Validate input data
            self.validate_dataframe(game_data, ['game_id', 'game_date'], "game data")
            self.validate_dataframe(team_data, ['team_id', 'game_date'], "team data")
            
            # Process team data
            team_features = team_data.copy()
            team_features = self.calculate_rolling_stats(team_features)
            team_features = self.calculate_team_momentum(team_features)
            team_features = self.add_rest_days_feature(team_features)
            team_features = self.add_home_court_features(team_features)
            
            # Process game data and create matchup features
            game_features = self.prepare_game_features(game_data, team_features)
            game_features = self.calculate_head2head_features(game_features)
            
            return game_features, team_features
            
        except Exception as e:
            self.logger.error(f"Error in feature preparation pipeline: {str(e)}")
            raise 