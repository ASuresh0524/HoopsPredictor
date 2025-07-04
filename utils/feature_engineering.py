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
        if windows is None:
            windows = self.rolling_windows
            
        df = df.sort_values('game_date')
        
        for window in windows:
            for col in columns:
                df[f'{col}_rolling_{window}'] = df.groupby(group_by)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
        return df
        
    def calculate_team_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team momentum indicators.
        
        Args:
            df (pd.DataFrame): Team game data
            
        Returns:
            pd.DataFrame: DataFrame with momentum features
        """
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        
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
        
        return df
        
    def calculate_head2head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate head-to-head statistics between teams.
        
        Args:
            df (pd.DataFrame): Team game data with both teams
            
        Returns:
            pd.DataFrame: DataFrame with head-to-head features
        """
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Create features for each matchup
        df['days_since_last_matchup'] = np.nan
        df['h2h_wins_home'] = 0
        df['h2h_wins_away'] = 0
        df['avg_point_diff'] = 0
        
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
            
            if len(prev_matchups) > 0:
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
            
        df = df.sort_values('game_date')
        
        # Calculate days since last game
        for entity in ['team_id', 'player_id']:
            if entity in df.columns:
                df[f'{entity}_rest_days'] = df.groupby(entity)['game_date'].diff().dt.days
                
        # Fill first game of season/career with median rest
        rest_cols = [col for col in df.columns if col.endswith('_rest_days')]
        for col in rest_cols:
            median_rest = df[col].median()
            df[col] = df[col].fillna(median_rest)
            
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
        Create features for game matchups by combining home and away team statistics.
        
        Args:
            home_team_stats (pd.DataFrame): Home team statistics
            away_team_stats (pd.DataFrame): Away team statistics
            
        Returns:
            pd.DataFrame: Combined matchup features
        """
        matchup_features = pd.DataFrame()
        
        # Basic differential features
        stats = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                'OFF_RATING', 'DEF_RATING', 'PACE', 'EFG_PCT', 'TOV_PCT', 'OREB_PCT', 'FT_RATE']
        
        for stat in stats:
            if stat in home_team_stats.columns and stat in away_team_stats.columns:
                matchup_features[f'{stat}_DIFF'] = home_team_stats[stat] - away_team_stats[stat]
        
        # Rolling statistics
        rolling_windows = [3, 5, 10]
        for window in rolling_windows:
            for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT']:
                col = f'{stat}_rolling_{window}'
                if col in home_team_stats.columns and col in away_team_stats.columns:
                    matchup_features[f'{col}_DIFF'] = home_team_stats[col] - away_team_stats[col]
        
        # Streak and rest features
        if 'STREAK' in home_team_stats.columns and 'STREAK' in away_team_stats.columns:
            matchup_features['STREAK_DIFF'] = home_team_stats['STREAK'] - away_team_stats['STREAK']
        
        if 'REST_DAYS' in home_team_stats.columns and 'REST_DAYS' in away_team_stats.columns:
            matchup_features['REST_DIFF'] = home_team_stats['REST_DAYS'] - away_team_stats['REST_DAYS']
        
        return matchup_features

    def prepare_game_features(
        self,
        games_df: pd.DataFrame,
        team_stats_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare features for game prediction by combining game data with team statistics.
        
        Args:
            games_df (pd.DataFrame): DataFrame containing game matchups
            team_stats_df (pd.DataFrame): DataFrame containing team statistics
            
        Returns:
            pd.DataFrame: Game features ready for prediction
        """
        features = []
        
        for _, game in games_df.iterrows():
            game_date = pd.to_datetime(game['Date'])
            
            # Get team stats before the game
            home_stats = team_stats_df[
                (team_stats_df['TEAM_NAME'] == game['HOME_TEAM']) &
                (team_stats_df['Date'] < game_date)
            ].sort_values('Date').iloc[-1]
            
            away_stats = team_stats_df[
                (team_stats_df['TEAM_NAME'] == game['AWAY_TEAM']) &
                (team_stats_df['Date'] < game_date)
            ].sort_values('Date').iloc[-1]
            
            # Create matchup features
            matchup_features = self.create_matchup_features(
                home_stats.to_frame().T,
                away_stats.to_frame().T
            )
            
            # Add game metadata
            matchup_features['Date'] = game_date
            matchup_features['HOME_TEAM'] = game['HOME_TEAM']
            matchup_features['AWAY_TEAM'] = game['AWAY_TEAM']
            
            if 'HOME_PTS' in game and 'AWAY_PTS' in game:
                matchup_features['HOME_WIN'] = int(game['HOME_PTS'] > game['AWAY_PTS'])
            
            features.append(matchup_features)
        
        return pd.concat(features, ignore_index=True)

    def prepare_features(
        self, 
        game_data: pd.DataFrame, 
        team_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare all features for model training.
        
        Args:
            game_data (pd.DataFrame): Game results data
            team_data (pd.DataFrame): Team statistics data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: X (features) and y (targets)
        """
        # Calculate all features
        team_data_processed = self.calculate_rolling_stats(team_data)
        team_data_processed = self.calculate_team_momentum(team_data_processed)
        
        game_data_processed = self.calculate_head2head_features(game_data)
        
        X = self.prepare_game_features(game_data_processed, team_data_processed)
        
        # Prepare target variables
        y = pd.DataFrame({
            'home_win': X['HOME_WIN'],
            'over_under': X['OU-Cover']
        })
        
        # Drop non-feature columns
        cols_to_drop = [
            'index', 'TEAM_ID', 'TEAM_NAME', 'TEAM_NAME.1', 'Date', 'Date.1',
            'Score', 'Home-Team-Win', 'OU', 'OU-Cover', 'Days-Rest-Home', 'Days-Rest-Away'
        ]
        X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X, y 