import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
import logging
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    log_loss,
    brier_score_loss,
    accuracy_score
)

from utils.data_loader import DataLoader
from utils.feature_engineering import FeatureEngineer
from models.time_series.player_stats_model import PlayerStatsPredictor
from models.bradley_terry.game_predictor import BradleyTerryPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionPipeline:
    """
    End-to-end pipeline for NBA game and player statistics prediction.
    
    This pipeline combines:
    1. Data loading and preprocessing
    2. Feature engineering
    3. Player statistics prediction
    4. Game outcome prediction
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline components.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.data_loader = DataLoader(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.player_predictor = PlayerStatsPredictor(config_path)
        self.game_predictor = BradleyTerryPredictor(config_path)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def prepare_player_features(self, player_data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Prepare feature and target columns for player prediction.
        
        Args:
            player_data (pd.DataFrame): Raw player data
            
        Returns:
            Tuple[List[str], List[str]]: Feature and target column names
        """
        # Basic stats to predict
        target_cols = ['points', 'rebounds', 'assists', 'steals', 'blocks']
        
        # Features for prediction
        feature_cols = target_cols.copy()  # Use past performance
        
        # Add engineered features
        rolling_stats = [col for col in player_data.columns if 'rolling' in col]
        rest_days = [col for col in player_data.columns if 'rest_days' in col]
        
        feature_cols.extend(rolling_stats)
        feature_cols.extend(rest_days)
        
        # Add game context features if available
        context_features = ['is_home', 'games_played', 'season']
        feature_cols.extend([col for col in context_features if col in player_data.columns])
        
        return feature_cols, target_cols
        
    def prepare_game_features(self, game_data: pd.DataFrame) -> List[str]:
        """
        Prepare features for game outcome prediction.
        
        Args:
            game_data (pd.DataFrame): Processed game data
            
        Returns:
            List[str]: Feature column names
        """
        feature_cols = []
        
        # Team performance features
        team_features = [
            'recent_win_pct', 'point_diff_trend', 'streak',
            'home_win_pct', 'away_win_pct'
        ]
        feature_cols.extend([col for col in team_features if col in game_data.columns])
        
        # Head-to-head features
        h2h_features = ['h2h_games', 'h2h_home_wins']
        feature_cols.extend([col for col in h2h_features if col in game_data.columns])
        
        # Rest days if available
        rest_features = [col for col in game_data.columns if 'rest_days' in col]
        feature_cols.extend(rest_features)
        
        return feature_cols
        
    def train(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Train both player stats and game outcome models.
        
        Args:
            start_date (str, optional): Training start date
            end_date (str, optional): Training end date
        """
        # Load and preprocess data
        logger.info("Loading data...")
        player_query = "SELECT * FROM player_stats"  # Customize based on your schema
        game_query = "SELECT * FROM game_logs"  # Customize based on your schema
        
        player_data = self.data_loader.load_player_data(player_query)
        game_data = self.data_loader.load_team_data(game_query)
        
        # Apply date filters if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            player_data = player_data[player_data['game_date'] >= start_date]
            game_data = game_data[game_data['game_date'] >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            player_data = player_data[player_data['game_date'] <= end_date]
            game_data = game_data[game_data['game_date'] <= end_date]
            
        # Preprocess data
        logger.info("Preprocessing data...")
        player_data = self.data_loader.preprocess_player_data(player_data)
        game_data = self.data_loader.preprocess_team_data(game_data)
        
        # Feature engineering
        logger.info("Engineering features...")
        player_data = self.feature_engineer.calculate_rolling_averages(
            player_data,
            ['points', 'rebounds', 'assists', 'steals', 'blocks'],
            'player_id'
        )
        player_data = self.feature_engineer.add_rest_days_feature(player_data)
        
        game_data = self.feature_engineer.calculate_team_momentum(game_data)
        game_data = self.feature_engineer.calculate_head2head_features(game_data)
        game_data = self.feature_engineer.add_home_court_features(game_data)
        
        # Prepare features
        player_features, player_targets = self.prepare_player_features(player_data)
        game_features = self.prepare_game_features(game_data)
        
        # Split data
        player_splits = self.data_loader.get_train_val_test_split(player_data)
        game_splits = self.data_loader.get_train_val_test_split(game_data)
        
        # Train player stats model
        logger.info("Training player stats model...")
        train_features, train_targets = self.player_predictor.prepare_sequences(
            player_splits['train'],
            player_features,
            player_targets
        )
        
        val_features, val_targets = self.player_predictor.prepare_sequences(
            player_splits['val'],
            player_features,
            player_targets
        )
        
        train_loader = self.player_predictor.create_data_loaders(train_features, train_targets)
        val_loader = self.player_predictor.create_data_loaders(val_features, val_targets)
        
        self.player_predictor.train(train_loader, val_loader)
        
        # Train game outcome model
        logger.info("Training game outcome model...")
        self.game_predictor.fit(
            game_splits['train'],
            'home_team_id',
            'away_team_id',
            'home_team_won',
            game_features
        )
        
        logger.info("Training completed successfully")
        
    def predict_game(self, 
                    home_team: str,
                    away_team: str,
                    game_date: str,
                    player_stats: Optional[pd.DataFrame] = None) -> Dict:
        """
        Make predictions for a specific game.
        
        Args:
            home_team (str): Home team identifier
            away_team (str): Away team identifier
            game_date (str): Game date
            player_stats (pd.DataFrame, optional): Recent player statistics
            
        Returns:
            Dict: Predictions including game outcome and player statistics
        """
        predictions = {
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team
        }
        
        # Predict game outcome
        game_features = None  # Prepare game features based on recent data
        win_prob = self.game_predictor.predict_proba(home_team, away_team, game_features)
        predictions['home_win_probability'] = float(win_prob)
        
        # Predict player stats if data provided
        if player_stats is not None:
            player_features, _ = self.prepare_player_features(player_stats)
            features = self.player_predictor._prepare_features(player_stats[player_features])
            player_predictions = self.player_predictor.predict(features)
            
            predictions['player_stats'] = {
                player_id: stats for player_id, stats in 
                zip(player_stats['player_id'].unique(), player_predictions)
            }
            
        return predictions
        
    def evaluate(self, 
                 player_data: pd.DataFrame,
                 game_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            player_data (pd.DataFrame): Player test data
            game_data (pd.DataFrame): Game test data
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        metrics = {}
        
        # Evaluate player stats predictions
        player_features, player_targets = self.prepare_player_features(player_data)
        features = self.player_predictor._prepare_features(player_data[player_features])
        predictions = self.player_predictor.predict(features)
        
        for i, stat in enumerate(player_targets):
            metrics[f'{stat}_mae'] = mean_absolute_error(
                player_data[stat],
                predictions[:, i]
            )
            metrics[f'{stat}_rmse'] = np.sqrt(mean_squared_error(
                player_data[stat],
                predictions[:, i]
            ))
            
        # Evaluate game outcome predictions
        game_features = self.prepare_game_features(game_data)
        game_probs = [
            self.game_predictor.predict_proba(
                row['home_team_id'],
                row['away_team_id'],
                game_features[i:i+1] if game_features else None
            )
            for i, row in game_data.iterrows()
        ]
        
        metrics['game_log_loss'] = log_loss(
            game_data['home_team_won'],
            game_probs
        )
        metrics['game_brier_score'] = brier_score_loss(
            game_data['home_team_won'],
            game_probs
        )
        metrics['game_accuracy'] = accuracy_score(
            game_data['home_team_won'],
            [p > 0.5 for p in game_probs]
        )
        
        return metrics
        
    def save_models(self, player_model_path: str, game_model_path: str):
        """
        Save both models to disk.
        
        Args:
            player_model_path (str): Path to save player stats model
            game_model_path (str): Path to save game outcome model
        """
        self.player_predictor.save_model(player_model_path)
        self.game_predictor.save_model(game_model_path)
        
    def load_models(self, player_model_path: str, game_model_path: str):
        """
        Load both models from disk.
        
        Args:
            player_model_path (str): Path to player stats model
            game_model_path (str): Path to game outcome model
        """
        self.player_predictor.load_model(player_model_path)
        self.game_predictor.load_model(game_model_path) 