import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import yaml
from datetime import datetime, timedelta
import joblib
from pathlib import Path

from utils.data_loader import DataLoader
from utils.feature_engineering import FeatureEngineer
from models.bradley_terry.game_predictor import BradleyTerryPredictor
from models.time_series.player_stats_model import PlayerStatsPredictor

class PredictionPipeline:
    def __init__(self, config_path="config.yaml"):
        """Initialize the prediction pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = DataLoader(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.game_predictor = BradleyTerryPredictor(config_path)
        self.player_predictor = PlayerStatsPredictor(config_path)
        
        self.setup_logging()
        self.setup_model_paths()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)

    def setup_model_paths(self):
        """Set up paths for model saving and loading."""
        self.model_dir = Path(self.config['models']['save_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.game_model_path = self.model_dir / 'game_predictor.joblib'
        self.player_model_path = self.model_dir / 'player_predictor.pt'

    def train_models(self, start_date=None, end_date=None) -> Dict:
        """
        Train both game and player prediction models.
        
        Args:
            start_date (str, optional): Start date for training data
            end_date (str, optional): End date for training data
            
        Returns:
            Dict: Training history and metrics
        """
        self.logger.info("Starting model training...")
        
        # Use default dates from config if not provided
        if start_date is None:
            start_date = self.config['data']['train_start_date']
        if end_date is None:
            end_date = self.config['data']['train_end_date']
        
        # Load and prepare data
        team_data = self.data_loader.load_team_data(start_date, end_date)
        player_data = self.data_loader.load_player_data(start_date, end_date)
        game_data = self.data_loader.load_game_data(start_date, end_date)
        
        # Process team data
        team_data_processed = self.feature_engineer.calculate_rolling_stats(team_data)
        
        # Train game predictor
        self.logger.info("Training game predictor...")
        game_features = self.feature_engineer.prepare_game_features(game_data, team_data_processed)
        
        game_history = self.game_predictor.train(
            game_features,
            save_path=self.game_model_path
        )
        
        # Train player predictor
        self.logger.info("Training player predictor...")
        player_history = self.player_predictor.train(
            player_data,
            save_path=self.player_model_path
        )
        
        return {
            'game_history': game_history,
            'player_history': player_history
        }

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        date: Optional[str] = None
    ) -> Dict:
        """
        Predict the outcome of a game between two teams.
        
        Args:
            home_team (str): Home team name/code
            away_team (str): Away team name/code
            date (str, optional): Game date in YYYY-MM-DD format
            
        Returns:
            Dict: Prediction results including win probability and expected stats
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Load recent team data
        start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
        team_data = self.data_loader.load_team_data(start_date, date)
        
        # Process team data
        team_data_processed = self.feature_engineer.calculate_rolling_stats(team_data)
        
        # Create game features
        game_df = pd.DataFrame([{
            'game_date': date,
            'game_id': f"{date}_{home_team}_{away_team}",
            'home_team_id': home_team,  # Using team name as ID for now
            'away_team_id': away_team,
            'team_name': home_team,
            'team_name.1': away_team,
            'is_home': 1
        }])
        
        game_features = self.feature_engineer.prepare_game_features(
            game_df,
            team_data_processed
        )
        
        # Make predictions
        predictions = self.game_predictor.predict(game_features)
        
        # Get team stats
        home_stats = team_data_processed[
            team_data_processed['team_name'] == home_team
        ].iloc[-1]
        
        away_stats = team_data_processed[
            team_data_processed['team_name'] == away_team
        ].iloc[-1]
        
        return {
            'home_win_probability': float(predictions['win_probability']),
            'predicted_spread': float(predictions['spread']),
            'predicted_total': float(predictions['total']),
            'home_team_stats': {
                'recent_pts': float(home_stats['pts_rolling_5']),
                'off_rating': float(home_stats['off_rating']),
                'def_rating': float(home_stats['def_rating']),
                'pace': float(home_stats.get('pace', 0))
            },
            'away_team_stats': {
                'recent_pts': float(away_stats['pts_rolling_5']),
                'off_rating': float(away_stats['off_rating']),
                'def_rating': float(away_stats['def_rating']),
                'pace': float(away_stats.get('pace', 0))
            }
        }

    def predict_player_stats(
        self,
        player_name: str,
        team: str,
        opponent: str,
        date: Optional[str] = None
    ) -> Dict:
        """
        Predict statistics for a player in an upcoming game.
        
        Args:
            player_name (str): Player's name
            team (str): Player's team name/code
            opponent (str): Opponent team name/code
            date (str, optional): Game date in YYYY-MM-DD format
            
        Returns:
            Dict: Predicted statistics for the player
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Load recent player data
        start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')
        player_data = self.data_loader.load_player_data(start_date, date)
        
        # Filter for the specific player
        player_history = player_data[
            (player_data['player_name'] == player_name) &
            (player_data['team_name'] == team)
        ]
        
        if len(player_history) == 0:
            raise ValueError(f"No data found for player {player_name} on team {team}")
        
        # Make predictions
        predictions = self.player_predictor.predict(
            player_history,
            opponent,
            date
        )
        
        return {
            'points': float(predictions['pts']),
            'rebounds': float(predictions['reb']),
            'assists': float(predictions['ast']),
            'steals': float(predictions['stl']),
            'blocks': float(predictions['blk']),
            'minutes': float(predictions['min']),
            'fantasy_points': float(predictions['fantasy_pts'])
        }

    def evaluate_models(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Evaluate both game and player prediction models on a test period.
        
        Args:
            start_date (str, optional): Start date for evaluation in YYYY-MM-DD format
            end_date (str, optional): End date for evaluation in YYYY-MM-DD format
            
        Returns:
            Dict: Evaluation metrics for both models
        """
        # Use default test dates from config if not provided
        if start_date is None:
            start_date = self.config['data']['test_start_date']
        if end_date is None:
            end_date = self.config['data']['test_end_date']
        
        # Load test data
        team_data = self.data_loader.load_team_data(start_date, end_date)
        player_data = self.data_loader.load_player_data(start_date, end_date)
        game_data = self.data_loader.load_game_data(start_date, end_date)
        
        # Process team data
        team_data_processed = self.feature_engineer.calculate_rolling_stats(team_data)
        
        # Evaluate game predictor
        game_features = self.feature_engineer.prepare_game_features(
            game_data,
            team_data_processed
        )
        
        game_metrics = self.game_predictor.evaluate(game_features)
        
        # Evaluate player predictor
        player_metrics = self.player_predictor.evaluate(player_data)
        
        return {
            'game_metrics': game_metrics,
            'player_metrics': player_metrics
        } 