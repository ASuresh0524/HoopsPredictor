from pipelines.prediction_pipeline import PredictionPipeline
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize pipeline
        logger.info("Initializing prediction pipeline...")
        pipeline = PredictionPipeline()

        # Train models
        logger.info("Training models...")
        history = pipeline.train_models()
        logger.info("Training completed!")

        # Example: Predict game outcome
        home_team = "LAL"  # Los Angeles Lakers
        away_team = "GSW"  # Golden State Warriors
        logger.info(f"Predicting game outcome: {home_team} vs {away_team}")
        
        game_prediction = pipeline.predict_game(home_team, away_team)
        
        print("\nGame Prediction:")
        print(f"Home Team ({home_team}) Win Probability: {game_prediction['home_win_probability']:.2%}")
        print(f"Away Team ({away_team}) Win Probability: {game_prediction['away_win_probability']:.2%}")
        print(f"Team Strengths:")
        print(f"- {home_team}: {game_prediction['home_strength']:.3f}")
        print(f"- {away_team}: {game_prediction['away_strength']:.3f}")

        # Example: Predict player stats
        player_id = "2544"  # LeBron James
        logger.info(f"Predicting statistics for player {player_id}")
        
        player_predictions = pipeline.predict_player_stats(player_id, num_games=3)
        
        print("\nPlayer Statistics Predictions (Next 3 Games):")
        for i, pred in enumerate(player_predictions, 1):
            print(f"\nGame {i}:")
            for stat, value in pred.items():
                print(f"- {stat}: {value:.1f}")

        # Save models for future use
        logger.info("Saving models...")
        pipeline.save_models("models/saved")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 