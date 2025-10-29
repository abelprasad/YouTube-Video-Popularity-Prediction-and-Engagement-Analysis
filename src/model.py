
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import logging
import json
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoEngagementModel:

    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        self.metrics = {}

        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        elif model_type == 'xgboost':
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Initialized {model_type} model")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Training data shape: {X_train.shape}")

        self.feature_names = list(X_train.columns)

        self.model.fit(X_train, y_train)

        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        logger.info("Training completed")

    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = 'test') -> Dict:
        logger.info(f"Evaluating on {dataset_name} data...")

        y_pred = self.model.predict(X)

        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        mape = np.mean(np.abs((np.expm1(y) - np.expm1(y_pred)) / (np.expm1(y) + 1))) * 100

        metrics = {
            'dataset': dataset_name,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }

        logger.info(f"{dataset_name.capitalize()} Metrics:")
        logger.info(f"  - RMSE: {rmse:.4f}")
        logger.info(f"  - MAE: {mae:.4f}")
        logger.info(f"  - R²: {r2:.4f}")
        logger.info(f"  - MAPE: {mape:.2f}%")

        return metrics

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        logger.info(f"Performing {cv}-fold cross-validation...")

        cv_scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='r2',
            n_jobs=-1
        )

        cv_metrics = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std())
        }

        logger.info(f"Cross-validation R² scores: {cv_scores}")
        logger.info(f"Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return cv_metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.feature_importance is None:
            logger.warning("Feature importance not available")
            return pd.DataFrame()

        return self.feature_importance.head(top_n)

    def save_model(self, filepath: str):
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class ModelTrainer:

    def __init__(self, data_source: str = 'scraped'):
        self.data_source = data_source
        self.models = {}
        self.results = {}

    def load_feature_data(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading feature data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        logger.info("Preparing data for training...")

        y = df['log_views']

        if self.data_source == 'scraped':
            feature_columns = [
                'duration_seconds', 'log_duration',
                'days_since_upload', 'log_days_since_upload',
                'title_length', 'title_uppercase_ratio', 'title_word_count',
                'has_clickbait_words', 'has_numbers', 'has_question', 'has_exclamation',
                'is_recent', 'is_short_video', 'is_long_video'
            ]

            encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
            feature_columns.extend(encoded_cols)

        else:
            feature_columns = [
                'duration_seconds', 'log_duration',
                'days_since_upload', 'log_days_since_upload',
                'title_length', 'title_uppercase_ratio', 'title_word_count',
                'has_clickbait_words', 'has_numbers', 'has_question', 'has_exclamation',
                'is_recent', 'is_short_video', 'is_long_video',
                'log_likes', 'log_comments', 'engagement_rate',
                'tag_count', 'description_length',
                'log_channel_subscribers', 'channel_video_count',
                'likes_per_view', 'comments_per_view'
            ]

            encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
            feature_columns.extend(encoded_cols)

        feature_columns = [col for col in feature_columns if col in df.columns]

        X = df[feature_columns]

        X = X.fillna(0)

        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Features used: {len(feature_columns)}")

        return X_train, X_test, y_train, y_test, feature_columns

    def train_models(self, X_train, X_test, y_train, y_test):
        model_types = ['random_forest', 'xgboost']

        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type.upper()} model for {self.data_source} data")
            logger.info(f"{'='*60}")

            model = VideoEngagementModel(model_type=model_type)

            model.train(X_train, y_train)

            train_metrics = model.evaluate(X_train, y_train, 'train')

            test_metrics = model.evaluate(X_test, y_test, 'test')

            cv_metrics = model.cross_validate(X_train, y_train, cv=5)

            feature_importance = model.get_feature_importance(top_n=20)

            self.models[model_type] = model
            self.results[model_type] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_metrics': cv_metrics,
                'feature_importance': feature_importance.to_dict('records')
            }

            logger.info(f"\n{model_type.upper()} Training Completed")

    def save_results(self, models_dir: str = 'models', results_dir: str = 'reports'):
        for model_type, model in self.models.items():
            model_path = f"{models_dir}/model_{self.data_source}_{model_type}.pkl"
            model.save_model(model_path)

            fi_path = f"{results_dir}/feature_importance_{self.data_source}_{model_type}.csv"
            model.get_feature_importance().to_csv(fi_path, index=False)

        results_path = f"{results_dir}/results_{self.data_source}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {results_path}")

    def print_summary(self):
        print("\n" + "="*60)
        print(f"MODEL TRAINING SUMMARY - {self.data_source.upper()} DATA")
        print("="*60)

        for model_type, results in self.results.items():
            print(f"\n{model_type.upper()}:")
            print(f"  Training R²: {results['train_metrics']['r2']:.4f}")
            print(f"  Test R²: {results['test_metrics']['r2']:.4f}")
            print(f"  Test RMSE: {results['test_metrics']['rmse']:.4f}")
            print(f"  Test MAE: {results['test_metrics']['mae']:.4f}")
            print(f"  Cross-Val R²: {results['cv_metrics']['cv_mean']:.4f} "
                  f"(+/- {results['cv_metrics']['cv_std']:.4f})")

            print(f"\n  Top 5 Features:")
            for i, feat in enumerate(results['feature_importance'][:5], 1):
                print(f"    {i}. {feat['feature']}: {feat['importance']:.4f}")

        print("\n" + "="*60)


def main():

    print("="*60)
    print("MODEL TRAINING")
    print("="*60)

    try:
        print("\n1. Training models for SCRAPED data...")
        trainer_scraped = ModelTrainer(data_source='scraped')
        df_scraped = trainer_scraped.load_feature_data('data/processed/scraped_features.csv')
        X_train, X_test, y_train, y_test, features = trainer_scraped.prepare_data(df_scraped)
        trainer_scraped.train_models(X_train, X_test, y_train, y_test)
        trainer_scraped.save_results()
        trainer_scraped.print_summary()

    except FileNotFoundError:
        print("\nScraped feature data not found. Please run feature engineering first.")
    except Exception as e:
        print(f"\nError training scraped models: {str(e)}")
        import traceback
        traceback.print_exc()

    try:
        print("\n2. Training models for API data...")
        trainer_api = ModelTrainer(data_source='api')
        df_api = trainer_api.load_feature_data('data/processed/api_features.csv')
        X_train, X_test, y_train, y_test, features = trainer_api.prepare_data(df_api)
        trainer_api.train_models(X_train, X_test, y_train, y_test)
        trainer_api.save_results()
        trainer_api.print_summary()

    except FileNotFoundError:
        print("\nAPI feature data not found. Please run API collector and feature engineering first.")
    except Exception as e:
        print(f"\nError training API models: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
