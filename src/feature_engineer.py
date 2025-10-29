
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import joblib
from typing import Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []

    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading processed data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records")
        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['video_age_category'] = pd.cut(
            df['days_since_upload'],
            bins=[-1, 7, 30, 90, 365, np.inf],
            labels=['very_recent', 'recent', 'month_old', 'quarter_old', 'old']
        )

        df['is_recent'] = (df['days_since_upload'] <= 30).astype(int)

        df['log_days_since_upload'] = np.log1p(df['days_since_upload'])

        return df

    def create_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['duration_category'] = pd.cut(
            df['duration_minutes'],
            bins=[-1, 1, 5, 10, 20, np.inf],
            labels=['very_short', 'short', 'medium', 'long', 'very_long']
        )

        df['is_short_video'] = (df['duration_seconds'] < 60).astype(int)

        df['is_long_video'] = (df['duration_minutes'] > 20).astype(int)

        df['log_duration'] = np.log1p(df['duration_seconds'])

        return df

    def create_title_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['title_lower'] = df['title'].str.lower()

        clickbait_words = ['how to', 'top 10', 'best', 'worst', 'shocking', 'amazing', 'incredible']
        df['has_clickbait_words'] = df['title_lower'].apply(
            lambda x: any(word in str(x) for word in clickbait_words)
        ).astype(int)

        df['has_numbers'] = df['title'].str.contains(r'\d', regex=True).astype(int)

        df['has_question'] = df['title'].str.contains(r'\?', regex=False).astype(int)

        df['has_exclamation'] = df['title'].str.contains(r'!', regex=False).astype(int)

        df['title_length_category'] = pd.cut(
            df['title_length'],
            bins=[-1, 30, 50, 70, np.inf],
            labels=['short', 'medium', 'long', 'very_long']
        )

        df = df.drop('title_lower', axis=1)

        return df

    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'likes' in df.columns and 'comments' in df.columns:
            df['log_likes'] = np.log1p(df['likes'])
            df['log_comments'] = np.log1p(df['comments'])

            df['engagement_score'] = (
                df['likes_per_view'] * 0.6 +
                df['comments_per_view'] * 0.4
            )

            engagement_threshold = df['engagement_rate'].quantile(0.75)
            df['is_high_engagement'] = (df['engagement_rate'] > engagement_threshold).astype(int)

            df['comment_like_ratio'] = np.where(
                df['likes'] > 0,
                df['comments'] / df['likes'],
                0
            )

        return df

    def create_channel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'channel_subscribers' in df.columns:
            df['log_channel_subscribers'] = np.log1p(df['channel_subscribers'])

            df['channel_size'] = pd.cut(
                df['channel_subscribers'],
                bins=[-1, 1000, 10000, 100000, 1000000, np.inf],
                labels=['micro', 'small', 'medium', 'large', 'mega']
            )

            df['channel_avg_views_per_video'] = np.where(
                df['channel_video_count'] > 0,
                df['views'] / df['channel_video_count'],
                0
            )

        return df

    def create_view_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['log_views'] = np.log1p(df['views'])

        df['log_views_per_day'] = np.log1p(df['views_per_day'])

        df['view_velocity'] = np.where(
            df['days_since_upload'] > 0,
            df['views'] / np.sqrt(df['days_since_upload']),
            df['views']
        )

        view_threshold = df['views'].quantile(0.75)
        df['is_high_view'] = (df['views'] > view_threshold).astype(int)

        return df

    def engineer_features_scraped(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features for scraped data...")

        df = df.copy()

        df = self.create_time_features(df)
        df = self.create_duration_features(df)
        df = self.create_title_features(df)
        df = self.create_view_features(df)

        logger.info(f"Engineered features for {len(df)} records")
        logger.info(f"Total features: {len(df.columns)}")

        return df

    def engineer_features_api(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features for API data...")

        df = df.copy()

        df = self.create_time_features(df)
        df = self.create_duration_features(df)
        df = self.create_title_features(df)
        df = self.create_engagement_features(df)
        df = self.create_channel_features(df)
        df = self.create_view_features(df)

        logger.info(f"Engineered features for {len(df)} records")
        logger.info(f"Total features: {len(df.columns)}")

        return df

    def prepare_for_modeling_scraped(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Preparing scraped data for modeling...")

        df = df.copy()

        y = df['log_views']

        feature_columns = [
            'duration_seconds', 'log_duration',
            'days_since_upload', 'log_days_since_upload',
            'title_length', 'title_uppercase_ratio', 'title_word_count',
            'has_clickbait_words', 'has_numbers', 'has_question', 'has_exclamation',
            'is_recent', 'is_short_video', 'is_long_video'
        ]

        categorical_features = ['video_age_category', 'duration_category', 'title_length_category']

        for cat_col in categorical_features:
            if cat_col in df.columns:
                le = LabelEncoder()
                df[f'{cat_col}_encoded'] = le.fit_transform(df[cat_col].astype(str))
                self.label_encoders[cat_col] = le
                feature_columns.append(f'{cat_col}_encoded')

        X = df[feature_columns].copy()

        X = X.fillna(0)

        self.feature_names = feature_columns

        logger.info(f"Prepared data: X shape = {X.shape}, y shape = {y.shape}")

        return X, y

    def prepare_for_modeling_api(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Preparing API data for modeling...")

        df = df.copy()

        y = df['log_views']

        feature_columns = [
            'duration_seconds', 'log_duration',
            'days_since_upload', 'log_days_since_upload',
            'title_length', 'title_uppercase_ratio', 'title_word_count',
            'has_clickbait_words', 'has_numbers', 'has_question', 'has_exclamation',
            'is_recent', 'is_short_video', 'is_long_video',
            'log_likes', 'log_comments', 'engagement_rate', 'engagement_score',
            'tag_count', 'description_length',
            'log_channel_subscribers', 'channel_video_count',
            'likes_per_view', 'comments_per_view'
        ]

        categorical_features = [
            'video_age_category', 'duration_category', 'title_length_category',
            'channel_size', 'category_name'
        ]

        for cat_col in categorical_features:
            if cat_col in df.columns:
                le = LabelEncoder()
                df[f'{cat_col}_encoded'] = le.fit_transform(df[cat_col].astype(str))
                self.label_encoders[cat_col] = le
                feature_columns.append(f'{cat_col}_encoded')

        X = df[feature_columns].copy()

        X = X.fillna(0)

        self.feature_names = feature_columns

        logger.info(f"Prepared data: X shape = {X.shape}, y shape = {y.shape}")

        return X, y

    def save_feature_data(self, df: pd.DataFrame, filepath: str):
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Saved feature data to {filepath}")

    def save_artifacts(self, scaler_path: str, encoders_path: str):
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoders, encoders_path)
        logger.info(f"Saved artifacts: {scaler_path}, {encoders_path}")


def main():

    engineer = FeatureEngineer()

    print("="*60)
    print("Feature Engineering")
    print("="*60)

    try:
        print("\n1. Engineering Features for Scraped Data...")
        scraped_df = engineer.load_processed_data('data/processed/scraped_processed.csv')
        scraped_features = engineer.engineer_features_scraped(scraped_df)
        engineer.save_feature_data(scraped_features, 'data/processed/scraped_features.csv')

        print(f"\nScraped Data Features:")
        print(f"- Total records: {len(scraped_features)}")
        print(f"- Total features: {len(scraped_features.columns)}")

        print(f"\nSample of engineered scraped data:")
        print(scraped_features[['title', 'views', 'duration_minutes', 'log_views']].head(3))

    except FileNotFoundError:
        print("\nProcessed scraped data file not found. Please run preprocessor first.")
    except Exception as e:
        print(f"\nError engineering scraped features: {str(e)}")

    try:
        print("\n2. Engineering Features for API Data...")
        api_df = engineer.load_processed_data('data/processed/api_processed.csv')
        api_features = engineer.engineer_features_api(api_df)
        engineer.save_feature_data(api_features, 'data/processed/api_features.csv')

        print(f"\nAPI Data Features:")
        print(f"- Total records: {len(api_features)}")
        print(f"- Total features: {len(api_features.columns)}")

        print(f"\nSample of engineered API data:")
        print(api_features[['title', 'views', 'duration_minutes', 'engagement_rate', 'log_views']].head(3))

    except FileNotFoundError:
        print("\nProcessed API data file not found. Please run API collector and preprocessor first.")
    except Exception as e:
        print(f"\nError engineering API features: {str(e)}")

    print("\n" + "="*60)
    print("Feature engineering completed!")
    print("="*60)


if __name__ == '__main__':
    main()
