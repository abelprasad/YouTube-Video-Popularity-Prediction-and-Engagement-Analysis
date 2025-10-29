
import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:

    def __init__(self):
        self.category_mapping = {
            '1': 'Film & Animation',
            '2': 'Autos & Vehicles',
            '10': 'Music',
            '15': 'Pets & Animals',
            '17': 'Sports',
            '19': 'Travel & Events',
            '20': 'Gaming',
            '22': 'People & Blogs',
            '23': 'Comedy',
            '24': 'Entertainment',
            '25': 'News & Politics',
            '26': 'Howto & Style',
            '27': 'Education',
            '28': 'Science & Technology',
            '29': 'Nonprofits & Activism'
        }

    def load_scraped_data(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading scraped data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records")
        return df

    def load_api_data(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading API data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records")
        return df

    def clean_scraped_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning scraped data...")

        df = df.copy()

        initial_count = len(df)
        df = df.drop_duplicates(subset=['video_id'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} duplicates")

        df = df.dropna(subset=['video_id', 'title'])
        logger.info(f"After removing null critical fields: {len(df)} records")

        df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0).astype(int)

        df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce').fillna(0).astype(int)

        df['upload_date_parsed'] = df['upload_date'].apply(self._parse_relative_date)

        df['days_since_upload'] = df['upload_date_parsed'].apply(
            lambda x: (pd.Timestamp.now(tz='UTC').replace(tzinfo=None) - x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else pd.Timestamp.now() - x).days if pd.notna(x) else np.nan
        )

        df['channel_name'] = df['channel_name'].fillna('Unknown').astype(str)

        df['title'] = df['title'].astype(str).str.strip()
        df['title_length'] = df['title'].str.len()

        df['title_uppercase_ratio'] = df['title'].apply(self._calculate_uppercase_ratio)
        df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))

        view_threshold = df['views'].quantile(0.995)
        df = df[df['views'] <= view_threshold]

        df = df[df['duration_seconds'] <= 14400]

        logger.info(f"Cleaned data: {len(df)} records remaining")

        return df

    def clean_api_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning API data...")

        df = df.copy()

        initial_count = len(df)
        df = df.drop_duplicates(subset=['video_id'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} duplicates")

        df = df.dropna(subset=['video_id', 'title'])
        logger.info(f"After removing null critical fields: {len(df)} records")

        numeric_fields = ['view_count', 'like_count', 'comment_count', 'duration_seconds',
                         'channel_subscriber_count', 'channel_video_count', 'channel_total_views']

        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0).astype(int)

        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

        df['days_since_upload'] = df['published_at'].apply(
            lambda x: (pd.Timestamp.now(tz='UTC').replace(tzinfo=None) - x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else pd.Timestamp.now() - x).days if pd.notna(x) else np.nan
        )

        df['title'] = df['title'].astype(str).str.strip()
        df['description'] = df['description'].fillna('').astype(str)
        df['tags'] = df['tags'].fillna('').astype(str)
        df['channel_name'] = df['channel_name'].fillna('Unknown').astype(str)

        df['title_length'] = df['title'].str.len()
        df['title_uppercase_ratio'] = df['title'].apply(self._calculate_uppercase_ratio)
        df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))

        df['tag_count'] = df['tags'].apply(lambda x: len(x.split('|')) if x else 0)

        df['description_length'] = df['description'].str.len()

        if 'category_name' not in df.columns and 'category_id' in df.columns:
            df['category_name'] = df['category_id'].astype(str).map(self.category_mapping)

        df['category_name'] = df['category_name'].fillna('Unknown')

        df['engagement_rate'] = np.where(
            df['view_count'] > 0,
            (df['like_count'] + df['comment_count']) / df['view_count'],
            0
        )

        view_threshold = df['view_count'].quantile(0.995)
        df = df[df['view_count'] <= view_threshold]

        df = df[df['duration_seconds'] <= 14400]

        logger.info(f"Cleaned data: {len(df)} records remaining")

        return df

    def _parse_relative_date(self, date_str: str) -> datetime:
        try:
            if not isinstance(date_str, str):
                return datetime.now()

            date_str = date_str.lower()

            numbers = re.findall(r'\d+', date_str)
            if not numbers:
                return datetime.now()

            num = int(numbers[0])

            if 'second' in date_str or 'sec' in date_str:
                return datetime.now()
            elif 'minute' in date_str or 'min' in date_str:
                return datetime.now()
            elif 'hour' in date_str:
                return datetime.now()
            elif 'day' in date_str:
                return datetime.now() - pd.Timedelta(days=num)
            elif 'week' in date_str:
                return datetime.now() - pd.Timedelta(weeks=num)
            elif 'month' in date_str:
                return datetime.now() - pd.Timedelta(days=num * 30)
            elif 'year' in date_str:
                return datetime.now() - pd.Timedelta(days=num * 365)
            else:
                return datetime.now()

        except:
            return datetime.now()

    def _calculate_uppercase_ratio(self, text: str) -> float:
        if not text or not isinstance(text, str):
            return 0.0

        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0

        uppercase_count = sum(1 for c in letters if c.isupper())
        return uppercase_count / len(letters)

    def normalize_scraped_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Normalizing scraped data...")

        normalized = pd.DataFrame()

        normalized['video_id'] = df['video_id']
        normalized['title'] = df['title']
        normalized['channel_name'] = df['channel_name']
        normalized['views'] = df['views']
        normalized['duration_seconds'] = df['duration_seconds']
        normalized['duration_minutes'] = df['duration_seconds'] / 60
        normalized['days_since_upload'] = df['days_since_upload']

        normalized['title_length'] = df['title_length']
        normalized['title_uppercase_ratio'] = df['title_uppercase_ratio']
        normalized['title_word_count'] = df['title_word_count']

        normalized['views_per_day'] = np.where(
            df['days_since_upload'] > 0,
            df['views'] / df['days_since_upload'],
            df['views']
        )

        normalized['data_source'] = 'scraped'

        logger.info(f"Normalized scraped data: {len(normalized)} records")

        return normalized

    def normalize_api_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Normalizing API data...")

        normalized = pd.DataFrame()

        normalized['video_id'] = df['video_id']
        normalized['title'] = df['title']
        normalized['channel_name'] = df['channel_name']
        normalized['views'] = df['view_count']
        normalized['duration_seconds'] = df['duration_seconds']
        normalized['duration_minutes'] = df['duration_seconds'] / 60
        normalized['days_since_upload'] = df['days_since_upload']

        normalized['title_length'] = df['title_length']
        normalized['title_uppercase_ratio'] = df['title_uppercase_ratio']
        normalized['title_word_count'] = df['title_word_count']

        normalized['likes'] = df['like_count']
        normalized['comments'] = df['comment_count']
        normalized['engagement_rate'] = df['engagement_rate']
        normalized['tag_count'] = df['tag_count']
        normalized['description_length'] = df['description_length']
        normalized['category_name'] = df['category_name']

        normalized['channel_subscribers'] = df.get('channel_subscriber_count', 0)
        normalized['channel_video_count'] = df.get('channel_video_count', 0)

        normalized['likes_per_view'] = np.where(
            df['view_count'] > 0,
            df['like_count'] / df['view_count'],
            0
        )

        normalized['comments_per_view'] = np.where(
            df['view_count'] > 0,
            df['comment_count'] / df['view_count'],
            0
        )

        normalized['views_per_day'] = np.where(
            df['days_since_upload'] > 0,
            df['view_count'] / df['days_since_upload'],
            df['view_count']
        )

        normalized['data_source'] = 'api'

        logger.info(f"Normalized API data: {len(normalized)} records")

        return normalized

    def save_processed_data(self, df: pd.DataFrame, filepath: str):
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Saved processed data to {filepath}")

    def get_summary_statistics(self, df: pd.DataFrame) -> dict:
        stats = {
            'total_records': len(df),
            'numeric_summary': df.describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }

        return stats


def main():

    preprocessor = DataPreprocessor()

    print("="*60)
    print("Data Preprocessing")
    print("="*60)

    try:
        print("\n1. Processing Scraped Data...")
        scraped_df = preprocessor.load_scraped_data('data/raw/scraped_data.csv')
        scraped_cleaned = preprocessor.clean_scraped_data(scraped_df)
        scraped_normalized = preprocessor.normalize_scraped_data(scraped_cleaned)
        preprocessor.save_processed_data(scraped_normalized, 'data/processed/scraped_processed.csv')

        print(f"\nScraped Data Summary:")
        print(f"- Original records: {len(scraped_df)}")
        print(f"- After cleaning: {len(scraped_cleaned)}")
        print(f"- Final normalized: {len(scraped_normalized)}")

        print(f"\nSample of processed scraped data:")
        print(scraped_normalized.head(2))

    except FileNotFoundError:
        print("\nScraped data file not found. Please run scraper first.")
    except Exception as e:
        print(f"\nError processing scraped data: {str(e)}")

    try:
        print("\n2. Processing API Data...")
        api_df = preprocessor.load_api_data('data/raw/api_data.csv')
        api_cleaned = preprocessor.clean_api_data(api_df)
        api_normalized = preprocessor.normalize_api_data(api_cleaned)
        preprocessor.save_processed_data(api_normalized, 'data/processed/api_processed.csv')

        print(f"\nAPI Data Summary:")
        print(f"- Original records: {len(api_df)}")
        print(f"- After cleaning: {len(api_cleaned)}")
        print(f"- Final normalized: {len(api_normalized)}")

        print(f"\nSample of processed API data:")
        print(api_normalized.head(2))

    except FileNotFoundError:
        print("\nAPI data file not found. Please run API collector first.")
    except Exception as e:
        print(f"\nError processing API data: {str(e)}")

    print("\n" + "="*60)
    print("Preprocessing completed!")
    print("="*60)


if __name__ == '__main__':
    main()
