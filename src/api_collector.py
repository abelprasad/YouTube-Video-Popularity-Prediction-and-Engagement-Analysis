
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YouTubeAPICollector:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "YouTube API key not found. "
                "Set YOUTUBE_API_KEY environment variable or pass api_key parameter."
            )

        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.quota_used = 0
        logger.info("YouTube API client initialized successfully")

    def search_videos(self, query: str, max_results: int = 50, order: str = 'relevance') -> List[str]:
        try:
            request = self.youtube.search().list(
                part='id',
                q=query,
                type='video',
                maxResults=min(max_results, 50),
                order=order,
                relevanceLanguage='en',
                safeSearch='none'
            )
            response = request.execute()
            self.quota_used += 100

            video_ids = [item['id']['videoId'] for item in response.get('items', [])]
            return video_ids

        except HttpError as e:
            logger.error(f"HTTP error during search: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []

    def get_video_details(self, video_ids: List[str]) -> List[Dict]:
        if not video_ids:
            return []

        try:
            chunks = [video_ids[i:i + 50] for i in range(0, len(video_ids), 50)]
            all_videos = []

            for chunk in chunks:
                request = self.youtube.videos().list(
                    part='snippet,contentDetails,statistics',
                    id=','.join(chunk)
                )
                response = request.execute()
                self.quota_used += 1

                for item in response.get('items', []):
                    video_data = self._parse_video_item(item)
                    if video_data:
                        all_videos.append(video_data)

                time.sleep(0.1)

            return all_videos

        except HttpError as e:
            logger.error(f"HTTP error getting video details: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error getting video details: {str(e)}")
            return []

    def _parse_video_item(self, item: Dict) -> Optional[Dict]:
        try:
            video_id = item['id']
            snippet = item.get('snippet', {})
            content_details = item.get('contentDetails', {})
            statistics = item.get('statistics', {})

            duration_iso = content_details.get('duration', 'PT0S')
            duration_seconds = self._parse_iso_duration(duration_iso)

            tags = snippet.get('tags', [])
            tags_str = '|'.join(tags[:10]) if tags else ''

            view_count = int(statistics.get('viewCount', 0))
            like_count = int(statistics.get('likeCount', 0))
            comment_count = int(statistics.get('commentCount', 0))

            return {
                'video_id': video_id,
                'title': snippet.get('title', ''),
                'description': snippet.get('description', '')[:500],
                'channel_id': snippet.get('channelId', ''),
                'channel_name': snippet.get('channelTitle', ''),
                'category_id': snippet.get('categoryId', ''),
                'published_at': snippet.get('publishedAt', ''),
                'duration_seconds': duration_seconds,
                'tags': tags_str,
                'view_count': view_count,
                'like_count': like_count,
                'comment_count': comment_count,
                'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                'video_url': f"https://www.youtube.com/watch?v={video_id}",
                'collected_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logger.debug(f"Error parsing video item: {str(e)}")
            return None

    def _parse_iso_duration(self, duration: str) -> int:
        import re

        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration)

        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    def get_channel_details(self, channel_ids: List[str]) -> Dict[str, Dict]:
        if not channel_ids:
            return {}

        try:
            channel_ids = list(set(channel_ids))

            chunks = [channel_ids[i:i + 50] for i in range(0, len(channel_ids), 50)]
            channel_data = {}

            for chunk in chunks:
                request = self.youtube.channels().list(
                    part='snippet,statistics',
                    id=','.join(chunk)
                )
                response = request.execute()
                self.quota_used += 1

                for item in response.get('items', []):
                    channel_id = item['id']
                    statistics = item.get('statistics', {})

                    channel_data[channel_id] = {
                        'subscriber_count': int(statistics.get('subscriberCount', 0)),
                        'video_count': int(statistics.get('videoCount', 0)),
                        'view_count': int(statistics.get('viewCount', 0))
                    }

                time.sleep(0.1)

            return channel_data

        except Exception as e:
            logger.error(f"Error getting channel details: {str(e)}")
            return {}

    def get_video_categories(self, region_code: str = 'US') -> Dict[str, str]:
        try:
            request = self.youtube.videoCategories().list(
                part='snippet',
                regionCode=region_code
            )
            response = request.execute()
            self.quota_used += 1

            categories = {}
            for item in response.get('items', []):
                category_id = item['id']
                category_name = item['snippet']['title']
                categories[category_id] = category_name

            return categories

        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            return {}

    def collect_from_queries(self, queries: List[str], videos_per_query: int = 50) -> pd.DataFrame:
        all_video_ids = []

        logger.info(f"Collecting video IDs from {len(queries)} queries...")

        for query in tqdm(queries, desc="Searching"):
            try:
                video_ids = self.search_videos(query, max_results=videos_per_query)
                all_video_ids.extend(video_ids)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error searching for '{query}': {str(e)}")
                continue

        all_video_ids = list(set(all_video_ids))
        logger.info(f"Collected {len(all_video_ids)} unique video IDs")

        logger.info("Fetching video details...")
        all_videos = self.get_video_details(all_video_ids)

        df = pd.DataFrame(all_videos)

        if not df.empty and 'channel_id' in df.columns:
            logger.info("Fetching channel details...")
            channel_ids = df['channel_id'].unique().tolist()
            channel_data = self.get_channel_details(channel_ids)

            df['channel_subscriber_count'] = df['channel_id'].map(
                lambda x: channel_data.get(x, {}).get('subscriber_count', 0)
            )
            df['channel_video_count'] = df['channel_id'].map(
                lambda x: channel_data.get(x, {}).get('video_count', 0)
            )
            df['channel_total_views'] = df['channel_id'].map(
                lambda x: channel_data.get(x, {}).get('view_count', 0)
            )

        logger.info("Fetching category names...")
        categories = self.get_video_categories()
        if not df.empty and 'category_id' in df.columns:
            df['category_name'] = df['category_id'].map(categories)

        logger.info(f"Total videos collected: {len(df)}")
        logger.info(f"Total API quota used: {self.quota_used} units")

        return df

    def save_data(self, df: pd.DataFrame, filepath: str):
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Data saved to {filepath}")


def main():

    search_queries = [
        'python tutorial',
        'machine learning',
        'cooking recipes',
        'travel vlog',
        'music video',
        'gaming',
        'movie trailers',
        'sports highlights',
        'tech reviews',
        'comedy sketches',
        'news today',
        'fitness workout',
        'fashion haul',
        'diy projects',
        'documentary',
        'animation',
        'podcast',
        'book review',
        'car review',
        'science experiment',
        'art tutorial',
        'dance',
        'meditation',
        'educational',
        'unboxing',
        'reaction video',
        'photography tips',
        'investing',
        'cryptocurrency',
        'gardening',
        'product review',
        'how to',
        'interview',
        'live stream',
        'vlog',
        'challenge',
        'prank',
        'haul',
        'routine',
        'story time',
        'motivational',
        'short film',
        'trailer',
        'behind the scenes',
        'acoustic',
        'cover song',
        'gameplay',
        'walkthrough',
        'speed run',
        'highlights',
        'compilation',
        'top 10',
        'best of',
        'vs',
        'comparison',
        'explained',
        'breakdown',
        'analysis',
        'review',
        'reaction',
        'first time',
        'unboxing'
    ]

    print("="*60)
    print("YouTube API Data Collector")
    print("="*60)
    print("\nIMPORTANT: You need a YouTube Data API v3 key to run this.")
    print("1. Go to: https://console.cloud.google.com/")
    print("2. Create a project and enable YouTube Data API v3")
    print("3. Create credentials (API key)")
    print("4. Set the API key in .env file: YOUTUBE_API_KEY=your_key_here")
    print("\nNote: YouTube API has daily quota limits (10,000 units/day)")
    print("This script will use approximately 3,000-4,000 quota units")
    print("="*60)

    try:
        collector = YouTubeAPICollector()

        df = collector.collect_from_queries(search_queries, videos_per_query=50)

        output_path = 'data/raw/api_data.csv'
        collector.save_data(df, output_path)

        print(f"\n{'='*60}")
        print(f"API collection completed!")
        print(f"Total videos collected: {len(df)}")
        print(f"Total API quota used: {collector.quota_used} units")
        print(f"Data saved to: {output_path}")
        print(f"{'='*60}")

    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease set up your YouTube API key before running this script.")


if __name__ == '__main__':
    main()
