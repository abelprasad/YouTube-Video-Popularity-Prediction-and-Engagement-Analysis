
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YouTubeScraper:

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.driver = None
        self.videos_data = []

    def setup_driver(self):
        options = uc.ChromeOptions()

        if self.headless:
            options.add_argument('--headless')

        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-notifications')
        options.add_argument('--start-maximized')

        self.driver = uc.Chrome(options=options, version_main=141)
        logger.info("Chrome driver initialized successfully")

    def close_driver(self):
        if self.driver:
            self.driver.quit()
            logger.info("Chrome driver closed")

    def parse_view_count(self, view_text: str) -> int:
        if not view_text:
            return 0

        view_text = view_text.lower().replace('views', '').replace('view', '').strip()

        if 'no' in view_text or view_text == '':
            return 0

        view_text = view_text.replace(',', '')

        multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}

        for suffix, multiplier in multipliers.items():
            if suffix in view_text:
                try:
                    number = float(view_text.replace(suffix, '').strip())
                    return int(number * multiplier)
                except ValueError:
                    return 0

        try:
            return int(float(view_text))
        except ValueError:
            return 0

    def parse_duration(self, duration_text: str) -> int:
        if not duration_text:
            return 0

        try:
            parts = duration_text.split(':')
            parts = [int(p) for p in parts]

            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
            elif len(parts) == 1:
                return parts[0]
        except:
            return 0

        return 0

    def parse_upload_date(self, date_text: str) -> str:
        if not date_text:
            return datetime.now().strftime('%Y-%m-%d')

        return date_text

    def scroll_page(self, scroll_pause_time: float = 2.0, max_scrolls: int = 50):
        last_height = self.driver.execute_script("return document.documentElement.scrollHeight")
        scrolls = 0

        while scrolls < max_scrolls:
            self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(scroll_pause_time)

            new_height = self.driver.execute_script("return document.documentElement.scrollHeight")

            if new_height == last_height:
                break

            last_height = new_height
            scrolls += 1

        logger.info(f"Scrolled {scrolls} times")

    def extract_video_data(self, video_element) -> Optional[Dict]:
        try:
            title_element = video_element.find_element(By.CSS_SELECTOR, '#video-title')
            title = title_element.get_attribute('title') or title_element.text
            video_url = title_element.get_attribute('href')

            if not title or not video_url:
                return None

            video_id_match = re.search(r'watch\?v=([^&]+)', video_url)
            video_id = video_id_match.group(1) if video_id_match else None

            try:
                channel_element = video_element.find_element(By.CSS_SELECTOR, '#channel-name a')
                channel_name = channel_element.text
            except:
                channel_name = 'Unknown'

            try:
                metadata_element = video_element.find_element(By.CSS_SELECTOR, '#metadata-line')
                metadata_text = metadata_element.text
                metadata_parts = metadata_text.split('\n')

                views_text = metadata_parts[0] if len(metadata_parts) > 0 else '0 views'
                upload_date = metadata_parts[1] if len(metadata_parts) > 1 else 'Unknown'

                views = self.parse_view_count(views_text)
            except:
                views = 0
                upload_date = 'Unknown'

            try:
                duration_element = video_element.find_element(By.CSS_SELECTOR, 'span.style-scope.ytd-thumbnail-overlay-time-status-renderer')
                duration_text = duration_element.text
                duration_seconds = self.parse_duration(duration_text)
            except:
                duration_seconds = 0

            try:
                thumbnail_element = video_element.find_element(By.CSS_SELECTOR, 'img#img')
                thumbnail_url = thumbnail_element.get_attribute('src')
            except:
                thumbnail_url = None

            return {
                'video_id': video_id,
                'title': title,
                'channel_name': channel_name,
                'views': views,
                'upload_date': upload_date,
                'duration_seconds': duration_seconds,
                'video_url': video_url,
                'thumbnail_url': thumbnail_url,
                'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logger.debug(f"Error extracting video data: {str(e)}")
            return None

    def scrape_search_results(self, query: str, target_count: int = 100) -> List[Dict]:
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        self.driver.get(search_url)

        logger.info(f"Searching for: {query}")
        time.sleep(3)

        scrolls_needed = (target_count // 20) + 2
        self.scroll_page(scroll_pause_time=2.0, max_scrolls=scrolls_needed)

        video_elements = self.driver.find_elements(By.CSS_SELECTOR, 'ytd-video-renderer')
        logger.info(f"Found {len(video_elements)} video elements")

        videos = []
        for element in tqdm(video_elements[:target_count], desc=f"Extracting '{query}'"):
            video_data = self.extract_video_data(element)
            if video_data:
                videos.append(video_data)

        return videos

    def scrape_trending(self, target_count: int = 100) -> List[Dict]:
        trending_url = "https://www.youtube.com/feed/trending"
        self.driver.get(trending_url)

        logger.info("Scraping trending videos")
        time.sleep(3)

        self.scroll_page(scroll_pause_time=2.0, max_scrolls=10)

        video_elements = self.driver.find_elements(By.CSS_SELECTOR, 'ytd-video-renderer')
        logger.info(f"Found {len(video_elements)} video elements")

        videos = []
        for element in tqdm(video_elements[:target_count], desc="Extracting trending"):
            video_data = self.extract_video_data(element)
            if video_data:
                videos.append(video_data)

        return videos

    def scrape_multiple_queries(self, queries: List[str], videos_per_query: int = 100, target_count: int = 3000) -> pd.DataFrame:
        all_videos = []
        unique_video_ids = set()

        try:
            self.setup_driver()

            query_index = 0
            while len(unique_video_ids) < target_count:
                query = queries[query_index % len(queries)]

                try:
                    videos = self.scrape_search_results(query, videos_per_query)

                    for video in videos:
                        if video.get('video_id') and video['video_id'] not in unique_video_ids:
                            all_videos.append(video)
                            unique_video_ids.add(video['video_id'])

                    logger.info(f"Collected {len(videos)} videos for query: {query}. Total unique: {len(unique_video_ids)}/{target_count}")

                    if len(unique_video_ids) >= target_count:
                        logger.info(f"Target of {target_count} videos reached!")
                        break

                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error scraping query '{query}': {str(e)}")

                query_index += 1

            df = pd.DataFrame(all_videos)
            logger.info(f"Total unique videos collected: {len(df)}")
            return df

        finally:
            self.close_driver()

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
        'makeup tutorial',
        'weight loss tips',
        'home workout',
        'healthy recipes',
        'pet training',
        'motivation speech',
        'guitar lessons',
        'piano tutorial',
        'how to draw',
        'origami tutorial',
        'woodworking projects',
        'home renovation',
        'interior design',
        'personal finance',
        'stock market',
        'real estate investing',
        'entrepreneur advice',
        'business tips',
        'productivity hacks',
        'study tips',
        'language learning',
        'history documentary',
        'space exploration',
        'wildlife documentary',
        'ocean life',
        'climate change',
        'renewable energy',
        'electric vehicles',
        'smartphone review',
        'laptop review',
        'gaming setup',
        'pc build guide',
        'coding tutorial',
        'web development',
        'app development',
        'data science',
        'artificial intelligence',
        'robotics',
        'drone footage',
        'gopro adventures',
        'camping tips',
        'hiking trails',
        'backpacking',
        'fishing tips',
        'motorcycle vlog',
        'car detailing',
        'sports training',
        'soccer skills',
        'basketball tricks',
        'yoga for beginners',
        'stretching routine',
        'meal prep',
        'baking recipes',
        'dessert recipes',
        'street food',
        'food review',
        'restaurant tour',
        'city tour',
        'culture vlog',
        'budget travel',
        'luxury travel',
        'hotel review',
        'airline review',
        'stand up comedy',
        'magic tricks',
        'card tricks',
        'illusion explained',
        'true crime',
        'mystery stories',
        'paranormal',
        'asmr'
    ]

    scraper = YouTubeScraper(headless=False)

    df = scraper.scrape_multiple_queries(search_queries, videos_per_query=100)

    output_path = 'data/raw/scraped_data.csv'
    scraper.save_data(df, output_path)

    print(f"\n{'='*50}")
    print(f"Scraping completed!")
    print(f"Total videos collected: {len(df)}")
    print(f"Data saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
