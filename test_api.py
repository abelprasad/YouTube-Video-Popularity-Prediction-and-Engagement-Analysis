
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

def test_api_connection():

    print("=" * 60)
    print("  YouTube API Connection Test")
    print("=" * 60)

    api_key = os.getenv('YOUTUBE_API_KEY')

    if not api_key:
        print("\n[ERROR] No API key found!")
        print("\nTo fix this:")
        print("1. Open the .env file")
        print("2. Add: YOUTUBE_API_KEY=your_api_key_here")
        print("3. Get your API key from: https://console.cloud.google.com/")
        return False

    print(f"\n[OK] API key found: {api_key[:10]}...{api_key[-4:]}")

    try:
        print("\n[1/3] Initializing YouTube API client...")
        youtube = build('youtube', 'v3', developerKey=api_key)
        print("[OK] Client initialized successfully")

        print("\n[2/3] Testing search request...")
        request = youtube.search().list(
            part='snippet',
            q='python programming',
            type='video',
            maxResults=5
        )
        response = request.execute()
        print(f"[OK] Search successful! Found {len(response.get('items', []))} videos")

        if response.get('items'):
            video_id = response['items'][0]['id']['videoId']
            print(f"\n[3/3] Testing video details request for ID: {video_id}...")

            request = youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            )
            video_response = request.execute()

            if video_response.get('items'):
                video = video_response['items'][0]
                print("[OK] Video details retrieved successfully!")
                print(f"\nSample video data:")
                print(f"  Title: {video['snippet']['title'][:50]}...")
                print(f"  Views: {video['statistics'].get('viewCount', 'N/A')}")
                print(f"  Likes: {video['statistics'].get('likeCount', 'N/A')}")
                print(f"  Duration: {video['contentDetails']['duration']}")

        print("\n" + "=" * 60)
        print("  [SUCCESS] API CONNECTION TEST PASSED!")
        print("=" * 60)
        print("\nYour API key is working correctly.")
        print("You can now run: python src/api_collector.py")
        return True

    except HttpError as e:
        print(f"\n[ERROR] API Error: {e}")

        if e.resp.status == 403:
            print("\nPossible causes:")
            print("1. Invalid API key")
            print("2. YouTube Data API v3 not enabled in Google Cloud Console")
            print("3. API quota exceeded")
            print("\nTo fix:")
            print("- Verify your API key is correct")
            print("- Enable YouTube Data API v3 at: https://console.cloud.google.com/")

        return False

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return False

if __name__ == '__main__':
    test_api_connection()
