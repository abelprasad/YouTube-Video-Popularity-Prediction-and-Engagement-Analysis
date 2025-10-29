
import subprocess
import sys
import os
from datetime import datetime


def print_header(message):
    print("\n" + "="*70)
    print(f"  {message}")
    print("="*70 + "\n")


def run_command(command, description):
    print(f"Starting: {description}")
    print(f"Command: python {command}")
    print("-" * 70)

    try:
        result = subprocess.run(
            [sys.executable, command],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ Completed: {description}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in: {description}")
        print(f"Error: {str(e)}")
        return False

    except Exception as e:
        print(f"\n✗ Unexpected error in: {description}")
        print(f"Error: {str(e)}")
        return False


def check_api_key():
    if not os.path.exists('.env'):
        print("⚠ Warning: .env file not found")
        print("API collection will be skipped unless you have a .env file with YOUTUBE_API_KEY")
        return False
    return True


def main():

    print_header("YouTube Video Popularity Prediction - Complete Pipeline")

    start_time = datetime.now()

    results = {
        'scraping': False,
        'api_collection': False,
        'preprocessing': False,
        'feature_engineering': False,
        'model_training': False,
        'visualization': False
    }

    print_header("Step 1/6: Web Scraping (3000+ videos)")
    print("This will take approximately 30-60 minutes...")
    print("You can skip this if you already have scraped_data.csv")

    response = input("\nRun web scraping? (y/n/skip): ").lower()

    if response == 'y':
        results['scraping'] = run_command('src/scraper.py', 'Web Scraping')
    elif response == 'skip':
        print("Skipping web scraping (using existing data)")
        results['scraping'] = True
    else:
        print("Skipped web scraping")

    print_header("Step 2/6: YouTube API Data Collection")

    has_api_key = check_api_key()

    if has_api_key:
        response = input("\nRun API collection? (y/n/skip): ").lower()

        if response == 'y':
            results['api_collection'] = run_command('src/api_collector.py', 'API Collection')
        elif response == 'skip':
            print("Skipping API collection (using existing data)")
            results['api_collection'] = True
        else:
            print("Skipped API collection")
    else:
        print("Skipping API collection (no API key configured)")

    print_header("Step 3/6: Data Preprocessing")

    if results['scraping'] or results['api_collection']:
        response = input("\nRun preprocessing? (y/n): ").lower()

        if response == 'y':
            results['preprocessing'] = run_command('src/preprocessor.py', 'Data Preprocessing')
        else:
            print("Skipped preprocessing")
    else:
        print("Skipping preprocessing (no raw data available)")

    print_header("Step 4/6: Feature Engineering")

    if results['preprocessing']:
        response = input("\nRun feature engineering? (y/n): ").lower()

        if response == 'y':
            results['feature_engineering'] = run_command('src/feature_engineer.py', 'Feature Engineering')
        else:
            print("Skipped feature engineering")
    else:
        print("Skipping feature engineering (preprocessing not completed)")

    print_header("Step 5/6: Model Training")

    if results['feature_engineering']:
        response = input("\nRun model training? (y/n): ").lower()

        if response == 'y':
            results['model_training'] = run_command('src/model.py', 'Model Training')
        else:
            print("Skipped model training")
    else:
        print("Skipping model training (feature engineering not completed)")

    print_header("Step 6/6: Generate Visualizations")

    if results['model_training']:
        response = input("\nGenerate visualizations? (y/n): ").lower()

        if response == 'y':
            results['visualization'] = run_command('src/visualizer.py', 'Visualization')
        else:
            print("Skipped visualization")
    else:
        print("Skipping visualization (model training not completed)")

    print_header("Pipeline Execution Summary")

    end_time = datetime.now()
    duration = end_time - start_time

    print("Results:")
    print(f"  Web Scraping:        {'✓' if results['scraping'] else '✗'}")
    print(f"  API Collection:      {'✓' if results['api_collection'] else '✗'}")
    print(f"  Preprocessing:       {'✓' if results['preprocessing'] else '✗'}")
    print(f"  Feature Engineering: {'✓' if results['feature_engineering'] else '✗'}")
    print(f"  Model Training:      {'✓' if results['model_training'] else '✗'}")
    print(f"  Visualization:       {'✓' if results['visualization'] else '✗'}")

    print(f"\nTotal execution time: {duration}")

    if all([results['preprocessing'], results['feature_engineering'],
            results['model_training'], results['visualization']]):
        print("\n🎉 Pipeline completed successfully!")
        print("\nNext steps:")
        print("  1. Check reports/figures/ for visualizations")
        print("  2. Review reports/results_*.json for model metrics")
        print("  3. Use models in models/ directory for predictions")
    else:
        print("\n⚠ Pipeline incomplete. Review errors above.")

    print("\n" + "="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
