
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from typing import Dict, List
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Visualizer:

    def __init__(self, output_dir: str = 'reports/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_results(self, filepath: str) -> Dict:
        with open(filepath, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded results from {filepath}")
        return results

    def plot_model_comparison(self, results_scraped: Dict, results_api: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison: Scraped vs API Data', fontsize=16, fontweight='bold')

        metrics = ['r2', 'rmse', 'mae', 'mape']
        titles = ['R² Score (Higher is Better)', 'RMSE (Lower is Better)',
                  'MAE (Lower is Better)', 'MAPE % (Lower is Better)']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            data = {
                'Scraped\nRandom Forest': results_scraped['random_forest']['test_metrics'][metric],
                'Scraped\nXGBoost': results_scraped['xgboost']['test_metrics'][metric],
                'API\nRandom Forest': results_api['random_forest']['test_metrics'][metric],
                'API\nXGBoost': results_api['xgboost']['test_metrics'][metric]
            }

            models = list(data.keys())
            values = list(data.values())

            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold')

            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = f"{self.output_dir}/model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved model comparison plot to {output_path}")

    def plot_feature_importance(self, results: Dict, data_source: str):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Feature Importance - {data_source.upper()} Data',
                    fontsize=16, fontweight='bold')

        for idx, (model_type, ax) in enumerate(zip(['random_forest', 'xgboost'], axes)):
            fi_data = results[model_type]['feature_importance'][:15]
            features = [item['feature'] for item in fi_data]
            importance = [item['importance'] for item in fi_data]

            y_pos = np.arange(len(features))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))

            ax.barh(y_pos, importance, color=colors, edgecolor='black', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title(f'{model_type.replace("_", " ").title()}', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            for i, v in enumerate(importance):
                ax.text(v, i, f' {v:.3f}', va='center', fontweight='bold')

        plt.tight_layout()
        output_path = f"{self.output_dir}/feature_importance_{data_source}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature importance plot to {output_path}")

    def plot_engagement_by_category(self, df: pd.DataFrame):
        if 'category_name' not in df.columns:
            logger.warning("Category information not available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Engagement Trends by Video Category', fontsize=16, fontweight='bold')

        ax1 = axes[0, 0]
        category_views = df.groupby('category_name')['views'].mean().sort_values(ascending=False).head(10)
        category_views.plot(kind='barh', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Average Views by Category', fontweight='bold')
        ax1.set_xlabel('Average Views')
        ax1.grid(axis='x', alpha=0.3)

        ax2 = axes[0, 1]
        if 'engagement_rate' in df.columns:
            category_engagement = df.groupby('category_name')['engagement_rate'].mean().sort_values(ascending=False).head(10)
            category_engagement.plot(kind='barh', ax=ax2, color='lightcoral', edgecolor='black')
            ax2.set_title('Average Engagement Rate by Category', fontweight='bold')
            ax2.set_xlabel('Engagement Rate')
            ax2.grid(axis='x', alpha=0.3)

        ax3 = axes[1, 0]
        category_count = df['category_name'].value_counts().head(10)
        category_count.plot(kind='barh', ax=ax3, color='lightgreen', edgecolor='black')
        ax3.set_title('Video Count by Category', fontweight='bold')
        ax3.set_xlabel('Number of Videos')
        ax3.grid(axis='x', alpha=0.3)

        ax4 = axes[1, 1]
        if 'likes_per_view' in df.columns:
            category_likes = df.groupby('category_name')['likes_per_view'].mean().sort_values(ascending=False).head(10)
            category_likes.plot(kind='barh', ax=ax4, color='plum', edgecolor='black')
            ax4.set_title('Average Likes per View by Category', fontweight='bold')
            ax4.set_xlabel('Likes per View')
            ax4.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_path = f"{self.output_dir}/engagement_by_category.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved engagement by category plot to {output_path}")

    def plot_engagement_by_duration(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Engagement Trends by Video Duration', fontsize=16, fontweight='bold')

        ax1 = axes[0, 0]
        sample = df.sample(min(1000, len(df)))
        ax1.scatter(sample['duration_minutes'], sample['views'],
                   alpha=0.5, s=20, c='steelblue')
        ax1.set_xlabel('Duration (minutes)')
        ax1.set_ylabel('Views')
        ax1.set_title('Views vs Duration', fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(alpha=0.3)

        ax2 = axes[0, 1]
        if 'duration_category' in df.columns:
            duration_views = df.groupby('duration_category')['views'].mean()
            duration_views.plot(kind='bar', ax=ax2, color='orange', edgecolor='black', alpha=0.7)
            ax2.set_title('Average Views by Duration Category', fontweight='bold')
            ax2.set_ylabel('Average Views')
            ax2.set_xlabel('Duration Category')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)

        ax3 = axes[1, 0]
        df['duration_minutes'].hist(bins=50, ax=ax3, color='teal', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Duration (minutes)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Video Durations', fontweight='bold')
        ax3.grid(alpha=0.3)

        ax4 = axes[1, 1]
        if 'engagement_rate' in df.columns and 'duration_category' in df.columns:
            duration_engagement = df.groupby('duration_category')['engagement_rate'].mean()
            duration_engagement.plot(kind='bar', ax=ax4, color='crimson', edgecolor='black', alpha=0.7)
            ax4.set_title('Average Engagement Rate by Duration', fontweight='bold')
            ax4.set_ylabel('Engagement Rate')
            ax4.set_xlabel('Duration Category')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(axis='y', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Engagement data not available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.axis('off')

        plt.tight_layout()
        output_path = f"{self.output_dir}/engagement_by_duration.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved engagement by duration plot to {output_path}")

    def plot_engagement_by_upload_time(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Engagement Trends by Upload Time', fontsize=16, fontweight='bold')

        ax1 = axes[0, 0]
        sample = df.sample(min(1000, len(df)))
        ax1.scatter(sample['days_since_upload'], sample['views'],
                   alpha=0.5, s=20, c='forestgreen')
        ax1.set_xlabel('Days Since Upload')
        ax1.set_ylabel('Views')
        ax1.set_title('Views vs Days Since Upload', fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(alpha=0.3)

        ax2 = axes[0, 1]
        if 'video_age_category' in df.columns:
            age_views = df.groupby('video_age_category')['views_per_day'].mean()
            age_views.plot(kind='bar', ax=ax2, color='purple', edgecolor='black', alpha=0.7)
            ax2.set_title('Average Views per Day by Video Age', fontweight='bold')
            ax2.set_ylabel('Views per Day')
            ax2.set_xlabel('Video Age Category')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)

        ax3 = axes[1, 0]
        df['days_since_upload'].hist(bins=50, ax=ax3, color='darkorange', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Days Since Upload')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Video Ages', fontweight='bold')
        ax3.grid(alpha=0.3)

        ax4 = axes[1, 1]
        if 'engagement_rate' in df.columns and 'video_age_category' in df.columns:
            age_engagement = df.groupby('video_age_category')['engagement_rate'].mean()
            age_engagement.plot(kind='bar', ax=ax4, color='navy', edgecolor='black', alpha=0.7)
            ax4.set_title('Average Engagement Rate by Video Age', fontweight='bold')
            ax4.set_ylabel('Engagement Rate')
            ax4.set_xlabel('Video Age Category')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(axis='y', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Engagement data not available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.axis('off')

        plt.tight_layout()
        output_path = f"{self.output_dir}/engagement_by_upload_time.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved engagement by upload time plot to {output_path}")

    def plot_correlation_heatmap(self, df: pd.DataFrame, data_source: str):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        key_features = [col for col in numeric_cols if not col.endswith('_encoded')][:15]

        if len(key_features) < 2:
            logger.warning("Not enough numerical features for correlation plot")
            return

        corr_matrix = df[key_features].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title(f'Feature Correlation Heatmap - {data_source.upper()} Data',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        output_path = f"{self.output_dir}/correlation_heatmap_{data_source}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlation heatmap to {output_path}")

    def plot_data_distribution(self, df: pd.DataFrame, data_source: str):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Data Distribution - {data_source.upper()} Data',
                    fontsize=16, fontweight='bold')

        axes[0, 0].hist(df['views'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Views')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Views Distribution')
        axes[0, 0].set_yscale('log')

        axes[0, 1].hist(df['log_views'], bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Log(Views)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Log-Views Distribution (Target)')

        axes[0, 2].hist(df['duration_minutes'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Duration (minutes)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Duration Distribution')

        axes[1, 0].hist(df['days_since_upload'], bins=50, color='purple', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Days Since Upload')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Video Age Distribution')

        axes[1, 1].hist(df['title_length'], bins=50, color='orange', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Title Length (characters)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Title Length Distribution')

        if 'engagement_rate' in df.columns:
            axes[1, 2].hist(df['engagement_rate'], bins=50, color='crimson', edgecolor='black', alpha=0.7)
            axes[1, 2].set_xlabel('Engagement Rate')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Engagement Rate Distribution')
        else:
            axes[1, 2].text(0.5, 0.5, 'Engagement data\nnot available',
                          ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].axis('off')

        plt.tight_layout()
        output_path = f"{self.output_dir}/data_distribution_{data_source}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved data distribution plot to {output_path}")


def main():

    visualizer = Visualizer()

    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    try:
        results_scraped = visualizer.load_results('reports/results_scraped.json')
        results_api = visualizer.load_results('reports/results_api.json')

        print("\n1. Creating model comparison plots...")
        visualizer.plot_model_comparison(results_scraped, results_api)

        print("2. Creating feature importance plots...")
        visualizer.plot_feature_importance(results_scraped, 'scraped')
        visualizer.plot_feature_importance(results_api, 'api')

    except FileNotFoundError as e:
        print(f"\nResults files not found: {e}")
        print("Please train models first.")

    try:
        print("\n3. Creating engagement analysis plots for API data...")
        df_api = pd.read_csv('data/processed/api_features.csv')

        visualizer.plot_engagement_by_category(df_api)
        visualizer.plot_engagement_by_duration(df_api)
        visualizer.plot_engagement_by_upload_time(df_api)
        visualizer.plot_correlation_heatmap(df_api, 'api')
        visualizer.plot_data_distribution(df_api, 'api')

    except FileNotFoundError:
        print("\nAPI feature data not found.")

    try:
        print("\n4. Creating analysis plots for scraped data...")
        df_scraped = pd.read_csv('data/processed/scraped_features.csv')

        visualizer.plot_engagement_by_duration(df_scraped)
        visualizer.plot_engagement_by_upload_time(df_scraped)
        visualizer.plot_correlation_heatmap(df_scraped, 'scraped')
        visualizer.plot_data_distribution(df_scraped, 'scraped')

    except FileNotFoundError:
        print("\nScraped feature data not found.")

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETED")
    print(f"All figures saved to: reports/figures/")
    print("="*60)


if __name__ == '__main__':
    main()
