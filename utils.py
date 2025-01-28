from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io
import base64
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import plotly.express as px
import json
from collections import Counter
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import base64
from io import BytesIO
from typing import Dict, Any
from pydantic import BaseModel
import os
import joblib
from pydantic import BaseModel
import sklearn
import openai
import matplotlib.dates as mdates

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Ensure stopwords are downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def fig_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 string
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return img_str


def perform_eda_on_dataframe(df):
    results = {}
    try:
        # Basic DataFrame Information
        results['shape'] = list(df.shape)
        results['columns'] = list(df.columns)
        results['dtypes'] = df.dtypes.astype(str).to_dict()
        first_5_rows = df.head().to_dict('records')
        results['first_5_rows'] = first_5_rows

        # Missing Values Analysis
        missing_values = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        results['missing_values'] = {
            'counts': missing_values,
            'percentages': missing_percentages
        }

        # Tweet Content Analysis
        if 'Tweet' in df.columns:
            # Convert all entries in 'Tweet' column to string
            df['Tweet'] = df['Tweet'].astype(str)
            valid_tweets = df['Tweet'].dropna()

            # Tweet Length Distribution
            tweet_lengths = valid_tweets.apply(len)
            results['tweet_length_stats'] = tweet_lengths.describe().to_dict()

            plt.figure(figsize=(10, 6))
            tweet_lengths.hist(bins=30, color='skyblue', edgecolor='black')
            plt.xlabel("Tweet Length")
            plt.ylabel("Frequency")
            plt.title("Tweet Length Distribution")
            results['tweet_length_distribution'] = fig_to_base64(plt.gcf())
            plt.close()

            # Word Cloud
            all_text = ' '.join(valid_tweets).lower()
            all_text = re.sub(r'\b\w{1,2}\b', '', all_text)  # remove short words
            word_tokens = all_text.split()
            filtered_words = [word for word in word_tokens if word not in stop_words]
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            results['wordcloud'] = fig_to_base64(plt.gcf())
            plt.close()

            # Hashtag Analysis
            df['Hashtags'] = df['Tweet'].apply(lambda x: re.findall(r"#(\w+)", str(x)) if isinstance(x, str) else [])
            hashtag_list = sum(df['Hashtags'], [])
            hashtag_freq = Counter(hashtag_list)
            top_hashtags = dict(hashtag_freq.most_common(15))

            plt.figure(figsize=(12, 8))
            pd.Series(top_hashtags).plot(kind='bar', color='orange')
            plt.xlabel("Hashtags")
            plt.ylabel("Frequency")
            plt.title("Top 15 Hashtags")
            results['hashtag_distribution'] = fig_to_base64(plt.gcf())
            plt.close()

           # Top Users by Tweet Volume (Updated section)
        top_users = df['Username'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        top_users.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Top 10 Users by Tweet Volume', fontsize=16)
        plt.xlabel('Username', fontsize=14)
        plt.ylabel('Tweet Volume (Number of Tweets)', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, count in enumerate(top_users):
            plt.text(i, count + 5, str(count), ha='center', fontsize=12)
        plt.tight_layout()
        results['top_users'] = fig_to_base64(plt.gcf())
        plt.close()


        # Time Series Analysis
        if 'Date' in df.columns:
            df = clean_datetime_column(df, 'Date')

            # Daily Tweet Counts
                # daily_counts = df.groupby(df['Date'].dt.date).size()

                # # Convert the index to strings for x-axis labels
                # date_labels = [date.strftime('%Y-%m-%d') for date in daily_counts.index]

                # # Plotting
                # plt.figure(figsize=(10, 6))
                # daily_counts.plot(kind='line', marker='o', color='purple')
                # plt.xlabel("Date")
                # plt.ylabel("Number of Tweets")
                # plt.title("Daily Tweet Counts")
                # plt.xticks(ticks=range(len(daily_counts)), labels=date_labels, rotation=45, ha='right')  # Use formatted date labels
                # results['daily_tweet_counts'] = fig_to_base64(plt.gcf())
                # plt.close()



            # Tweet Activity by Day of the Week
            weekday_counts = df['Date'].dt.day_name().value_counts()
            plt.figure(figsize=(8, 5))
            weekday_counts.plot(kind='bar', color='blue')
            plt.xlabel("Weekday")
            plt.ylabel("Frequency")
            plt.title("Tweet Activity by Day of the Week")
            results['weekday_distribution'] = fig_to_base64(plt.gcf())
            plt.close()

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in EDA process: {str(e)}")


# Function to handle timezone-aware and timezone-naive datetime conversions
def serialize_datetime(obj):
    """
    Serialize datetime objects to ISO format strings.
    """
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def clean_datetime_column(df, date_column):
    """
    Clean a date column in the DataFrame to ensure all values are consistently timezone-naive datetime.
    """
    try:
        # Convert to datetime with flexible parsing
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Force timezone-naive if needed
        df[date_column] = df[date_column].apply(lambda x: x.tz_localize(None) if pd.notnull(x) and x.tzinfo else x)
        return df
    except Exception as e:
        logging.error(f"Date cleaning error: {str(e)}")
        print(f"Debug info - Date column sample: {df[date_column].head()}")
        raise ValueError(f"Error in cleaning datetime column: {str(e)}")

def safe_len(x):
    """
    Safely calculate length of a value, returning 0 for None or float values.
    """
    if pd.isna(x) or isinstance(x, float):
        return 0
    return len(str(x))

def serialize_dataframe_dict(d):
    """
    Recursively serialize all values in a dictionary that came from a DataFrame.
    """
    if isinstance(d, list):
        return [serialize_dataframe_dict(item) if isinstance(item, (dict, list)) else item for item in d]
    elif isinstance(d, dict):
        result = {}
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                result[k] = serialize_dataframe_dict(v)
            elif isinstance(v, (pd.Timestamp, datetime)):
                result[k] = v.isoformat()
            elif isinstance(v, np.integer):
                result[k] = int(v)
            elif isinstance(v, np.floating):
                result[k] = float(v)
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            else:
                result[k] = v
        return result
    return d

def convert_datetime_to_string(data):
    """
    Convert datetime objects in data (if any) to string using .isoformat().
    """
    if isinstance(data, pd.DataFrame):
        # Convert datetime columns in a DataFrame to string
        for col in data.select_dtypes(include=['datetime']):
            data[col] = data[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
        return data

    elif isinstance(data, dict):
        # Recursively convert datetime objects in a dictionary to string
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, dict):
                data[key] = convert_datetime_to_string(value)
            elif isinstance(value, list):
                data[key] = [convert_datetime_to_string(item) if isinstance(item, dict) else item for item in value]
        return data

    return data

def clean_datetime_column(df, column):
    cleaned_dates = []
    for date in df[column]:
        try:
            parsed_date = pd.to_datetime(date, errors='coerce')
            if parsed_date.tz is not None:
                parsed_date = parsed_date.tz_convert(None)
            cleaned_dates.append(parsed_date)
        except Exception as e:
            cleaned_dates.append(None)
    df[column] = pd.Series(cleaned_dates)
    return df

def preprocess_tweet(tweet):
    if isinstance(tweet, str):  # Check if tweet is a string
        # Convert to lowercase
        tweet = tweet.lower()

        # Remove URLs, mentions, hashtags, and numbers
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'@\w+|\#\w+', '', tweet)
        tweet = re.sub(r'\d+', '', tweet)

        # Remove special characters and punctuation
        tweet = re.sub(r'[^\w\s]', '', tweet)

        # Remove extra spaces
        tweet = tweet.strip()

        return tweet
    else:
        return ""  # Return empty string if the tweet is not a valid string
    
    # 4. Labeling Endpoint
def assign_party_label(tweet):
    parties_keywords = {
        'IND/PTI': ['pti', 'imran khan', 'ik', '804', 'imran', 'khan', 'ind', 'qaidi no''tehreek e insaf', 'pakistan tehreek e insaf'],
        'PMLN': ['pmln', 'nawaz sharif', 'maryam nawaz', 'shehbaz sharif', 'muslim league'],
        'PPP': ['ppp', 'bhutto', 'bilawal bhutto', 'zardari', 'peoples party', 'people party', 'jiyala'],
        'Others': ['mqm', 'jui', 'anp', 'ipp', 'mwmp', 'bnp', 'bap', 'np', 'pmap', 'karachi', 'urban sindh']
    }
    
    if isinstance(tweet, str):
        scores = {party: 0 for party in parties_keywords}

        # Score based on keyword presence
        for party, keywords in parties_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', tweet.lower()):
                    scores[party] += 1

        # Determine the party with the highest score
        max_party = max(scores, key=scores.get)
        return max_party if scores[max_party] > 0 else 'No Party'
    return 'No Party'


# 6. Sentiment Analysis Visualization Endpoint
def create_overall_sentiment_distribution(df: pd.DataFrame) -> str:
    """Create overall sentiment distribution plot and return as base64 string."""
    plt.figure(figsize=(8, 5))
    
    # Define a custom color palette for sentiments
    custom_palette = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'blue'
    }
    
    sns.countplot(data=df, x='Vader_Sentiment', palette=custom_palette)
    plt.title('Overall Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Tweet Count')
    plt.tight_layout()
    
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    return img_str


def create_sentiment_distribution_by_party(df: pd.DataFrame) -> str:
    """Create sentiment distribution by party plot with custom colors and return as base64 string."""
    plt.figure(figsize=(12, 6))
    
    # Define custom color mapping for sentiments
    sentiment_colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}
    
    # Create countplot with custom colors for each sentiment
    sns.countplot(data=df, x='Party', hue='Vader_Sentiment', palette=sentiment_colors)
    
    # Set plot title and labels
    plt.title('Sentiment Distribution by Party')
    plt.xlabel('Political Party')
    plt.ylabel('Number of Tweets')
    
    # Adjust legend and x-ticks
    plt.legend(title='Sentiment', loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64 string
    img_str = fig_to_base64(plt.gcf())
    plt.close()  # Close the plot to avoid display in notebook
    return img_str



def create_wordclouds_by_party(df: pd.DataFrame) -> dict:
    """Generate word clouds for each party and return as a dictionary of base64 strings."""
    parties_keywords = {
        'IND/PTI': ['pti', 'imran khan', 'ik', '804', 'imran', 'khan', 'ind', 'qaidi no''tehreek e insaf', 'pakistan tehreek e insaf'],
        'PMLN': ['pmln', 'nawaz sharif', 'maryam nawaz', 'shehbaz sharif', 'muslim league'],
        'PPP': ['ppp', 'bhutto', 'bilawal bhutto', 'zardari', 'peoples party', 'people party', 'jiyala'],
        'Others': ['mqm', 'jui', 'anp', 'ipp', 'mwmp', 'bnp', 'bap', 'np', 'pmap', 'karachi', 'urban sindh']
    }
    wordclouds = {}
    for party, keywords in parties_keywords.items():
        party_tweets = ' '.join(df[df['Party'] == party]['cleaned_tweet'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(party_tweets)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {party}')
        img_str = fig_to_base64(plt.gcf())
        plt.close()
        wordclouds[party] = img_str
    return wordclouds


def create_sentiment_count_by_party(df: pd.DataFrame) -> str:
    """Create stacked bar plot of sentiment count per party and return as base64 string."""
    sentiment_counts = df.groupby(['Party', 'Vader_Sentiment']).size().unstack().fillna(0)
    sentiment_counts.plot(kind='bar', stacked=True, figsize=(12, 6), color=['red', 'gray', 'green'])
    plt.title('Count of Sentiments per Political Party')
    plt.xlabel('Political Party')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    return img_str

def create_tweet_volume_over_time(df: pd.DataFrame) -> str:
    """Create tweet volume over time plot for Jan-Feb 2024 and return as base64 string."""
    tweets_per_day = df.groupby('Date').size().reset_index(name='Tweet Count')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=tweets_per_day, x='Date', y='Tweet Count', marker="o")
    plt.title('Tweet Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.xlim(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-02-29'))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    return img_str

def create_sentiment_trend_over_time(df: pd.DataFrame) -> str:
    """Create sentiment trend over time plot for Jan-Feb 2024 and return as base64 string."""
    tweets_sentiment_per_day = df.groupby(['Date', 'Vader_Sentiment']).size().unstack(fill_value=0)
    tweets_sentiment_per_day.plot(kind='line', marker='o', colormap='viridis', figsize=(12, 6))
    plt.title('Sentiment Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.legend(title="Sentiment")
    plt.xlim(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-02-29'))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    return img_str



def create_interactive_stacked_bar_chart(df: pd.DataFrame) -> str:
    """Create interactive stacked bar chart for sentiment distribution by party and return as base64 string."""
    try:
        # Check if required columns exist in the DataFrame
        if 'Party' not in df.columns or 'Vader_Sentiment' not in df.columns:
            raise ValueError("DataFrame must contain 'Party' and 'Vader_Sentiment' columns.")

        # Create the stacked bar chart
        fig = px.histogram(df, x='Party', color='Vader_Sentiment', barmode='stack',
                           title='Sentiment Distribution by Party',
                           labels={'Vader_Sentiment': 'Sentiment'},
                           category_orders={'Party': sorted(df['Party'].unique())})
        fig.update_layout(xaxis_title='Political Party', yaxis_title='Number of Tweets', legend_title='Sentiment')
        
        # Convert figure to image bytes and encode to base64
        img_bytes = fig.to_image(format="png", engine="kaleido")
        return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    except Exception as e:
        print(f"Error creating stacked bar chart: {e}")
        return ""

def create_sunburst_chart(df: pd.DataFrame) -> str:
    """Create sunburst chart for sentiment distribution by party and return as base64 string."""
    try:
        # Check if required columns exist in the DataFrame
        if 'Party' not in df.columns or 'Vader_Sentiment' not in df.columns:
            raise ValueError("DataFrame must contain 'Party' and 'Vader_Sentiment' columns.")
        
        # Create the sentiment counts for sunburst chart
        sentiment_counts = df.groupby(['Party', 'Vader_Sentiment']).size().reset_index(name='count')

        # Create the sunburst chart
        fig = px.sunburst(sentiment_counts, path=['Party', 'Vader_Sentiment'], values='count',
                          title='Sentiment Distribution by Party')
        
        # Convert figure to image bytes and encode to base64
        img_bytes = fig.to_image(format="png", engine="kaleido")
        return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    except Exception as e:
        print(f"Error creating sunburst chart: {e}")
        return ""
    
    
def create_party_plot(data, title, is_percentage=False):
    """Create a party support visualization."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        sns.barplot(
            x=data.index,
            y=data.values,
            palette='viridis',
            ax=ax
        )
        
        # Customize plot
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Party', fontsize=12)
        ax.set_ylabel('Percentage' if is_percentage else 'Number of Tweets', fontsize=12)
        
        # Rotate x-labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, v in enumerate(data.values):
            ax.text(
                i, 
                v, 
                f'{v:.1f}{"%" if is_percentage else ""}',
                ha='center',
                va='bottom'
            )
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None
# Helper functions
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

def assign_party_label(tweet, parties_keywords):
    if isinstance(tweet, str):
        scores = {party: 0 for party in parties_keywords}
        for party, keywords in parties_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', tweet):
                    scores[party] += 1
        max_party = max(scores, key=scores.get)
        return max_party if scores[max_party] > 0 else 'No Party'
    return 'No Party'

def analyze_sentiment(tweet, analyzer):
    if isinstance(tweet, str):
        sentiment_scores = analyzer.polarity_scores(tweet)
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.3:
            return 'Positive'
        elif compound_score <= -0.3:
            return 'Negative'
        else:
            return 'Neutral'
    return 'Neutral'