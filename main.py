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
from typing import Dict, Any, Union
from pydantic import BaseModel
import os
import joblib
from pydantic import BaseModel
import sklearn
import openai
from logging import Logger
from utils import fig_to_base64,perform_eda_on_dataframe,convert_datetime_to_string,serialize_datetime,clean_datetime_column,safe_len,serialize_dataframe_dict,clean_datetime_column,preprocess_tweet,assign_party_label,create_overall_sentiment_distribution,create_sentiment_distribution_by_party,create_wordclouds_by_party,create_sentiment_count_by_party,create_tweet_volume_over_time,create_sentiment_trend_over_time,create_interactive_stacked_bar_chart,create_sunburst_chart,create_party_plot,create_party_plot,preprocess_tweet,analyze_sentiment
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch





# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Ensure stopwords are downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


app = FastAPI(
    title="Election Data Analysis API",
    description="Comprehensive API for election data analysis and visualization",
    version="1.0.0"
)


app = FastAPI()


# Global sentiment analyzer and lemmatizer
analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()



# Enable CORS for frontend served from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Directory to save the cleaned file
SAVE_DIR = "data"
FIGURE_DPI = 100
os.makedirs(SAVE_DIR, exist_ok=True)  # Create the directory 

# Store uploaded DataFrame globally
uploaded_df = None

# Absolute paths to static and template directories
static_dir = os.path.abspath("static")
templates_dir = os.path.abspath("templates")

# Mount static files
app.mount("/static", StaticFiles(directory="./static"), name="static")

# Initialize Jinja2 template directory
templates = Jinja2Templates(directory='./templates')

# Define the root route to render index.html
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})




@app.post("/load_data")
async def load_data():
    global uploaded_df
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "pre_election_processed_dates.csv")  # Set the path to your Excel file here

    try:
        # Load the Excel data into a DataFrame
        uploaded_df = pd.read_csv(file_path)

        # Convert datetime columns in the DataFrame to string
        uploaded_df = convert_datetime_to_string(uploaded_df)

        # Prepare dataset info for the response with datetime-safe sample data
        sample_data = convert_datetime_to_string(uploaded_df.head().to_dict())

        data_info = {
            "shape": uploaded_df.shape,
            "columns": list(uploaded_df.columns),
            "sample_data": sample_data,
            "data_types": uploaded_df.dtypes.astype(str).to_dict()
        }

        return JSONResponse(content=data_info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")
    










@app.post("/eda")
async def perform_eda():
    global uploaded_df
    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")
    
    try:
        eda_results = perform_eda_on_dataframe(uploaded_df)
        serializable_results = json.loads(json.dumps(eda_results, default=serialize_datetime))
        return JSONResponse(content=serializable_results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 3. Preprocessing Endpoint


@app.post("/preprocess")
async def preprocess_data():
    """
    Preprocess the tweet data from the uploaded file and save the cleaned tweets.
    """
    global uploaded_df
    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")

    try:
        # Initialize the WordNet Lemmatizer
        lemmatizer = WordNetLemmatizer()

        # Preprocess tweets
        uploaded_df['cleaned_tweet'] = uploaded_df['Tweet'].fillna('').astype(str).apply(preprocess_tweet)

        # Prepare sample data (first few cleaned tweets)
        sample_processed = uploaded_df[['Tweet', 'cleaned_tweet']].head(5).to_dict(orient="records")

        # Prepare results for JSON response
        results = {
            "original_count": len(uploaded_df),
            "processed_count": len(uploaded_df),
            "sample_processed": sample_processed,
        }

        # Define the file path and save the cleaned DataFrame to a CSV file
        save_path = os.path.join(SAVE_DIR, "cleaned_tweets.csv")
        uploaded_df.to_csv(save_path, index=False)

        # Return JSON response with a message confirming the saved file
        results["file_saved_path"] = save_path
        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


    


# party labeling function with internal keywords dictionary
def assign_label(tweet):
    parties_keywords = {
        'IND/PTI': ['pti', 'imran khan', 'ik', '804', 'imran', 'khan', 'ind', 'qaidi no', 'tehreek e insaf', 'pakistan tehreek e insaf'],
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

@app.post("/label_tweets")
async def label_tweets() -> JSONResponse:
    """
    Label tweets with party affiliations using the preprocessed file.
    """
    try:
        # Log start of process
        logger.debug("Starting tweet labeling process...")
        
        # Load the preprocessed file from the specified directory
        preprocessed_file_path = os.path.join(SAVE_DIR, "cleaned_tweets.csv")
        if not os.path.exists(preprocessed_file_path):
            raise FileNotFoundError(f"File not found at path: {preprocessed_file_path}")
        
        logger.debug("File found, loading data...")
        df = pd.read_csv(preprocessed_file_path)
        
        # Check if 'cleaned_tweet' column exists
        if 'cleaned_tweet' not in df.columns:
            raise KeyError("Column 'cleaned_tweet' not found in the CSV file.")
        
        # Apply the labeling function
        logger.debug("Applying party labeling function...")
        df['Party'] = df['cleaned_tweet'].apply(assign_label)
        
        # Filter out "No Party" tweets
        logger.debug("Filtering 'No Party' tweets...")
        df = df[df['Party'] != 'No Party']
        
        # Calculate party distribution
        logger.debug("Calculating party distribution...")
        party_distribution = df['Party'].value_counts().to_dict()
        
        # Prepare JSON results
        results = {
            "message": "Tweets labeled successfully.",
            "party_distribution": party_distribution,
            "sample_labeled": df[['cleaned_tweet', 'Party']].head(5).to_dict(orient='records')  # Return top 5 for preview
        }
        
        # Define the file path and save the labeled DataFrame to a CSV file
        save_path = os.path.join(SAVE_DIR, "labeled_tweets.csv")
        df.to_csv(save_path, index=False)
        
        # Add the file path to results
        results["file_saved_path"] = save_path
        
        # Return JSON response with results and download path
        logger.debug("Labeling complete, returning JSON response.")
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")  # Log the exact error for debugging
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")



logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Logging Configuration
logger = logging.getLogger("sentiment_logger")
logger.setLevel(logging.DEBUG)

#stream handler to print to the console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

# formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Ensure Uvicorn handlers are removed 
uvicorn_logger = logging.getLogger("uvicorn")
for handler in uvicorn_logger.handlers:
    uvicorn_logger.removeHandler(handler)

uvicorn_logger.addHandler(stream_handler)

@app.post("/annotate_sentiment")
async def annotate_sentiment():
    """
    Annotate tweets with VADER sentiment analysis using the labeled tweets file
    """
    try:
        logger.info("Starting sentiment annotation process...")

        labeled_file_path = os.path.join(SAVE_DIR, "labeled_tweets.csv")
        logger.debug(f"Looking for file at path: {labeled_file_path}")
        
        if not os.path.exists(labeled_file_path):
            logger.error(f"File not found: {labeled_file_path}")
            raise FileNotFoundError(f"File not found: {labeled_file_path}")

        # Load file
        df = pd.read_csv(labeled_file_path)
        logger.info("Loaded labeled tweets file successfully.")

        # Initialize VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        logger.debug("Initialized VADER sentiment analyzer.")

        def analyze_sentiment(tweet):
            if isinstance(tweet, str):
                sentiment_scores = analyzer.polarity_scores(tweet)
                compound_score = sentiment_scores['compound']
                logger.debug(f"Analyzed tweet '{tweet}': compound score = {compound_score}")
                if compound_score >= 0.2:
                    return 'Positive'
                elif compound_score <= -0.2:
                    return 'Negative'
                else:
                    return 'Neutral'
            return 'Neutral'

        # Apply sentiment analysis
        df['Vader_Sentiment'] = df['cleaned_tweet'].apply(analyze_sentiment)
        logger.info("Applied VADER sentiment analysis to tweets.")

        # Compute sentiment distribution
        sentiment_distribution = df['Vader_Sentiment'].value_counts().to_dict()
        logger.debug(f"Sentiment distribution: {sentiment_distribution}")

        output_file_path = os.path.join(SAVE_DIR, "labeled_tweets_with_sentiments.csv")
        logger.debug(f"Saving annotated data to {output_file_path}")

        # Save as CSV
        df.to_csv(output_file_path, index=False)


        # Select a sample of annotated tweets for display
        sample_annotated = df[['cleaned_tweet', 'Vader_Sentiment']].head(5).to_dict(orient="records")
        logger.info("Sample of annotated tweets prepared for response.")

        return {
            "message": "Sentiment annotation completed successfully.",
            "sentiment_distribution": sentiment_distribution,
            "sample_annotated": sample_annotated,
            "file_path": output_file_path
        }

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error annotating sentiment: {str(e)}")
        raise HTTPException(status_code=400, detail="An error occurred while annotating sentiment. Please try again.")



#sentiment analysis endpoint
@app.post("/sentiment_analysis")
async def analyze_sentiments() -> Dict[str, Any]:
    try:
        # Load and validate data
        labeled_file_path = os.path.join(SAVE_DIR, "labeled_tweets_with_sentiments.csv")
        if not os.path.exists(labeled_file_path):
            raise FileNotFoundError(f"Labeled tweets file not found at: {labeled_file_path}")

        df = pd.read_csv(labeled_file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # df = df.dropna(subset=['Date'])
        df['Date'] = df['Date'].dt.date

        # Generate the plots
        results = {
            "overall_sentiment_distribution": create_overall_sentiment_distribution(df),
            "sentiment_distribution_by_party": create_sentiment_distribution_by_party(df),
            "party_wordclouds": create_wordclouds_by_party(df),
            "sentiment_count_by_party": create_sentiment_count_by_party(df),
            "tweet_volume_over_time": create_tweet_volume_over_time(df),
            "sentiment_trend_over_time": create_sentiment_trend_over_time(df),
            "interactive_stacked_bar_chart": create_interactive_stacked_bar_chart(df),
            "sunburst_chart": create_sunburst_chart(df)
        }

        return JSONResponse(content=results)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")





#party support count 
@app.post("/party_support")
async def calculate_party_support():
    """Calculate party support statistics from annotated tweets."""
    try:
        # Load data
        labeled_file_path = os.path.join(SAVE_DIR, "labeled_tweets_with_sentiments.csv")
        if not os.path.exists(labeled_file_path):
            raise FileNotFoundError(f"File not found: {labeled_file_path}")
        
        df = pd.read_csv(labeled_file_path)
        
        # Validate data
        if 'Party' not in df.columns:
            raise ValueError("Required column 'Party' not found in dataset")
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Calculate statistics
        party_support_count = df['Party'].value_counts()
        party_support_percentage = df['Party'].value_counts(normalize=True) * 100
        
        # Create visualizations
        count_plot = create_party_plot(
            party_support_count,
            'Number of Tweets Supporting Each Party'
        )
        
        percentage_plot = create_party_plot(
            party_support_percentage,
            'Percentage of Tweets Supporting Each Party',
            is_percentage=True
        )
        
        # Convert plots to base64
        count_plot_base64 = fig_to_base64(count_plot) if count_plot else None
        percentage_plot_base64 = fig_to_base64(percentage_plot) if percentage_plot else None
        
        # Prepare response
        results = {
            "party_counts": party_support_count.to_dict(),
            "party_percentages": party_support_percentage.round(2).to_dict(),
            "support_plot": f"data:image/png;base64,{count_plot_base64}" if count_plot_base64 else None,
            "support_percentage_plot": f"data:image/png;base64,{percentage_plot_base64}" if percentage_plot_base64 else None
        }
        
        return JSONResponse(content=results)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")




# 8. Sentiment vs Actual Results Endpoint


@app.post("/sentiment_vs_actual")
async def compare_sentiment_actual():
    """
    Compare sentiment analysis results with actual election results.
    """
    try:
        # Load tweet data from the annotated file
        tweet_file_path = os.path.join(SAVE_DIR, "labeled_tweets_with_sentiments.csv")
        #S3
        seat_file_path = os.path.join(SAVE_DIR, "party_seat_counts.csv")

        # Check if files exist
        if not os.path.exists(tweet_file_path):
            raise FileNotFoundError(f"Tweet file not found: {tweet_file_path}")
        if not os.path.exists(seat_file_path):
            raise FileNotFoundError(f"Seat file not found: {seat_file_path}")

        # Load tweet data and seat data
        df = pd.read_csv(tweet_file_path)
        seat_data = pd.read_csv(seat_file_path)

        # Standardize party labels in both tweet and seat data
      
        party_mapping = {
            'IND/PTI': 'IND/PTI', 'PML(N)': 'PMLN', 'PPP': 'PPP', 'ppp': 'PPP',
            'MQM': 'Others', 'JUI': 'Others', 'PML': 'Others', 'IPP': 'Others',
            'MWMP': 'Others', 'BNP': 'Others', 'PML (Z)': 'Others', 'PNAPF': 'Others',
            'BAP': 'Others', 'NP': 'Others', 'PMAP': 'Others'
        }

        # Apply the standardized party labels to both seat_data and df
        seat_data['Party'] = seat_data['Party'].replace(party_mapping)
        df['Party'] = df['Party'].replace(party_mapping)


        # Group tweet counts and seat counts by Party
        tweet_counts = df['Party'].value_counts().reset_index()
        tweet_counts.columns = ['Party', 'Tweet_Count']
        seat_counts = seat_data.groupby('Party')['Seats'].sum().reset_index()

        # Merge tweet counts and seat counts
        combined_data = pd.merge(tweet_counts, seat_counts, on='Party', how='outer').fillna(0)

        # Plot comparison of tweet counts and actual seats won
        fig, ax1 = plt.subplots(figsize=(12, 6))
        bar_width = 0.35
        positions = range(len(combined_data))

        ax1.bar([p - bar_width / 2 for p in positions], combined_data['Tweet_Count'],
                color='skyblue', width=bar_width, label='Tweet Count')
        ax1.set_xlabel('Party')
        ax1.set_ylabel('Tweet Count', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax1.set_xticks(positions)
        ax1.set_xticklabels(combined_data['Party'])

        ax2 = ax1.twinx()
        ax2.bar([p + bar_width / 2 for p in positions], combined_data['Seats'],
                color='lightgreen', width=bar_width, label='Seats Won')
        ax2.set_ylabel('Seats Won', color='lightgreen')
        ax2.tick_params(axis='y', labelcolor='lightgreen')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.set_title('Comparison of Tweet Support and Actual Seats Won by Parties')
        fig.tight_layout()
        comparison_plot = fig_to_base64(fig)
        plt.close()

        # Calculate percentages and differences
        total_tweets = combined_data['Tweet_Count'].sum()
        total_seats = combined_data['Seats'].sum()
        combined_data['Tweet_Percentage'] = (combined_data['Tweet_Count'] / total_tweets) * 100
        combined_data['Seat_Percentage'] = (combined_data['Seats'] / total_seats) * 100
        combined_data['Numerical_Difference'] = combined_data['Seats'] - combined_data['Tweet_Count']
        combined_data['Percentage_Difference'] = combined_data['Seat_Percentage'] - combined_data['Tweet_Percentage']
        combined_data_sorted = combined_data.sort_values(by='Numerical_Difference', ascending=False)

        # Plot side-by-side comparison of percentages
        plt.figure(figsize=(14, 7))
        ax1 = plt.subplot(1, 2, 1)
        ax1.bar([p - bar_width / 2 for p in positions], combined_data['Tweet_Percentage'],
                color='skyblue', width=bar_width, label='Tweet Count (%)')
        ax1.bar([p + bar_width / 2 for p in positions], combined_data['Seat_Percentage'],
                color='lightgreen', width=bar_width, label='Seats Won (%)')
        ax1.set_xticks(positions)
        ax1.set_xticklabels(combined_data['Party'], rotation=45, ha='right')
        ax1.set_xlabel('Party')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Tweet Support vs Seats Won by Party (Percentage)')
        ax1.legend()

        ax2 = plt.subplot(1, 2, 2)
        sns.barplot(x='Party', y='Percentage_Difference', data=combined_data, palette='coolwarm', ax=ax2)
        ax2.set_title('Percentage Difference between Tweet Support and Seats Won')
        ax2.set_xlabel('Party')
        ax2.set_ylabel('Percentage Difference (%)')

        for i, v in enumerate(combined_data['Percentage_Difference']):
            if v > 0:
                ax2.text(i, v + 0.5, f"+{v:.2f}%", color='green', ha='center')
            else:
                ax2.text(i, v - 1.5, f"{v:.2f}%", color='red', ha='center')

        plt.tight_layout()
        percentage_difference_plot = fig_to_base64(plt.gcf())
        plt.close()

        # Prepare the response data
        results = {
            "tweet_counts": tweet_counts.to_dict(orient='records'),
            "seat_counts": seat_counts.to_dict(orient='records'),
            "numerical_differences": combined_data_sorted[['Party', 'Tweet_Count', 'Seats', 'Numerical_Difference']].to_dict(orient='records'),
            "comparison_plot": comparison_plot,
            "percentage_difference_plot": percentage_difference_plot
        }

        return JSONResponse(results)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    



#post election analysis
@app.post("/post-election-analysis/")
async def post_election_analysis():
    try:
        # Define file paths
        #S3
        tweet_file_path = os.path.join(SAVE_DIR, "post election dataset with processed_dates.csv")

        # Check if the file exists
        if not os.path.exists(tweet_file_path):
            raise FileNotFoundError(f"Tweet file not found: {tweet_file_path}")

        # Load and preprocess the data
        df = pd.read_csv(tweet_file_path)

        # Preprocessing function
        def preprocess_tweet(tweet):
            tweet = tweet.lower()
            tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet)
            tweet = re.sub(r'@\w+|\#\w+', '', tweet)
            tweet = re.sub(r'\d+', '', tweet)
            tweet = re.sub(r'[^\w\s]', '', tweet)
            return tweet.strip()

        df['cleaned_tweet'] = df['Tweet'].astype(str).apply(preprocess_tweet)
      

        # Party labeling
        parties_keywords = {
            'IND/PTI': ['pti', 'imran khan', 'ik', '804', 'imran', 'khan', 'ind', 'qaidi no', 'tehreek e insaf', 'pakistan tehreek e insaf'],
            'PMLN': ['pmln', 'nawaz sharif', 'maryam nawaz', 'shehbaz sharif', 'muslim league'],
            'PPP': ['ppp', 'bhutto', 'bilawal bhutto', 'zardari', 'peoples party', 'people party', 'jiyala'],
            'Others': ['mqm', 'jui', 'anp', 'ipp', 'mwmp', 'bnp', 'bap', 'np', 'pmap', 'karachi', 'urban sindh']
        }

        def assign_party_label(tweet):
            scores = {party: 0 for party in parties_keywords}
            for party, keywords in parties_keywords.items():
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', tweet.lower()):
                        scores[party] += 1
            return max(scores, key=scores.get) if max(scores.values()) > 0 else 'No Party'

        df['Party'] = df['cleaned_tweet'].apply(assign_party_label)
        df = df[df['Party'] != 'No Party']

        # Sentiment analysis
        def analyze_sentiment(tweet):
            sentiment_scores = analyzer.polarity_scores(tweet)
            compound_score = sentiment_scores['compound']
            return 'Positive' if compound_score >= 0.2 else 'Negative' if compound_score <= -0.2 else 'Neutral'

        df['Vader_Sentiment'] = df['cleaned_tweet'].apply(analyze_sentiment)

        # Visualization functions
        def generate_plots(df):
            figures = {}

            # Overall Sentiment Distribution
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='Vader_Sentiment', palette="viridis", order=['Positive', 'Neutral', 'Negative'])
            plt.title('Overall Sentiment Distribution')
            figures['overall_sentiment'] = fig_to_base64(plt.gcf())
            plt.close()

            # Sentiment Distribution by Party
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df, x='Party', hue='Vader_Sentiment', palette='Set1', hue_order=['Positive', 'Neutral', 'Negative'])
            plt.title('Sentiment Distribution by Party')
            plt.xticks(rotation=45)
            plt.tight_layout()
            figures['sentiment_by_party'] = fig_to_base64(plt.gcf())
            plt.close()

            # Word Clouds for Each Party
            for party in parties_keywords.keys():
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['Party'] == party]['cleaned_tweet']))
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud for {party}')
                figures[f'wordcloud_{party}'] = fig_to_base64(plt.gcf())
                plt.close()

            # Stacked Bar Plot for Sentiment per Party
            sentiment_counts = df.groupby(['Party', 'Vader_Sentiment']).size().unstack().fillna(0)[['Positive', 'Neutral', 'Negative']]
            sentiment_counts.plot(kind='bar', stacked=True, figsize=(12, 6), color=['green', 'gray', 'red'])
            plt.title('Count of Sentiments per Political Party')
            plt.xticks(rotation=45)
            plt.tight_layout()
            figures['stacked_bar_sentiment'] = fig_to_base64(plt.gcf())
            plt.close()

            # Tweet Volume Over Time
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
            tweets_per_day = df.groupby('Date').size().reset_index(name='Tweet Count')
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=tweets_per_day, x='Date', y='Tweet Count', marker="o")
            plt.title('Tweet Volume Over Time')
            figures['tweet_volume_time'] = fig_to_base64(plt.gcf())
            plt.close()

            return figures

        plots = generate_plots(df)

        # Save results
        df.to_csv('./data/post_election_sentiment_analysis.csv', index=False)

        return JSONResponse(content={"plots": plots})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






#pre post comparison

@app.post("/compare-pre-post-election")
async def compare_pre_post_election():
    pre_file = os.path.join(SAVE_DIR, "labeled_tweets_with_sentiments.csv")
    post_file = os.path.join(SAVE_DIR, "post_election_sentiment_analysis.csv")
    
    try:
        # Load datasets
        pre_df = pd.read_csv(pre_file)
        post_df = pd.read_csv(post_file)

        # Ensure Date is parsed correctly
        pre_df['Date'] = pd.to_datetime(pre_df['Date'], errors='coerce')
        post_df['Date'] = pd.to_datetime(post_df['Date'], errors='coerce')

        # Drop rows with invalid dates
        #pre_df = pre_df.dropna(subset=['Date'])
        #post_df = post_df.dropna(subset=['Date'])

        # Ensure date only
        pre_df['Date'] = pre_df['Date'].dt.date
        post_df['Date'] = post_df['Date'].dt.date

        # Define sentiment order
        sentiment_order = ['Positive', 'Neutral', 'Negative']

        # Prepare base64 image dictionary
        base64_images = {}

        # 1. Overall Sentiment Distribution Comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        sns.countplot(data=pre_df, x='Vader_Sentiment', palette="viridis", order=sentiment_order, ax=axes[0])
        axes[0].set_title('Pre-Election Sentiment Distribution')
        sns.countplot(data=post_df, x='Vader_Sentiment', palette="viridis", order=sentiment_order, ax=axes[1])
        axes[1].set_title('Post-Election Sentiment Distribution')
        for ax in axes:
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Tweet Count')
        plt.tight_layout()
        base64_images["overall_sentiment"] = fig_to_base64(fig)
        plt.close(fig)

        # 2. Sentiment Distribution by Party Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        sns.countplot(data=pre_df, x='Party', hue='Vader_Sentiment', palette='Set1', hue_order=sentiment_order, ax=axes[0])
        axes[0].set_title('Pre-Election Sentiment by Party')
        sns.countplot(data=post_df, x='Party', hue='Vader_Sentiment', palette='Set1', hue_order=sentiment_order, ax=axes[1])
        axes[1].set_title('Post-Election Sentiment by Party')
        for ax in axes:
            ax.set_xlabel('Political Party')
            ax.set_ylabel('Number of Tweets')
            ax.legend(title='Sentiment', loc='upper right')
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        base64_images["sentiment_by_party"] = fig_to_base64(fig)
        plt.close(fig)

        # 3. Sentiment Trend Over Time Comparison
        pre_trend = pre_df.groupby(['Date', 'Vader_Sentiment']).size().unstack(fill_value=0)[sentiment_order]
        post_trend = post_df.groupby(['Date', 'Vader_Sentiment']).size().unstack(fill_value=0)[sentiment_order]

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=pre_trend, dashes=False, palette='viridis', markers=True)
        plt.title('Pre-Election Sentiment Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Tweets')
        plt.legend(title='Sentiment')
        base64_images["pre_trend"] = fig_to_base64(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=post_trend, dashes=False, palette='viridis', markers=True)
        plt.title('Post-Election Sentiment Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Tweets')
        plt.legend(title='Sentiment')
        base64_images["post_trend"] = fig_to_base64(fig)
        plt.close(fig)

        # 4. Interactive Sunburst Comparison
        pre_counts = pre_df.groupby(['Party', 'Vader_Sentiment']).size().reset_index(name='count')
        post_counts = post_df.groupby(['Party', 'Vader_Sentiment']).size().reset_index(name='count')
        
        pre_fig = px.sunburst(pre_counts, path=['Party', 'Vader_Sentiment'], values='count',
                              title='Pre-Election Sentiment Distribution by Party',
                              color='Vader_Sentiment', color_discrete_sequence=px.colors.qualitative.Vivid)
        post_fig = px.sunburst(post_counts, path=['Party', 'Vader_Sentiment'], values='count',
                               title='Post-Election Sentiment Distribution by Party',
                               color='Vader_Sentiment', color_discrete_sequence=px.colors.qualitative.Vivid)
        
        base64_images["pre_sunburst"] = pre_fig.to_json()
        base64_images["post_sunburst"] = post_fig.to_json()
        plt.close(fig)

        return base64_images

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generating comparison: {str(e)}")


#constituency level analysis
@app.post("/constituency-level-analysis")
async def constituency_level_analysis():
    try:
        #S3
        pre_file = os.path.join(SAVE_DIR, "Form 47.xlsx")
        post_file = os.path.join(SAVE_DIR, "Constituency tweets.xlsx")

        # Initialize VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()

        # Define party keywords and phrases for exact matching
        parties_keywords = {
            'IND/PTI': ['pti', 'imran khan', 'ik', '804', 'imran', 'khan', 'ind', 'qaidi no', 'tehreek e insaf', 'pakistan tehreek e insaf'],
            'PMLN': ['pmln', 'nawaz sharif', 'maryam nawaz', 'shehbaz sharif', 'muslim league'],
            'PPP': ['ppp', 'bhutto', 'bilawal bhutto', 'zardari', 'peoples party', 'people party', 'jiyala'],
            'Others': ['mqm', 'jui', 'anp', 'ipp', 'mwmp', 'bnp', 'bap', 'np', 'pmap', 'karachi', 'urban sindh']
        }

        # Load and preprocess voting data
        sheets = pd.ExcelFile(pre_file).sheet_names
        combined_df = pd.DataFrame()

        for sheet in sheets:
            sheet_df = pd.read_excel(pre_file, sheet_name=sheet)
            sheet_df['Constituency'] = sheet  # Add constituency name from the sheet name
            combined_df = pd.concat([combined_df, sheet_df], ignore_index=True)

        combined_df.columns = combined_df.columns.str.strip()
        combined_df.dropna(subset=['Candidate Name', 'Party', 'Total Votes'], inplace=True)
        combined_df['Total Votes'] = pd.to_numeric(combined_df['Total Votes'], errors='coerce')

        # Load and preprocess tweet data
        tweets_sheets = pd.ExcelFile(post_file).sheet_names
        combined_tweets_df = pd.DataFrame()

        for sheet in tweets_sheets:
            sheet_df = pd.read_excel(post_file, sheet_name=sheet)
            sheet_df['Constituency'] = sheet  # Add constituency name from the sheet name
            combined_tweets_df = pd.concat([combined_tweets_df, sheet_df], ignore_index=True)

        combined_tweets_df.columns = combined_tweets_df.columns.str.strip()
        combined_tweets_df.dropna(subset=['Tweet'], inplace=True)
        combined_tweets_df['cleaned_tweet'] = combined_tweets_df['Tweet'].apply(preprocess_tweet)

        # Apply party label and sentiment analysis functions
        combined_tweets_df['Party'] = combined_tweets_df['cleaned_tweet'].apply(lambda tweet: assign_party_label(tweet, parties_keywords))
        combined_tweets_df = combined_tweets_df[combined_tweets_df['Party'] != 'No Party']
        combined_tweets_df['Vader_Sentiment'] = combined_tweets_df['cleaned_tweet'].apply(lambda tweet: analyze_sentiment(tweet, analyzer))

        # Generate and save plots
        plot_images = []

        constituencies = combined_df['Constituency'].unique()
        sentiment_order = ['Positive', 'Neutral', 'Negative']

        for constituency in constituencies:
            # Plot vote distribution
            plt.figure(figsize=(10, 6))
            constituency_df = combined_df[combined_df['Constituency'] == constituency]
            plt.barh(constituency_df['Party'], constituency_df['Total Votes'], color='skyblue')
            plt.title(f'Vote Distribution in {constituency}')
            plt.xlabel('Total Votes')
            plt.ylabel('Party')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Convert plot to base64
            vote_dist_base64 = fig_to_base64(plt.gcf())
            plot_images.append({'constituency': constituency, 'type': 'vote_distribution', 'image': vote_dist_base64})
            plt.close()

            # Plot sentiment distribution
            constituency_tweets_df = combined_tweets_df[combined_tweets_df['Constituency'] == constituency]
            sentiment_summary = constituency_tweets_df.groupby(['Party', 'Vader_Sentiment']).size().unstack(fill_value=0)
            sentiment_summary = sentiment_summary.reindex(sentiment_order, axis=1, fill_value=0)

            plt.figure(figsize=(12, 8))
            sentiment_summary.plot(kind='bar', stacked=True, color=['lightgreen', 'lightgrey', 'lightcoral'])
            plt.title(f'Sentiment Distribution by Party in {constituency}')
            plt.xlabel('Party')
            plt.ylabel('Number of Tweets')
            plt.xticks(rotation=0)
            plt.legend(title='Sentiment', loc='upper right')
            plt.tight_layout()

            # Convert plot to base64
            sentiment_dist_base64 = fig_to_base64(plt.gcf())
            plot_images.append({'constituency': constituency, 'type': 'sentiment_distribution', 'image': sentiment_dist_base64})
            plt.close()

            # Comparison of vote distribution and sentiment
            fig, ax1 = plt.subplots(figsize=(12, 8))
            ax2 = ax1.twinx()

            constituency_votes = combined_df[combined_df['Constituency'] == constituency].groupby('Party')['Total Votes'].sum()
            sentiment_count = combined_tweets_df[combined_tweets_df['Constituency'] == constituency].groupby(['Party', 'Vader_Sentiment']).size().unstack(fill_value=0)
            sentiment_summary = sentiment_count.reindex(sentiment_order, axis=1, fill_value=0)
            sentiment_summary['Total Tweets'] = sentiment_summary.sum(axis=1)

            # Align total votes and sentiment counts by Party index
            constituency_votes = constituency_votes.reindex(sentiment_summary.index).fillna(0)

            constituency_votes.plot(kind='bar', color='skyblue', alpha=0.6, ax=ax1, position=0, width=0.4, label='Total Votes')
            ax1.set_xlabel('Party')
            ax1.set_ylabel('Total Votes', color='skyblue')
            ax1.tick_params(axis='y', labelcolor='skyblue')
            ax1.set_title(f'Comparison of Vote Distribution and Sentiment for {constituency}')

            sentiment_summary[['Positive', 'Neutral', 'Negative']].plot(kind='bar', stacked=True, ax=ax2, position=1, alpha=0.6, width=0.4, color=['lightgreen', 'lightgrey', 'lightcoral'])
            ax2.set_ylabel('Number of Tweets', color='lightcoral')
            ax2.tick_params(axis='y', labelcolor='lightcoral')

            ax1.legend(loc='upper left')
            ax2.legend(title='Sentiment', loc='upper right')
            plt.tight_layout()

            # Convert comparison plot to base64
            comparison_base64 = fig_to_base64(fig)
            plot_images.append({'constituency': constituency, 'type': 'comparison', 'image': comparison_base64})
            plt.close()

        return JSONResponse(content={"message": "Analysis complete", "plots": plot_images})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Load party model
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the dynamic paths
model_path = os.path.join(current_dir, "models", "party_model.pkl")
vectorizer_path = os.path.join(current_dir, "models", "vectorizer.pkl")

# Load the models using the dynamic paths
party_model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Load Cardiff NLP sentiment model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Define input data structure
class TextInput(BaseModel):
    text: str
# Define a party prediction function
def predict_party(text):
    preprocessed_text = vectorizer.transform([text])
    party_pred = party_model.predict(preprocessed_text)
    return party_pred[0]

# Define a sentiment prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_labels[sentiment]

@app.post("/predict")
async def predict(input: TextInput):
    try:
        party = predict_party(input.text)
        sentiment = predict_sentiment(input.text)
        return {"party": party, "sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)