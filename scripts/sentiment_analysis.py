# src/sentiment_analysis.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Tuple, List, Dict

# Ensure you have the required nltk resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')

class SentimentAnalyzer:
    
    @staticmethod
    def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
        """
            Analyze sentiment of headlines using VADER and include stock column.

        Parameters:
        - df (pd.DataFrame): DataFrame containing 'headline' and 'stock' columns.

        Returns:
        - pd.DataFrame: DataFrame with headlines, stock, and their sentiment scores.
        """
        df['headline']= df['headline'].astype(str).replace('nan', '')
        sia = SentimentIntensityAnalyzer()
        sentiments = df['headline'].apply(lambda x: sia.polarity_scores(x))
        sentiment_df = pd.DataFrame(sentiments.tolist())
        sentiment_df = pd.concat([df[['headline', 'stock']], sentiment_df], axis=1)
        return sentiment_df

    @staticmethod
    def categorize_sentiment(compound_score: float) -> str:
        """
        Categorize sentiment based on the compound score.

        Parameters:
        - compound_score (float): The compound score from VADER.

        Returns:
        - str: 'Positive', 'Neutral', or 'Negative'.
        """
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    @staticmethod
    def apply_sentiment_categories(data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sentiment categories to the DataFrame.

        Parameters:
        - data (pd.DataFrame): DataFrame with sentiment analysis.

        Returns:
        - pd.DataFrame: DataFrame with an additional 'Sentiment' column.
        """
        data['Sentiment'] = data['compound'].apply(SentimentAnalyzer.categorize_sentiment)
        return data

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess the input text by removing punctuation, stopwords, and non-alphabetic characters.
        
        Parameters:
        - text (str): The input text to preprocess.
        
        Returns:
        - str: The cleaned and preprocessed text.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^a-z\s]', '', text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

    @staticmethod
    def get_common_keywords(headlines: pd.Series, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Identify the most common keywords in the headlines.
        
        Parameters:
        - headlines (pd.Series): The series of headlines to analyze.
        - top_n (int): The number of top keywords to return.
        
        Returns:
        - List[Tuple[str, int]]: A list of tuples with the top keywords and their counts.
        """
        cleaned_headlines = headlines.apply(SentimentAnalyzer.preprocess_text)
        all_words = ' '.join(cleaned_headlines).split()
        word_freq = Counter(all_words)
        return word_freq.most_common(top_n)

    @staticmethod
    def plot_wordcloud(word_freq: Counter) -> None:
        """
        Plot a word cloud of the most common words.
        
        Parameters:
        - word_freq (Counter): A counter object containing word frequencies.
        """
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Headlines')
        plt.show()

    @staticmethod
    def perform_nlp_analysis(headlines: pd.Series) -> None:
        """
        Perform NLP analysis on the headlines, including keyword extraction and word cloud visualization.
        
        Parameters:
        - headlines (pd.Series): The series of headlines to analyze.
        """
        common_keywords = SentimentAnalyzer.get_common_keywords(headlines)
        print("Most common keywords:")
        for word, freq in common_keywords:
            print(f'{word}: {freq}')
        
        word_freq = Counter(dict(common_keywords))
        SentimentAnalyzer.plot_wordcloud(word_freq)


    def calculate_sentiment(df):
        """
        Calculate daily average sentiment scores.

        Parameters:
        - df (pd.DataFrame): DataFrame with sentiment columns and multi-index (Date, stock).

        Returns:
        - pd.DataFrame: DataFrame with daily average sentiment scores for each stock.
        """
        # Define sentiment columns
        sentiment_cols = ['neg', 'neu', 'pos']
        
        # Group by Date and stock, then compute mean of sentiment columns
        daily_sentiment = df.groupby(level=['Date', 'stock'])[sentiment_cols].mean().reset_index()
        
        return daily_sentiment

    def plot_sentiment(daily_sentiment, stock):
        """
        Plot sentiment scores for a given stock.

        Parameters:
        - daily_sentiment (pd.DataFrame): DataFrame with daily sentiment scores.
        - stock (str): The stock symbol for which to plot sentiment scores.

        Returns:
        - plt.Figure: Matplotlib figure object.
        """
        # Filter data for the selected stock
        stock_data = daily_sentiment[daily_sentiment['stock'] == stock]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot sentiment scores
        ax.plot(stock_data['Date'], stock_data['neg'], label='Negative Sentiment', color='red')
        ax.plot(stock_data['Date'], stock_data['neu'], label='Neutral Sentiment', color='grey')
        ax.plot(stock_data['Date'], stock_data['pos'], label='Positive Sentiment', color='green')
        # ax.plot(stock_data['Date'], stock_data['compound'], label='Compound Sentiment', color='blue')
        
        # Set plot title and labels
        ax.set_title(f'Daily Sentiment Scores for {stock}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig