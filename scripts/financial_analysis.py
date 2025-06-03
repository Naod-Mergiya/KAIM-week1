
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os


class FinancialAnalysis:
    def __init__(self, df=None):
        self.df = df

    def load_data(self, filepath='../data/raw_analyst_ratings/raw_analyst_ratings.csv'):
        """Load data from CSV file into dataframe"""
        self.df = pd.read_csv(filepath)
        return self

    def explore_data(self):
        """Perform initial exploratory analysis focusing on key data quality aspects"""
        # Basic dataset info
        print("\nDataset Overview:")
        print("-" * 30)
        print(f"Number of records: {len(self.df)}")
        print(f"Columns: {', '.join(self.df.columns)}")

        # Missing values check
        print("\nMissing Values:")
        print("-" * 30)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])

        # Duplicates check
        print("\nDuplicates:")
        print("-" * 30)
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates} ({(duplicates/len(self.df))*100:.1f}%)")

        # Check duplicates excluding Unnamed: 0 column

        duplicates_excl = self.df.drop('Unnamed: 0', axis=1).duplicated().sum()
        print(f"\nDuplicates (excluding Unnamed: 0): {duplicates_excl} ({(duplicates_excl/len(self.df))*100:.1f}%)")

        # Basic statistics
        print("\nNumerical Statistics:")
        print("-" * 30)
        print(self.df.describe())

        # Categorical columns summary
        print("\nCategorical Columns:")
        print("-" * 30)
        for col in self.df.select_dtypes(include=['object']).columns:
            print(f"\n{col}:")
            print(f"Unique values: {self.df[col].nunique()}")
            print("Top 3 most common:")
            print(self.df[col].value_counts().head(3))

        return self

    def clean_data(self):
        """Clean and preprocess the dataset"""
        # Handle missing values first
        self.df = self.df.dropna(subset=['headline', 'stock', 'publisher'])  # Drop rows missing critical info

        # Remove duplicate rows while keeping first occurrence
        self.df = self.df.drop_duplicates(subset=['headline', 'stock', 'date'], keep='first')

        # Drop unnecessary columns
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)

        # Convert and standardize date format
        self.df['date'] = pd.to_datetime(self.df['date'], format='ISO8601')
        self.df = self.df.dropna(subset=['date'])  # Drop rows with invalid dates

        # Clean text fields
        self.df['headline'] = self.df['headline'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        self.df['headline'] = self.df['headline'].str.replace(r'\s+', ' ')  # Remove extra whitespace

        self.df['publisher'] = (self.df['publisher'].str.strip()
                                                   .str.title()
                                                   .str.replace(r'\s+', ' '))

        self.df['stock'] = (self.df['stock'].str.strip()
                                           .str.upper()
                                           .str.replace(r'[^\w\s]', ''))  # Remove special chars

        self.df['url'] = (self.df['url'].str.strip()
                                       .str.lower())  # Standardize URLs to lowercase

        # Remove rows with empty strings after cleaning
        self.df = self.df.replace('', pd.NA).dropna()

        # Sort chronologically and reset index
        self.df = self.df.sort_values('date').reset_index(drop=True)

        # Add derived features
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['headline_length'] = self.df['headline'].str.len()
        self.df['hour'] = self.df['date'].dt.hour  # Capture time of day

        return self
    def exploratory_analysis(self):
        """Perform initial exploratory analysis after data cleaning"""
        print("\n=== Exploratory Analysis ===")

        # Basic dataset info
        print("\nDataset Shape:", self.df.shape)
        print(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}")

        # Stock coverage
        print("\nTop 10 Most Covered Stocks:")
        print(self.df['stock'].value_counts().head(10))

        # Publishing patterns
        print("\nArticles by Hour of Day:")
        hour_dist = self.df['hour'].value_counts().sort_index()
        print(hour_dist)

        # Content analysis
        print("\nHeadline Length Summary:")
        print(self.df['headline_length'].describe())

        # Publisher analysis
        print("\nNumber of Unique Publishers:", self.df['publisher'].nunique())
        print("\nTop 5 Publishers by Article Count:")
        print(self.df['publisher'].value_counts().head())

        # Temporal patterns
        print("\nArticles by Day of Week:")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print(self.df['day_of_week'].value_counts().reindex(day_order))

        return self

    def descriptive_statistics(self):
        """Generate descriptive statistics and visualizations for the dataset"""
        # Create figure for subplots
        fig = plt.figure(figsize=(15, 10))

        # Calculate publisher and day counts
        publisher_counts = self.df['publisher'].value_counts()
        day_counts = self.df['day_of_week'].value_counts()

        # 1. Headline Length Distribution
        plt.subplot(2, 2, 1)
        sns.histplot(data=self.df, x='headline_length', bins=50)
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Number of Characters')
        plt.ylabel('Count')

        # 2. Top Publishers Bar Chart
        plt.subplot(2, 2, 2)
        publisher_counts.head(10).plot(kind='bar')
        plt.title('Top 10 Publishers by Article Count')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Articles')

        # 3. Articles by Day of Week
        plt.subplot(2, 2, 3)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts_ordered = day_counts.reindex(day_order)
        day_counts_ordered.plot(kind='bar')
        plt.title('Article Distribution by Day of Week')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Articles')

        # 4. Monthly Article Trends
        plt.subplot(2, 2, 4)
        monthly_counts = self.df.groupby([self.df['date'].dt.year, self.df['date'].dt.month]).size()
        monthly_counts.plot(kind='line')
        plt.title('Monthly Article Count Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Articles')

        plt.tight_layout()
        plt.show()

        # Print detailed statistics
        print("\nHeadline Length Statistics:")
        print(f"Average length: {self.df['headline_length'].mean():.1f} characters")
        print(f"Median length: {self.df['headline_length'].median():.1f} characters")
        print(f"Most common length: {self.df['headline_length'].mode().iloc[0]} characters")

        print("\nPublication Patterns:")
        print(f"Most active day: {day_counts.index[0]} ({day_counts.iloc[0]} articles)")
        print(f"Least active day: {day_counts.index[-1]} ({day_counts.iloc[-1]} articles)")

        print("\nPublisher Diversity:")
        print(f"Total number of unique publishers: {self.df['publisher'].nunique()}")
        print(f"Top publisher: {publisher_counts.index[0]} ({publisher_counts.iloc[0]} articles)")

        return self

    def analyze_sentiment(self):
        """
        Performs sentiment analysis on news headlines and visualizes the results
        """
        print("\nPerforming Sentiment Analysis...")

        # Calculate sentiment scores and categories
        self.df['sentiment'] = self.df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
        self.df['sentiment_category'] = pd.cut(self.df['sentiment'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['Negative', 'Neutral', 'Positive'])

        # Calculate and display sentiment distribution
        sentiment_dist = self.df['sentiment_category'].value_counts()
        print("\nSentiment Distribution:")
        for category, count in sentiment_dist.items():
            percentage = (count/len(self.df))*100
            print(f"{category}: {percentage:.1f}%")

        # Visualize sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='sentiment_category', order=['Negative', 'Neutral', 'Positive'])
        plt.title('Distribution of Headline Sentiments')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Count')
        plt.show()

        # Convert dates to datetime and preserve timezone
        self.df['date'] = pd.to_datetime(self.df['date'])
        # Group by month while preserving timezone
        monthly_sentiment = self.df.groupby(self.df['date'].dt.strftime('%Y-%m'))['sentiment'].mean()
        plt.figure(figsize=(12, 6))
        monthly_sentiment.plot(kind='line')
        plt.title('Average Sentiment Score Over Time')
        plt.xlabel('Month')
        plt.ylabel('Average Sentiment Score')
        plt.grid(True)
        plt.show()

        return self

    def extract_topics(self):
        """
        Performs topic modeling using LDA and visualizes topic distributions
        """
        print("\nPerforming Topic Modeling...")

        # Prepare text data
        vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(self.df['headline'])

        # Apply LDA with optimal parameters
        n_topics = 5
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        topic_results = lda.fit_transform(doc_term_matrix)
        print(topic_results)

        # Extract and visualize top terms for each topic
        feature_names = vectorizer.get_feature_names_out()
        n_top_words = 10

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for topic_idx, topic in enumerate(lda.components_):
            if topic_idx < n_topics:
                top_words_idx = topic.argsort()[:-n_top_words-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]

                axes[topic_idx].barh(top_words, top_weights)
                axes[topic_idx].set_title(f'Topic {topic_idx + 1}')
                axes[topic_idx].invert_yaxis()

        plt.tight_layout()
        plt.show()

        return self

    def analyze_key_phrases(self):
        """
        Analyzes and visualizes frequency of common financial phrases in headlines
        """
        print("\nAnalyzing Key Financial Phrases...")

        financial_phrases = [
            'price target', 'upgrade', 'downgrade', 'earnings',
            'FDA approval', 'merger', 'acquisition', 'IPO',
            'stock split', 'dividend', 'guidance', 'analyst rating'
        ]

        # Calculate phrase frequencies
        phrase_counts = {}
        for phrase in financial_phrases:
            mask = self.df['headline'].str.lower().str.contains(phrase)
            count = mask.sum()
            phrase_counts[phrase] = count

        # Create DataFrame for visualization
        phrase_df = pd.DataFrame.from_dict(phrase_counts, orient='index', columns=['count'])
        phrase_df['percentage'] = (phrase_df['count'] / len(self.df)) * 100
        phrase_df = phrase_df.sort_values('count', ascending=True)

        # Print statistics
        print("\nCommon Financial Phrases Frequency:")
        for phrase, row in phrase_df.iterrows():
            print(f"{phrase}: {row['count']} occurrences ({row['percentage']:.1f}%)")

        # Visualize phrase frequencies
        plt.figure(figsize=(12, 6))
        phrase_df['count'].plot(kind='barh')
        plt.title('Frequency of Common Financial Phrases')
        plt.xlabel('Number of Occurrences')
        plt.tight_layout()
        plt.show()

        return self

    def analyze_publication_patterns(self):
        """
        Analyzes and visualizes temporal patterns in news publication frequency
        """
        print("\nAnalyzing Publication Patterns...")

        # Convert date column to datetime if not already
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Daily publication frequency
        daily_pubs = self.df.groupby(self.df['date'].dt.date).size()

        # Hourly distribution
        hourly_pubs = self.df.groupby(self.df['date'].dt.hour).size()

        # Day of week distribution
        dow_pubs = self.df.groupby(self.df['date'].dt.day_name()).size()

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        # Plot daily trend
        daily_pubs.plot(ax=axes[0])
        axes[0].set_title('Daily Publication Frequency')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Number of Articles')

        # Plot hourly distribution
        hourly_pubs.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Hourly Publication Distribution')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Number of Articles')

        # Plot day of week distribution
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_pubs = dow_pubs.reindex(dow_order)
        dow_pubs.plot(kind='bar', ax=axes[2])
        axes[2].set_title('Day of Week Publication Distribution')
        axes[2].set_xlabel('Day of Week')
        axes[2].set_ylabel('Number of Articles')

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\nPublication Pattern Statistics:")
        print(f"Average daily publications: {daily_pubs.mean():.1f}")
        print(f"Peak publishing hour: {hourly_pubs.idxmax()}:00")
        print(f"Most active day: {dow_pubs.idxmax()}")
        print(f"Least active day: {dow_pubs.idxmin()}")

        return self

    def analyze_publishers(self):
        """
        Analyzes publisher distribution and patterns in the dataset.
        """
        print("\nAnalyzing Publisher Patterns...")

        # Get publisher frequency counts
        pub_counts = self.df['publisher'].value_counts()

        # Print top publishers
        print("\nTop 10 Publishers by Article Count:")
        print(pub_counts.head(10))

        # Extract email domains if present
        def extract_domain(publisher):
            if '@' in str(publisher):
                return publisher.split('@')[1]
            return publisher

        self.df['domain'] = self.df['publisher'].apply(extract_domain)
        domain_counts = self.df['domain'].value_counts()

        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot top 15 publishers
        pub_counts.head(15).plot(kind='bar', ax=ax1)
        ax1.set_title('Top 15 Publishers by Article Count')
        ax1.set_xlabel('Publisher')
        ax1.set_ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')

        # Plot top 15 domains
        domain_counts.head(15).plot(kind='bar', ax=ax2)
        ax2.set_title('Top 15 Publisher Domains')
        ax2.set_xlabel('Domain')
        ax2.set_ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

        # Calculate and print statistics
        print("\nPublisher Statistics:")
        print(f"Total unique publishers: {len(pub_counts)}")
        print(f"Total unique domains: {len(domain_counts)}")
        print(f"\nTop publishing domain: {domain_counts.index[0]} ({domain_counts.iloc[0]} articles)")

        # Analyze content patterns by top publishers
        top_publishers = pub_counts.head(5).index
        print("\nContent Analysis for Top 5 Publishers:")
        for publisher in top_publishers:
            pub_df = self.df[self.df['publisher'] == publisher]
            print(f"\n{publisher}:")
            print(f"Total articles: {len(pub_df)}")
            print(f"Average headline length: {pub_df['headline'].str.len().mean():.1f} characters")

        return self

    def save_to_csv(self, output_path="../data/processed_financial_news.csv"):
        """
        Save the processed DataFrame to a CSV file
        """
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save the dataframe to a CSV file
        self.df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        return self
