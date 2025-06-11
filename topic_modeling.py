# NASA Climate Change Topic Modeling & Trend Analysis
# Advanced topic discovery and temporal trend analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class ClimateTopicAnalyzer:
    """Advanced topic modeling for climate change discussions"""
    
    def __init__(self, n_topics=6):
        self.n_topics = n_topics
        self.stop_words = self.create_climate_stopwords()
        self.lda_model = None
        self.vectorizer = None
        self.topic_labels = {}
        
    def create_climate_stopwords(self):
        """Create custom stopwords for climate discussions"""
        basic_stops = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                      'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                      'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                      'her', 'its', 'our', 'their', 'myself', 'yourself', 'himself', 'herself', 'itself',
                      'ourselves', 'yourselves', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
                      'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'get',
                      'got', 'like', 'one', 'two', 'also', 'back', 'good', 'way', 'well', 'new', 'first', 'last',
                      'long', 'great', 'little', 'right', 'old', 'see', 'come', 'know', 'take', 'make', 'think',
                      'go', 'want', 'say', 'look', 'use', 'work', 'time', 'year', 'day', 'people', 'man', 'woman']
        
        # Climate-specific common words that don't add meaning
        climate_stops = ['nasa', 'post', 'comment', 'page', 'facebook', 'link', 'article', 'read', 'watch',
                        'video', 'image', 'photo', 'share', 'thanks', 'thank', 'please', 'much', 'really',
                        'today', 'yesterday', 'tomorrow', 'said', 'says', 'tell', 'told', 'ask', 'asked']
        
        return basic_stops + climate_stops
    
    def preprocess_text(self, text):
        """Clean and preprocess text for topic modeling"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(filtered_words)

def extract_topics_lda(df, n_topics=6):
    """Extract topics using Latent Dirichlet Allocation"""
    print(f"\n🔍 EXTRACTING {n_topics} TOPICS USING LDA...")
    
    analyzer = ClimateTopicAnalyzer(n_topics=n_topics)
    
    # Preprocess text data
    print("Preprocessing text data...")
    processed_texts = df['text'].apply(analyzer.preprocess_text)
    
    # Remove empty texts
    valid_texts = processed_texts[processed_texts.str.len() > 0]
    print(f"Valid texts for analysis: {len(valid_texts)}")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=200,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Fit and transform the data
    tfidf_matrix = vectorizer.fit_transform(valid_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Apply LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10,
        learning_method='online'
    )
    
    lda.fit(tfidf_matrix)
    
    # Extract topic words
    topics = {}
    topic_labels = {}
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_weights = [topic[i] for i in top_words_idx]
        
        topics[f'Topic_{topic_idx + 1}'] = {
            'words': top_words,
            'weights': topic_weights
        }
        
        # Create descriptive labels based on top words
        label = create_topic_label(top_words)
        topic_labels[f'Topic_{topic_idx + 1}'] = label
    
    # Assign topics to documents
    doc_topic_probs = lda.transform(tfidf_matrix)
    dominant_topics = doc_topic_probs.argmax(axis=1)
    
    # Create results dataframe
    results_df = df.copy()
    results_df['processed_text'] = processed_texts
    
    # Add topic assignments for valid texts
    topic_assignments = ['No_Topic'] * len(df)
    valid_indices = processed_texts[processed_texts.str.len() > 0].index
    
    for i, idx in enumerate(valid_indices):
        topic_num = dominant_topics[i]
        topic_assignments[idx] = f'Topic_{topic_num + 1}'
    
    results_df['dominant_topic'] = topic_assignments
    results_df['topic_label'] = results_df['dominant_topic'].map(topic_labels).fillna('No Topic')
    
    print("Topic extraction completed!")
    
    return results_df, topics, topic_labels, lda, vectorizer

def create_topic_label(top_words):
    """Create descriptive labels for topics based on keywords"""
    # Define topic categories based on common climate themes
    topic_patterns = {
        'Climate_Science': ['climate', 'temperature', 'warming', 'data', 'research', 'study', 'science', 'evidence'],
        'Carbon_Emissions': ['carbon', 'co2', 'emissions', 'fossil', 'fuels', 'pollution', 'dioxide'],
        'Climate_Impacts': ['weather', 'extreme', 'flood', 'drought', 'hurricane', 'wildfire', 'sea', 'ice'],
        'Renewable_Energy': ['solar', 'wind', 'renewable', 'energy', 'clean', 'power', 'green', 'technology'],
        'Policy_Action': ['government', 'policy', 'action', 'law', 'regulation', 'agreement', 'paris'],
        'Climate_Skepticism': ['fake', 'hoax', 'lie', 'wrong', 'natural', 'cycle', 'sun', 'myth'],
        'Personal_Impact': ['future', 'children', 'kids', 'generation', 'planet', 'earth', 'home', 'life'],
        'Economic_Concerns': ['cost', 'money', 'economy', 'job', 'business', 'industry', 'economic']
    }
    
    # Score each category
    category_scores = {}
    for category, keywords in topic_patterns.items():
        score = sum(1 for word in top_words[:5] if any(keyword in word.lower() for keyword in keywords))
        if score > 0:
            category_scores[category] = score
    
    if category_scores:
        best_category = max(category_scores, key=category_scores.get)
        return best_category.replace('_', ' ')
    else:
        # Fallback: use top 2 words
        return f"{top_words[0].title()} {top_words[1].title()}"

def analyze_topic_trends(df):
    """Analyze how topics change over time"""
    print("\nANALYZING TOPIC TRENDS OVER TIME...")
    
    # Topic distribution over time
    topic_time = df.groupby(['year', 'topic_label']).size().unstack(fill_value=0)
    topic_time_pct = topic_time.div(topic_time.sum(axis=1), axis=0) * 100
    
    print("Topic Distribution by Year (%):")
    print(topic_time_pct.round(1))
    
    # Monthly trends
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_topics = df.groupby(['year_month', 'topic_label']).size().unstack(fill_value=0)
    
    # Topic engagement analysis
    topic_engagement = df.groupby('topic_label').agg({
        'engagement_score': ['mean', 'sum', 'count'],
        'vader_compound': 'mean',
        'text_length': 'mean'
    }).round(2)
    
    topic_engagement.columns = ['avg_engagement', 'total_engagement', 'comment_count', 
                               'avg_sentiment', 'avg_length']
    
    print("\nTopic Engagement Analysis:")
    print(topic_engagement.sort_values('avg_engagement', ascending=False))
    
    return topic_time_pct, monthly_topics, topic_engagement

def extract_trending_keywords(df):
    """Find trending keywords over time"""
    print("\nEXTRACTING TRENDING KEYWORDS...")
    
    # Keywords by year
    yearly_keywords = {}
    
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        
        # Combine all text for the year
        year_text = ' '.join(year_data['text'].fillna('').astype(str))
        
        # Clean and extract keywords
        analyzer = ClimateTopicAnalyzer()
        clean_text = analyzer.preprocess_text(year_text)
        
        # Count word frequencies
        words = clean_text.split()
        word_freq = Counter(words)
        
        # Get top keywords (filter out very common words)
        top_keywords = [(word, count) for word, count in word_freq.most_common(20) 
                       if len(word) > 3 and count > 3]
        
        yearly_keywords[year] = top_keywords
    
    print("Top Keywords by Year:")
    for year, keywords in yearly_keywords.items():
        print(f"\n{year}:")
        for word, count in keywords[:5]:
            print(f"  {word}: {count} mentions")
    
    return yearly_keywords

def analyze_sentiment_by_topic(df):
    """Analyze sentiment patterns within each topic"""
    print("\nSENTIMENT ANALYSIS BY TOPIC...")
    
    # Sentiment distribution by topic
    topic_sentiment = df.groupby(['topic_label', 'sentiment_category']).size().unstack(fill_value=0)
    topic_sentiment_pct = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0) * 100
    
    print("Sentiment Distribution by Topic (%):")
    print(topic_sentiment_pct.round(1))
    
    # Average sentiment scores by topic
    avg_sentiment = df.groupby('topic_label')['vader_compound'].mean().sort_values(ascending=False)
    
    print("\nAverage Sentiment Score by Topic:")
    for topic, sentiment in avg_sentiment.items():
        emotion = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😞"
        print(f"  {topic}: {sentiment:.3f} {emotion}")
    
    return topic_sentiment_pct, avg_sentiment

def create_topic_insights(df, topics, topic_engagement):
    """Generate key insights about topics"""
    print("\nKEY TOPIC INSIGHTS")
    print("=" * 30)
    
    # Most discussed topic
    topic_counts = df['topic_label'].value_counts()
    most_discussed = topic_counts.index[0]
    print(f"Most Discussed Topic: {most_discussed} ({topic_counts.iloc[0]} comments)")
    
    # Most engaging topic
    most_engaging = topic_engagement['avg_engagement'].idxmax()
    engagement_score = topic_engagement.loc[most_engaging, 'avg_engagement']
    print(f"Most Engaging Topic: {most_engaging} ({engagement_score:.2f} avg engagement)")
    
    # Most positive topic
    most_positive = topic_engagement['avg_sentiment'].idxmax()
    sentiment_score = topic_engagement.loc[most_positive, 'avg_sentiment']
    print(f"Most Positive Topic: {most_positive} ({sentiment_score:.3f} sentiment)")
    
    # Topic evolution insights
    topic_years = df.groupby(['topic_label', 'year']).size().unstack(fill_value=0)
    
    print("\nTopic Evolution Patterns:")
    for topic in topic_years.index:
        years = topic_years.columns
        counts = topic_years.loc[topic]
        
        if len(counts) > 1:
            trend = "Increasing" if counts.iloc[-1] > counts.iloc[0] else "📉 Decreasing"
            print(f"  {topic}: {trend}")
    
    # Topic-specific insights
    print(f"\nDetailed Topic Analysis:")
    for topic_name, topic_data in topics.items():
        if topic_name in df['dominant_topic'].values:
            topic_comments = df[df['dominant_topic'] == topic_name]
            avg_engagement = topic_comments['engagement_score'].mean()
            avg_sentiment = topic_comments['vader_compound'].mean()
            
            print(f"\n{topic_name}:")
            print(f"  Keywords: {', '.join(topic_data['words'][:5])}")
            print(f"  Comments: {len(topic_comments)}")
            print(f"  Avg Engagement: {avg_engagement:.2f}")
            print(f"  Avg Sentiment: {avg_sentiment:.3f}")

def save_topic_results(df, topics, topic_labels):
    """Save topic modeling results"""
    print("\n💾 Saving Topic Analysis Results...")
    
    # Save main dataset with topics
    output_file = 'nasa_climate_with_topics.csv'
    df.to_csv(output_file, index=False)
    
    # Save topic details
    topic_details = []
    for topic_id, topic_data in topics.items():
        topic_details.append({
            'topic_id': topic_id,
            'topic_label': topic_labels.get(topic_id, 'Unknown'),
            'top_words': ', '.join(topic_data['words'][:10]),
            'comment_count': (df['dominant_topic'] == topic_id).sum()
        })
    
    topic_df = pd.DataFrame(topic_details)
    topic_df.to_csv('topic_analysis_summary.csv', index=False)
    
    print(f"Dataset with topics saved as: {output_file}")
    print(f"Topic summary saved as: topic_analysis_summary.csv")
    
    return df

def main():
    """Main topic modeling and trend analysis pipeline"""
    print("CLIMATE TOPIC MODELING & TREND ANALYSIS")
    print("=" * 60)
    
    # Load data with sentiment analysis
    try:
        df = pd.read_csv('nasa_climate_with_sentiment.csv')
        print(f"Loaded sentiment data: {len(df)} comments")
    except FileNotFoundError:
        print("Sentiment data not found. please run sentiment analysis first.")
        return None
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract topics using LDA
    df, topics, topic_labels, lda_model, vectorizer = extract_topics_lda(df, n_topics=6)
    
    # Analyze topic trends over time
    topic_time_trends, monthly_topics, topic_engagement = analyze_topic_trends(df)
    
    # Extract trending keywords
    yearly_keywords = extract_trending_keywords(df)
    
    # Analyze sentiment by topic
    topic_sentiment_dist, avg_topic_sentiment = analyze_sentiment_by_topic(df)
    
    # Generate insights
    create_topic_insights(df, topics, topic_engagement)
    
    # Save results
    df = save_topic_results(df, topics, topic_labels)
    
    print(f"\nTopic modeling and trend analysis finished.")
    
    return df, topics, topic_labels

# Run the topic analysis pipeline
if __name__ == "__main__":
    topic_df, topics, labels = main()