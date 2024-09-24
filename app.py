
from youtube_comment_downloader import YoutubeCommentDownloader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt



def get_youtube_comments(video_url):
    downloader = YoutubeCommentDownloader()
    comments = []
    
    for comment in downloader.get_comments_from_url(video_url):
        comments.append(comment['text'])
    
    # print(comments)
    total_comments = len(comments)
    print(f"Total comments: {total_comments}")
    return comments

def analyze_comments(comments):
    analyzer = SentimentIntensityAnalyzer()
    data = []
    
    for comment in comments:
        sentiment = analyzer.polarity_scores(comment)
        sentiment_category = "Neutral"
        if sentiment['compound'] > 0.05:
            sentiment_category = "Positive"
        elif sentiment['compound'] < -0.05:
            sentiment_category = "Negative"
        
        # Append comment and its sentiment category to the data list
        data.append([comment,  sentiment['compound'],sentiment_category])

    # Create a DataFrame for comments and sentiment
    df = pd.DataFrame(data, columns=['Comment', 'Compound_Score', 'Sentiment'])
    return df

def export_to_csv(df, filename="youtube_comments_analysis.csv"):
    # Export the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Data exported to {filename}")

def generate_feedback_graph(df):
    # Count the number of comments per sentiment category
    sentiment_counts = df['Sentiment'].value_counts()
    
    # Prepare labels with sentiment counts
    labels = [f'{sentiment} ({count})' for sentiment, count in sentiment_counts.items()]
    
    # Generate a pie chart for the sentiment distribution
    plt.figure(figsize=(6, 5))
    plt.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#99ff99' ,'#ff9999','#66b3ff'])
    plt.title('Sentiment Analysis of YouTube Comments')
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    
    # Save the pie chart to a file
    plt.savefig("sentiment_analysis_pie_chart.png")
    print("Sentiment analysis pie chart saved as 'sentiment_analysis_pie_chart.png'.")
    
    # Show the plot
    plt.show()

def display_comment_counts(df):
    # Count the number of positive, negative, and neutral comments
    positive_comments = df[df['Sentiment'] == 'Positive'].shape[0]
    negative_comments = df[df['Sentiment'] == 'Negative'].shape[0]
    neutral_comments = df[df['Sentiment'] == 'Neutral'].shape[0]
    
    # Print out the counts
    print(f"Total Positive Comments: {positive_comments}")
    print(f"Total Negative Comments: {negative_comments}")
    print(f"Total Neutral Comments: {neutral_comments}")




# Example usage
video_url = 'https://youtu.be/GdyAhNCovGc?si=0yf6iv9su8hzP-6x'

comments = get_youtube_comments(video_url)

df = analyze_comments(comments)
export_to_csv(df)

# Display comment counts
display_comment_counts(df)

# Generate and show the pie chart for sentiment analysis
generate_feedback_graph(df)

