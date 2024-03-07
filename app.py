
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import streamlit as st
import time
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import plotly.express as px
import pandas as pd
import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import os
import pandas as pd
import streamlit as st
import plotly as pt





#from tensorflow.keras.models import load_model

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

wnl = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = stopwords.words('english')

def return_yt_comments():
   data= []
   data= pd.read_csv("output_comments.csv")
    #data.append(df)
   return data

def main():
    st.title("YouTube Comment Sentiment Analysis")
    dfs = return_yt_comments()
    st.write(dfs)
    fields = ['Text']
    df = pd.read_csv('output_comments.csv', usecols=fields)
    all_comments = ' '.join(dfs['Text'].astype(str))
    wordcloud_positive = WordCloud(background_color="black", width=800, height=400).generate(all_comments )
    wordcloud_negative = WordCloud(background_color="black", width=800, height=400).generate(all_comments )
    wordcloud_neutral = WordCloud(background_color="black", width=800, height=400).generate(all_comments )
    print("analysing.....")
    sentiment_counts = dfs['Sentiment'].value_counts()
    plt.bar(sentiment_counts.index, sentiment_counts.values)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Analysis of Comments")
    plt.show()
    if st.button("Analyze Comments"):
       
        st.image(wordcloud_positive.to_array(), caption="Positive Sentiments",use_column_width=True)
        st.image(wordcloud_neutral.to_array(), caption="Neutral Sentiments",use_column_width=True)
        st.image(wordcloud_negative.to_array(), caption="Negative Sentiments",use_column_width=True)

        df = pd.read_csv('output_comments.csv')
        # Calculate sentiment counts
        sentiment_counts = df['Sentiment'].value_counts()
        # Create a bar graph using Plotly
        fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values, labels={'x':'Sentiment', 'y':'Count'}, title='Sentiment Analysis of Comments')
        # Display the bar graph in the Streamlit app
        st.plotly_chart(fig)

        df = pd.read_csv('video_details.csv')
        # Create a bar graph for 'Like'
        fig1 = px.bar(df, x='Title', y='Like', labels={'x':'Video Title', 'y':'Likes'}, title='Number of Likes per Video')
        st.plotly_chart(fig1)

        # Create a bar graph for 'views'
        fig2 = px.bar(df, x='Title', y='views', labels={'x':'Video Title', 'y':'Views'}, title='Number of Views per Video')
        st.plotly_chart(fig2)
        # Count the number of videos published each month
        month_counts = df['Month'].value_counts()
        # Create a bar graph for the number of videos published each month
        fig3 = px.bar(month_counts, x=month_counts.index, y=month_counts.values, labels={'x':'Month', 'y':'Number of Videos'}, title='Number of Videos Published per Month')
        st.plotly_chart(fig3)
        
if __name__ == '__main__':
    main()
