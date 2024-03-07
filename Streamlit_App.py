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

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

wnl = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = stopwords.words('english')

def return_yt_comments(url):
    data = []
    chrome_options = Options()
    chrome_options.add_argument(r'--executable_path=C:/Program Files/chromedriver.exe')

    with Chrome(options=chrome_options) as driver:
        wait = WebDriverWait(driver, 15)
        driver.get(url)

        for item in range(5):
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
            time.sleep(2)

        for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content"))):
            data.append(comment.text)

    return data

def clean(org_comments):
    y = []
    for x in org_comments:
        x = x.split()
        x = [i.lower().strip() for i in x]
        x = [i for i in x if i not in stop_words]
        x = [i for i in x if len(i) > 2]
        x = [wnl.lemmatize(i) for i in x]
        y.append(' '.join(x))
    return y

def create_wordcloud(clean_reviews):
    for_wc = ' '.join(clean_reviews)
    wc_stops = set(STOPWORDS)
    wc = WordCloud(width=1400, height=800, stopwords=wc_stops, background_color='white').generate(for_wc)
    st.image(wc.to_array())

def return_sentiment(x):
    score = sia.polarity_scores(x)['compound']

    if score > 0:
        sent = 'Positive'
    elif score == 0:
        sent = 'Negative'
    else:
        sent = 'Neutral'
    return score, sent

def main():
    st.title("YouTube Comment Sentiment Analysis")

    url = st.text_input("Enter YouTube video URL:")

    if st.button("Analyze Comments"):
        org_comments = return_yt_comments(url)
        temp = [i for i in org_comments if 5 < len(i) <= 500]
        org_comments = temp

        clean_comments = clean(org_comments)

        np, nn, nne = 0, 0, 0
        predictions = []
        scores = []

        for i in clean_comments:
            score, sent = return_sentiment(i)
            scores.append(score)
            if sent == 'Positive':
                predictions.append('POSITIVE')
                np += 1
            elif sent == 'Negative':
                predictions.append('NEGATIVE')
                nn += 1
            else:
                predictions.append('NEUTRAL')
                nne += 1

        dic = []

        for i, cc in enumerate(clean_comments):
            x = {}
            x['Comment'] = cc
            x['Sentiment'] = predictions[i]
            dic.append(x)

        st.header("Analysis Results")
        st.write(f"Total comments analyzed: {len(clean_comments)}")

        # Color-coded sentiment counts
        st.header("Sentiment Counts:")
        st.write(f"Positive comments: <span style='color:green'>{np}</span>", unsafe_allow_html=True)
        st.write(f"Negative comments: <span style='color:red'>{nn}</span>", unsafe_allow_html=True)
        st.write(f"Neutral comments: <span style='color:blue'>{nne}</span>", unsafe_allow_html=True)

        # Bar graph of sentiment distribution using Plotly
        st.subheader("Sentiment Distribution")
        fig = px.bar(x=['Positive', 'Negative', 'Neutral'], y=[np, nn, nne], color=['Positive', 'Negative', 'Neutral'],
                     labels={'x': 'Sentiment', 'y': 'Count'}, color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'})
        st.plotly_chart(fig)
        
        

        df = pd.DataFrame(dic)
        st.subheader("Comments DataFrame:")
        st.dataframe(df)

        create_wordcloud(clean_comments)

if __name__ == '__main__':
    main()