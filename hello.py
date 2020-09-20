import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
import plotly.express as px
import re
import pickle
import plotly.express as px
from PIL import Image
from wordcloud import WordCloud, STOPWORDS 
from colorama import Fore, Back, Style
import gcsfs

# page styling with css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")


logo = Image.open('SentimentSmart.jpg')
pos = Image.open('pos.jpg')
neg = Image.open('neg.jpg')
neu = Image.open('neu.jpg')
st.image(logo, use_column_width=True)

# st.title('SentimentSmart')
st.subheader('A text sentiment analysis tool.')
# st.markdown(
# """
# ### A text sentiment analysis tool.
# [See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
# """)
sentence_raw = st.text_input('Write a sentence here:') 

fs = gcsfs.GCSFileSystem(project='my-first-project')
fs.ls('sentimentsmart-data')

pickle_file = fs.open('sentimentsmart-data/sentiment.pkl')
m = pickle.load(pickle_file)

pickle_file2 = open("topfeatures.pkl","rb")
top_features = pickle.load(pickle_file2)

if sentence_raw:
    sentence=[sentence_raw]
    prediction = m.predict(sentence)[0]
    predict_proba = m.predict_proba(sentence)[0]
    x = ['Negative', 'Neutral', 'Positive']
    sent_data = pd.DataFrame(zip(prediction, predict_proba), columns = ['x', 'predict_proba'])
    fig = px.pie(sent_data, values='predict_proba', names=['Negative', 'Neutral', 'Positive'])
    # st.write('This sentence is', prediction)
    
    
    def smiley_resize(pp):
        return pp*600

    resize_pos = smiley_resize(predict_proba[2]).round(0).astype(int)
    new_pos = pos.resize((resize_pos, resize_pos))
    resize_neu = smiley_resize(predict_proba[1]).round(0).astype(int)
    new_neu = neu.resize((resize_neu, resize_neu))
    resize_neg = smiley_resize(predict_proba[0]).round(0).astype(int)
    new_neg = neg.resize((resize_neg, resize_neg))

    st.image([new_pos, new_neu, new_neg])

    margin = ((sorted(predict_proba, reverse=True)[0] - sorted(predict_proba, reverse=True)[1])*100).round(0).astype(int)
    
    if margin <= 5:
        st.write('Unsure of sentiment (<5% confidence)')
    else:
        st.write('This sentence is', prediction, 'with', margin, '% confidence.')
        st.write((predict_proba[2]*100).astype(int), '% positive', (predict_proba[1]*100).astype(int), '% neutral and ', (predict_proba[0]*100).astype(int), '% negative.')

    

    highlight_words = []
    for feature in top_features.feature:
        
        if feature in sentence_raw:
            highlight_words.append(feature)

    dic = dict(zip(highlight_words, highlight_words))

    if len(highlight_words) != 0:
        # def replace_all(text, dic):
        #     for i, j in dic.items():
        #         j = '**'+ j + '**'
        #         text = sentence_raw.replace(i, j)
        #     return text
        
        # st.markdown('Sentiment predictors are shown in bold:')
        # st.markdown(replace_all(sentence_raw, dic))
        st.write('Top 100 sentiment predictor words in this sentence:', ', '.join(highlight_words))
    # st.plotly_chart(fig)

st.markdown('_____________________________________________________________________________________')
st.title('SentimentSmart on COVID-19 tweets')

st.markdown('### An analysis of sentiments by location')

option = st.selectbox('Please choose a location',('---', 'India', 'United States', 'New Delhi, India', 'London, England',
       'United Kingdom', 'Mumbai, India', 'Washington, DC',
       'New York, NY', 'Australia', 'Canada', 'Los Angeles, CA',
       'South Africa', 'Lagos, Nigeria', 'California, USA',
       'Nairobi, Kenya', 'Nigeria', 'Atlanta, GA', 'Chicago, IL',
       'Switzerland', 'Johannesburg, South Africa'))

if option != '---':
   
    @st.cache(persist=True)
    def load_data(option):
        data = pd.read_csv('https://storage.googleapis.com/sentimentsmart-data/covid_deploy.csv')
        return data[data.user_location == option]
    
    data = load_data(option)

    compound_location = data.groupby(by=['tweet_date']).mean()[['negative', 'neutral', 'positive']].join(data.groupby(by=['tweet_date']).count().text).reset_index()
    compound_location['days'] = (pd.to_datetime(compound_location.tweet_date) - pd.to_datetime(compound_location.tweet_date.min())).dt.days


    plotdata = pd.melt(compound_location, id_vars = ['tweet_date', 'text', 'days'], value_vars = ['negative', 'neutral', 'positive'])

    fig = px.scatter(plotdata, x='variable', y='value',
                animation_frame='days',
                animation_group = 'variable',
                size = 'text', 
                color = 'value', 
                color_continuous_scale='viridis',
                range_y = [0,1],  
                size_max=55,
                labels={'value': 'Sentiment score', 'days': 'Day', 'text': 'Number of tweets', 'variable':'Sentiment'},
                title='Sentiment scores of COVID-19 tweets per day Jul-Aug 2020')


    st.plotly_chart(fig)
    
    # Wordcloud
    comment_words = '' 
    stopwords = nltk.corpus.stopwords.words('english')
    newstopwords = ['coronavirus','covid', '19', 'covid19', 'amp']
    stopwords.extend(newstopwords)

    mask = np.array(Image.open('./twitter2.jpg'))

    @st.cache
    def one_color_func(word=None, font_size=None, 
                    position=None, orientation=None, 
                    font_path=None, random_state=None):
        h = 200 # 0 - 360
        s = 100 # 0 - 100
        l = random_state.randint(30, 70) # 0 - 100
        return "hsl({}, {}%, {}%)".format(h, s, l)

    font_path = './coolvetica rg.ttf'
    # iterate through the csv file 
    for val in data.cleaned_tweets: 
        
    # typecaste each val to string 
        val = str(val) 

        # split the value 
        tokens = val.split() 
            
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
            
        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 1600, height = 1600, 
                    background_color ='white', 
                    mask=mask,
                    font_path=font_path,
                    stopwords = stopwords, 
                    min_font_size = 10,
                    color_func=one_color_func).generate(comment_words) 

    # plot the WordCloud image                        
    plt.figure(figsize = (15, 15), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    st.pyplot()

