import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import matplotlib.ticker as mtick
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.patches as mpatches
def lineplot_monthly(tweet):
    tweet['date'] = pd.to_datetime(tweet['date'], errors='coerce')
    plotting_monthly = tweet.groupby(tweet.date.dt.month)['tweet'].count()
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(x = plotting_monthly.index, palette=['o'] , y = plotting_monthly)
    plt.title('ETLE Tweets and Manual Ticket Quantity in 2022')
    plt.xticks(np.arange(1, 13, step=1))
    plt.ylabel('Frequency')
    plt.xlabel('Monthly')
    st.pyplot(fig)

def distribution_before(tweet):
    Tweet_content_len = tweet['translate'].apply(lambda p: len(p.split(' ')))
    max_Tweet_content_len = Tweet_content_len.max()
    print('max Tweet_content len: {0}'.format(max_Tweet_content_len))
    fig = plt.figure(figsize = (10, 4))
    sns.distplot(Tweet_content_len)
    plt.title('Length Distribution of Tweets')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.title('Length of Tweets before Cleaning Data')
    st.pyplot(fig)
def distribution_after(tweet):
    Tweet_content_len = tweet['translate'].apply(lambda p: len(p.split(' ')))
    max_Tweet_content_len = Tweet_content_len.max()
    print('max Tweet_content len: {0}'.format(max_Tweet_content_len))
    fig = plt.figure(figsize = (10, 4))
    sns.distplot(Tweet_content_len)
    plt.title('Length Distribution of Tweets')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.title('Length of Tweets after Cleaning Data')
    st.pyplot(fig)
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
def after_remove_stopwords(tweet):
    common_words = get_top_n_words(tweet['translate'], 5)
    df2 = pd.DataFrame(common_words, columns = ['translate' , 'count'])
    fig = plt.figure(figsize = (10, 4))
    sns.barplot(x='translate', y='count', data=df2, hue='translate',dodge=False)
    plt.title('Top Five Words After Deleting Stop Words')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.xlabel('words')
    plt.ylabel('Frequency')
    plt.legend(loc=1)
    st.pyplot(fig)
def distribution_sentimen(tweet):
    fig = plt.figure(figsize = (10, 4))
    sns.countplot(x=tweet['sentimen'], hue=tweet['sentimen'],dodge=False)
    plt.title('Distribution of Sentiment')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('Frequency')
    plt.xlabel('Sentiment')
    plt.legend(labels=["Negative", 'Positive'])
    st.pyplot(fig)
def distribution_sentimen_balance(tweet, y):
    fig = plt.figure(figsize = (10, 4))
    sns.countplot(x=y, hue=y,dodge=False)
    plt.title('Distribution of Sentiment')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('Frequency')
    plt.xlabel('Sentiment')
    plt.legend(labels=["Negative", 'Positive'])
    st.pyplot(fig)
def distribusi_report(akurasi, presisi, recall, f1):
    height = [akurasi, presisi, recall, f1]
    width = ['Akurasi', 'Presisi', 'Recall', 'F1']
    fig = plt.figure(figsize = (10, 4))
    sns.barplot(x = width, y = height,hue=width,dodge=False)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    for i in range(len(width)):
        plt.text(i, height[i]/2, '{}%'.format(height[i]*100), ha='center', va='center')
    plt.title("Evaluation Report")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('Percentage %')
    plt.xlabel('Evaluation')
    akurasi_label = mpatches.Patch(color='blue', label='Accuracy')
    presisi_label = mpatches.Patch(color='orange', label='Precision')
    recall_label = mpatches.Patch(color='green', label='Recall')
    f1_label = mpatches.Patch(color='red', label='F1')
    plt.legend(['Akurasi', 'Presisi', 'Recall', 'F1'], handles=[akurasi_label, presisi_label, recall_label, f1_label ], bbox_to_anchor=(1.05, 1))
    st.pyplot(fig)