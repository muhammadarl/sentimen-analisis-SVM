import streamlit as st
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from predict import predict, classification_report, hasil_predict
from preprocessing import pre_processing, generate_sentiment, vektorisasi_teks, normalisasi_vektor, balancing, train_test_split_data
from plotting import lineplot_monthly, distribution_before, distribution_after, after_remove_stopwords, distribution_sentimen, distribution_sentimen_balance, distribusi_report
import matplotlib as mpl
from connection import insert_data
import datetime
from datetime import datetime
host="127.0.0.1"
port = 3306
database="sentimen_etle"
user="root"

try:
    conn = create_engine('mysql+pymysql://{}:@{}:{}/{}'.format(user,host,port,database))
except (Exception, NameError) as error:
    st.write("Error while connecting to mysql", error)

st.sidebar.header('Variable Input')
st.sidebar.subheader('Tweet')
tweet = st.sidebar.text_area('Text to analyze', 'Masukan Teks Bahasa Indonesia')


st.title('Analisa sentimen masyarakat terhadap kebijakan polisi tilang manual')
mpl.rcParams['figure.dpi'] = 300
tweet_df = pd.read_sql('SELECT * FROM dataset_etle', conn)
st.markdown('## Hasil Predict')
st.markdown('#### Hasil Prediksi dan Probabilitas')
predict, predict_proba = predict(tweet)
hasil_predict(predict)
hasil_df = pd.DataFrame(predict_proba, columns=['Probabilitas Negatif', 'Probabilitas Positif'])
st.write(hasil_df)
st.markdown('#### Perkembangan Tweet Perbulan 2022')
st.markdown('###### Perkembangan Jumlah Tweet dengan jangka waktu Perbulan pada tahun 2022')
lineplot_monthly(tweet_df)
c1, c2 = st.columns((5,5))
with c1:
    st.markdown('#### Panjang Tweet Sebelum Cleaning Data')
    st.markdown('###### Panjang dari tweet sebelum tweet melewati tahap cleaning data')
    distribution_before(tweet_df)
tweet_df['translate'] = tweet_df['translate'].apply(lambda x: pre_processing(x))
with c2:
    st.markdown('#### Panjang Tweet Setelah Cleaning Data')
    st.markdown('###### Panjang dari tweet setelah tweet melewati tahap cleaning data, remove stopwords & lemmatization')
    distribution_after(tweet_df)
st.markdown('#### 5 Kata terbanyak')
st.markdown('###### 5 Kata terbanyak setelah remove stopwords')
after_remove_stopwords(tweet_df)
generate_sentiment(tweet_df, 'translate')
tweet_df = tweet_df[tweet_df['sentimen']!=2]
tweet_df['sentimen'] = tweet_df['sentimen'].astype('int')
c1, c2 = st.columns((5,5))
with c1:
    st.markdown('#### Distribution Sentimen')
    st.markdown('###### Distribution sentimen sebelum balancing data')
    distribution_sentimen(tweet_df)
x = tweet_df['translate']
y = tweet_df['sentimen']
y = np.array(y)
x = vektorisasi_teks(x)
x = normalisasi_vektor(x)
x, y = balancing(x,y)
with c2:
    st.markdown('#### Distribution Sentimen balance')
    st.markdown('###### Distribution sentimen setelah balancing data')
    distribution_sentimen_balance(tweet_df, y)
x_train, x_test, y_train, y_test = train_test_split_data(x, y)
st.markdown('#### Evaluation report')
st.markdown('###### Hasil Evaluasi akurasi, presisi, recall dan f1')
akurasi, presisi, recall, f1 = classification_report(x_test, y_test)
distribusi_report(akurasi = akurasi, presisi = presisi, recall = recall, f1 = f1)


