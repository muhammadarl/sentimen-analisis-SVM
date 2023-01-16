import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
import pickle
from sklearn.model_selection import train_test_split
import translators as ts
import datetime
from datetime import datetime

lemmatizer = WordNetLemmatizer()
english_stop_words = stopwords.words('english')
def pre_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    tweet = re.sub('https', ' ', tweet)
    tweet = re.sub('@[^\s]+',' ',tweet)
    # tweet = re.sub(r'[0-9]', '', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub('ð', '', tweet)
    tweet = tweet.strip('\'"')
    tweet = ' '.join([word for word in tweet.split()  if word not in english_stop_words])
    tweet = ' '.join([lemmatizer.lemmatize(word) for word in tweet.split()])
    return tweet
def generate_sentiment(tweet, feature_text):
  def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
  def getPolarity(text):
    return TextBlob(text).sentiment.polarity
  tweet['TextBlob_Subjectivity'] = tweet[feature_text].apply(getSubjectivity)
  tweet['TextBlob_Polarity'] = tweet[feature_text].apply(getPolarity)
  def getAnalysis(score):
    if score < 0:
      return 0
    elif score == 0:
      return 2
    else:
      return 1
  tweet['sentimen'] = tweet['TextBlob_Polarity'].apply(getAnalysis )
  return tweet
def convert_datetime(datetime):
  datetime_obj = datetime.fromtimestamp(datetime).strftime('%d-%m-%y')
  date = datetime_obj.date()
  return date
def translate(tweet):
  tweet = ts.translate_text(tweet, translator='google')
  return tweet
def vektorisasi_teks(data_train):
  model = pickle.load(open('model_tv.pkl', 'rb'))
  data_train = model.transform(data_train)
  return data_train
def normalisasi_vektor(data_train):
  model = pickle.load(open('model_scaler.pkl', 'rb'))
  data_train = model.transform(data_train)
  return data_train
def balancing(x_train, y_train):
  model_ = pickle.load(open('pipeline.pkl', 'rb'))
  x_train, y_train = model_.fit_resample(x_train, y_train)
  return x_train, y_train
def train_test_split_data(X, Y):
  x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.80, random_state=10)
  return x_train, x_test, y_train, y_test