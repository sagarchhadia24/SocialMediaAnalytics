#******************************************************************************************
#       Sagar Chhadia (Computer Science)
#******************************************************************************************

# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from twython import TwythonStreamer
import sys
import json
from pymongo import MongoClient
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from os import path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import decomposition
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models


# DataBase Connection
class MongoDB():
    def __init__(self):
        client = MongoClient('')
        self.db = client['']
        self.db.authenticate('', '')

    def get_collection(self,collection_name):
        return self.db[collection_name]

tweets = []

class MyStreamer(TwythonStreamer):
    '''our own subclass of TwythonStremer'''

    # overriding
    def on_success(self, data):
        if 'lang' in data and data['lang'] == 'en' and ('trump' in data['text'] or 'Trump' in data['text']):
            tweets.append(data)
            testimonial = TextBlob(data['text'])
            textPolarity = testimonial.sentiment.polarity
            textSubjectivity = testimonial.sentiment.subjectivity
            postData = {"text": data['text'], "Polarity" : textPolarity, "Subjectivity" : textSubjectivity}
            client = MongoDB()
            countTweetsQuery = client.get_collection('texas').count()
            if countTweetsQuery < 1000:
                self.upload_data(postData)

        client = MongoDB()
        countTweetsQuery = client.get_collection('texas').count()
        if countTweetsQuery >= 1000:
            self.store_json()
            self.create_sentiment_histogram()
            self.create_word_cloud()
            self.nmf_topic_modeling()
            self.lda_topic_modeling()
            self.disconnect()

    # overriding
    def on_error(self, status_code, data):
        print status_code, data
        self.disconnect()

    def upload_data(self, postData):
        client = MongoDB()
        tweetsCollection = client.get_collection('texas')
        insertDataQuery = tweetsCollection.insert(postData)

    def create_sentiment_histogram(self):
        polList = []
        subList = []
        client = MongoDB()
        tweetsCollection = client.get_collection('texas')
        sentimentQuery = tweetsCollection.find()

        for data in sentimentQuery:
            polList.append(data["Polarity"])
            subList.append(data["Subjectivity"])

        avgPolarity = sum(polList) / len(polList)
        avgSubjectivity = sum(subList) / len(subList)
        plt.hist(polList, bins=40)  # , normed=1, alpha=0.75)
        plt.xlabel('Avg. Polarity Score : {}'.format(avgPolarity))
        plt.ylabel('tweet count')
        plt.grid(True)
        plt.show()

        plt.hist(subList, bins=40)  # , normed=1, alpha=0.75)
        plt.xlabel('Avg. Subjectivity Score : {}'.format(avgSubjectivity))
        plt.ylabel('tweet count')
        plt.grid(True)
        plt.show()

    def create_word_cloud(self):
        rawAllTweets = ""
        client = MongoDB()
        tweetsCollection = client.get_collection('texas')
        getTweetsQuery = tweetsCollection.find()

        for data in getTweetsQuery:
            rawAllTweets += data["text"]

        d = path.dirname(__file__)
        alice_coloring = np.array(Image.open(path.join(d, "stormtrooper_mask.png")))
        cachedStopWords = set(STOPWORDS)
        cachedStopWords.update(('well', 'say', 'New', 'RT', 'now', 'https', 'via', 'CIA', 'make','says','new','will','said',
                                'take', 'amp', 'one','go', 'know', 'day', 'look', 'think', 'keep', 'call', 'right', 'follow',
                                'want', 'got', 'right', 'MAGA', 'people', 'good', 'need', 'see', 'let'))

        filteredData = ''
        for word in rawAllTweets.split():
            if len(word) == 1 or word in cachedStopWords:
                continue
            filteredData += ' {}'.format(word)

        wc = WordCloud(background_color="white", max_words=2000, mask=alice_coloring,
                       stopwords=cachedStopWords, max_font_size=40, random_state=42)

        wc.generate(filteredData)

        plt.imshow(wc)
        plt.axis("off")
        plt.show()

    def nmf_topic_modeling(self):
        corpus = []
        client = MongoDB()
        tweetsCollection = client.get_collection('texas')
        findCorpusQuery = tweetsCollection.find()

        for fileid in findCorpusQuery:
            corpus.append(fileid['text'])

        cachedStopWords = set(STOPWORDS)
        cachedStopWords.update(
            ('well', 'say', 'New', 'RT', 'now', 'https', 'via', 'CIA', 'make', 'says', 'new', 'will', 'said',
             'take', 'amp', 'one', 'go', 'know', 'day', 'look', 'think', 'gt', 'lt', 'co', 'rt', 'zbvnkkvl48', 'gtwkxla6bn',
             'cia', 'elizasoul80', '5febckimwt', 'af', 'ex', 'realdonaldtrump', 'thanx', 're', 'us', 'tell', 'dworkinreport', 'people',
             'mr'))

        vectorizer = TfidfVectorizer(stop_words=cachedStopWords, min_df=2)
        dtm = vectorizer.fit_transform(corpus)
        vocab = vectorizer.get_feature_names()

        num_topics = 5

        clf = decomposition.NMF(n_components=num_topics, random_state=1)
        doctopic = clf.fit_transform(dtm)

        topic_words = []
        num_top_words = 5
        for topic in clf.components_:
            word_idx = np.argsort(topic)[::-1][0:num_top_words]  # get indexes with highest weights
            topic_words.append([vocab[i] for i in word_idx])

        for t in range(len(topic_words)):
            print "Topic {}: {}".format(t, ' '.join(topic_words[t][:15]))

    def lda_topic_modeling(self):
        docs = []
        client = MongoDB()
        tweetsCollection = client.get_collection('texas')
        getAllTweetsQuery = tweetsCollection.find()

        cachedStopWords = set(STOPWORDS)
        cachedStopWords.update(
            ('well', 'say', 'new', 'rt', 'now', 'https', 'via', 'cia', 'make', 'says', 'new', 'will', 'said', 'take','amp',
             'one', 'go', 'know', 'day', 'look', 'think', 'gt', 'lt', 'co', 'rt', 'zbvnkkvl48', 'gtwkxla6bn', 'cia',
             'elizasoul80', '5febckimwt', 'af', 'ex', 'realdonaldtrump', 're', 'mikepencevp', 'un', 'recv', 'cmmte',
             'gop', 'devinnunes', 'ahca', 'want', 'see', 'aca', 'us', 'mr', 'gorsuch', '@realdonaldtrump', 'https', '&amp;',
             'the', 'a', 'have', '@potus', 'trump\'s', 'https://t.co/wvnvoevrht', 'https://t.co/zapa754bvc',
             'https://t.co/tmurxjv5a4', 'is', 'you', 'we', 'be', 'ur', 'trump', 'the', 'https://t.co/epucokw6b3', 'https://t.co/vqpgqgfwye',
             'https://t.co/f0mei9xcwv', 'https://t.co/heioaednwr', 'https://t.co/jkksztxvt0'))

        for fileid in getAllTweetsQuery:
            txtData = fileid['text']
            tmpList = txtData.lower().split()
            for word in tmpList:
                if len(word) == 1 or word in cachedStopWords:
                    tmpList.remove(word)
            docs.append(tmpList)

        dic = corpora.Dictionary(docs)

        corpus = [dic.doc2bow(text) for text in docs]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        NUM_TOPICS = 5
        model = models.ldamodel.LdaModel(corpus_tfidf,
                                         num_topics=NUM_TOPICS,
                                         id2word=dic,
                                         update_every=1,
                                         passes=100)

        print("LDA model")
        topics_found = model.print_topics(20)
        counter = 1
        for t in topics_found:
            print("Topic #{} {}".format(counter, t))
            counter += 1

    def store_json(self):
        with open('tweet_stream_{}_{}.json'.format('texas', len(tweets)), 'w') as f:
            json.dump(tweets, f, indent=4)


if __name__ == '__main__':

    with open('your_twitter_credentials.json', 'r') as f:
        credentials = json.load(f)

    # App consumer key and secret
    CONSUMER_KEY = credentials['CONSUMER_KEY']
    CONSUMER_SECRET = credentials['CONSUMER_SECRET']
    ACCESS_TOKEN = credentials['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = credentials['ACCESS_TOKEN_SECRET']

    stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    #Texas
    lat1 = 26.380999
    long1 = -98.821477
    lat2 = 28.140393
    long2 = -96.988305
    stream.statuses.filter(locations=[long1,lat1,long2,lat2])
