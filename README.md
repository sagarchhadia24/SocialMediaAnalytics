# SocialMediaAnalytics

We have used the twitter streaming API to collect the tweets. While we are collecting the tweets we are extracting only the text from the tweets to reduce the space complexity, calculating the polarity and subjectivity score. Once this is done we are storing this data in MongoDB. After the tweets are collected we are using the data from database to implement the following functions:

• Sentiment Analysis (create_sentiment_histogram): We are querying the database and
fetching the polarity and subjectivity values in lists and then putting these values in the
function plt.hist to get the polarity and subjectivity graphs.

• Word Cloud (create_word_cloud): We are querying the database and fetching the text
value in a string. After this we are filtering the data by removing the stop words and
storing the filtered data in a new string. Then passing the data of this new string to the
word cloud function.

• Topic Modeling (nmf_topic_modeling/lda_topic_modeling): We are querying the
database and fetching the text in a list and filtering the data to remove the stop words.
We have a list of 10000 tweets using this list we are generating topic models.
