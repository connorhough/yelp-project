"""
Sentiment classifier of yelp reviews using
SVM and nltk tweet tokenizer. Accuracy score of 0.86
"""

import pandas as pd
import wandb

run = wandb.init()
config = run.config
summary = run.summary

# Setup nltk TweetTokenizer
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
def tokenizer(doc):
    return tknzr.tokenize(doc)

# Get a pandas DataFrame object of all the data in the csv file
# and remove blank rows:
df = pd.read_csv('yelp.csv').dropna()

# Get pandas Series object of the text and sentiment columns:
text = df['text']
target = df['sentiment']

# Make sklearn Pipeline to perform feature extraction, perform
# TFIDF transform, and train with the SGDClassifier:
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenizer)),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(max_iter=1000))
])

# Train model and evaluate its score with cross-validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(text_clf, text, target)
print(score)
print("Mean Score: %.3f" % score.mean())
