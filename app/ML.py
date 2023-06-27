import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv')
print(data.head())

#Discover null values
data.isnull().sum()
data['language'].value_counts()

x = np.array(data['Text'])
y = np.array(data['language'])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Using multinominal Naive Bayes to train the model
model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)

