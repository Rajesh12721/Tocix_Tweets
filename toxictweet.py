#import
import pandas as pd
import nltk
import string
import re
import matplotlib.pyplot as plt
from sklearn import metrics
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Remove Punctuvation
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
# Remove Space
def remove_spaces(text):
    tweet = text.replace('\n',' ').replace('\t',' ').replace('   ',' ').replace('  ',' ')
    return tweet
# Clean the data using stopwords, isalpha funtions and get Tokens
def clean_data(text):
    tokens = word_tokenize(text)
    tweet = [word.lower() for word in tokens if (word.lower() not in stopword_list) and(len(word)>2) and (word.isalpha())]
    return tweet
# Stemming
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text
# Lemmatizing
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    final_tweet = []
    for word in text:
        lemmatized_word = lemmatizer.lemmatize(word)
        final_tweet.append(lemmatized_word)
    return " ".join(final_tweet)
# File import
data = pd.read_csv("FinalBalancedDataset.csv")
a = data['Toxicity']
b = data['tweet']
df = pd.concat([b,a],axis=1)
# Download the functions
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Function call for data cleaning
df['tweet'] = df['tweet'].apply(lambda x: remove_punct(x))
df['tweet'] = df['tweet'].apply(lambda x: remove_spaces(x))
stopword_list = stopwords.words("english")
df['tweet'] = df['tweet'].apply(lambda x: clean_data(x))
ps = nltk.PorterStemmer()
df['tweet'] = df['tweet'].apply(lambda x: stemming(x))
df['tweet'] = df['tweet'].apply(lambda x: lemmatization(x))
# Train and Test the Data
x_train,x_test,y_train,y_test = train_test_split(df.tweet,df.Toxicity,test_size = 0.2,random_state=42)
# Model Processing Function
def model_process(model,x_train, y_train, x_test, y_test): #model_type = 1
    print(model)
    print("Bag of Words")
    cv = CountVectorizer()
    x_train_cv = cv.fit_transform(x_train.values)
    x_test_cv = cv.transform(x_test)
    model.fit(x_train_cv, y_train)
    y_pred = model.predict(x_test_cv)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print("TF-IDF")
    tfidf = TfidfVectorizer()
    x_train_tf = tfidf.fit_transform(x_train.values)
    x_test_tf = tfidf.transform(x_test.values)
    model.fit(x_train_tf, y_train)
    y_pred = model.predict(x_test_tf)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    return x_test_cv, x_test_tf
# Dicision Tree
model = tree.DecisionTreeClassifier()
x_test_cv, x_test_tf = model_process(model,x_train, y_train, x_test, y_test)
preds1 = model.predict_proba(x_test_cv)[:,1]
preds2 = model.predict_proba(x_test_tf)[:,1]
fprcv1, tprcv1, thresholdcv1 = metrics.roc_curve(y_test, preds1)
fprtf1, tprtf1, thresholdtf1 = metrics.roc_curve(y_test, preds2)
roc_auc_cv1 = metrics.auc(fprcv1, tprcv1)
roc_auc_tf1 = metrics.auc(fprtf1, tprtf1)
# Random Forest
model = ExtraTreesClassifier()
x_test_cv, x_test_tf = model_process(model,x_train, y_train, x_test, y_test)
preds1 = model.predict_proba(x_test_cv)[:,1]
preds2 = model.predict_proba(x_test_tf)[:,1]
fprcv2, tprcv2, thresholdcv2 = metrics.roc_curve(y_test, preds1)
fprtf2, tprtf2, thresholdtf2 = metrics.roc_curve(y_test, preds2)
roc_auc_cv2 = metrics.auc(fprcv2, tprcv2)
roc_auc_tf2 = metrics.auc(fprtf2, tprtf2)
# KNN Classifier
model = KNeighborsClassifier()
x_test_cv, x_test_tf = model_process(model,x_train, y_train, x_test, y_test)
preds1 = model.predict_proba(x_test_cv)[:,1]
preds2 = model.predict_proba(x_test_tf)[:,1]
fprcv3, tprcv3, thresholdcv1 = metrics.roc_curve(y_test, preds1)
fprtf3, tprtf3, thresholdtf1 = metrics.roc_curve(y_test, preds2)
roc_auc_cv3 = metrics.auc(fprcv3, tprcv3)
roc_auc_tf3 = metrics.auc(fprtf3, tprtf3)
# SVM
model = svm.SVC(probability=True)
cv, tf = model_process(model,x_train, y_train, x_test, y_test)
# SVM OvO
model = svm.SVC(decision_function_shape='ovo', probability=True)
cv, tf = model_process(model,x_train, y_train, x_test, y_test)
# Linear SVM
model = svm.LinearSVC()
cv, tf = model_process(model,x_train, y_train, x_test, y_test)
# SVM RBF
model = svm.SVC(kernel='rbf')
cv, tf = model_process(model,x_train, y_train, x_test, y_test)
# Bag of Words ROC Curve Ploting
plt.title('ROC and AUC Curve (Bag of Words)')
plt.plot(fprcv1, tprcv1, linestyle='--', color='blue', label = 'Dicision Tree = %0.2f' % roc_auc_cv1)
plt.plot(fprcv2, tprcv2, linestyle='--', color='green', label = 'Random Forest = %0.2f' % roc_auc_cv2)
plt.plot(fprcv3, tprcv3, linestyle='--', color='black', label = 'KNN Classifier = %0.2f' % roc_auc_cv3)
plt.legend(loc='best')
plt.plot([0, 1], [0, 1],'r')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Y Label')
plt.xlabel('X Label')
plt.savefig('Toxictweet_BoW',dpi=300)
plt.show()
print("\n")
# TF - IDF ROC Curve Ploting
plt.title('ROC and AUC Curve (TF - IDF)')
plt.plot(fprtf1, tprtf1, linestyle='--', color='blue', label = 'Dicision Tree = %0.2f' % roc_auc_tf1)
plt.plot(fprtf2, tprtf2, linestyle='--', color='green', label = 'Random Forest = %0.2f' % roc_auc_tf2)
plt.plot(fprtf3, tprtf3, linestyle='--', color='black', label = 'KNN Classifier = %0.2f' % roc_auc_tf3)
plt.legend(loc='best')
plt.plot([0, 1], [0, 1],'r')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Y Label')
plt.xlabel('X Label')
plt.savefig('Toxictweet_TF',dpi=300)
plt.show()