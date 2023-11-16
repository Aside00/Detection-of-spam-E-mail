#!/usr/bin/env python
# coding: utf-8

# In[1]:


#by amjad and majd and amani
import numpy as np
import pandas as pd


# this project ML by : Amjad&majd althobiti and amani alqtami

# In[2]:


df = pd.read_csv('spam.csv', encoding='latin1')


# df.sample(5)

# In[4]:


df.shape


# In[5]:


# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


#  1. Data cleaning
# 

# In[6]:


df.info()


# In[7]:


# drop last 3 cols
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[8]:


df.sample(5)


# In[9]:


# renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[11]:


df['target'] = encoder.fit_transform(df['target'])


# In[12]:


df.head()


# In[13]:


# missing values
df.isnull().sum()


# In[14]:


# check for duplicate values
df.duplicated().sum()


# In[15]:


# remove duplicates
df = df.drop_duplicates(keep='first')


# In[16]:


df.duplicated().sum()


# In[17]:


df.shape


# 2.EDA

# In[18]:


df.head()


# In[19]:


df['target'].value_counts()


# In[20]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[21]:


# Data is imbalanced


# In[22]:


import nltk


# In[23]:


nltk.download('punkt')


# In[24]:


df['num_characters'] = df['text'].apply(len)


# In[25]:


df.head()


# In[26]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[27]:


df.head()


# In[28]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[29]:


df.head()


# In[30]:


df[['num_characters','num_words','num_sentences']].describe()


# In[31]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[32]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[33]:


import seaborn as sns


# In[34]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[35]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[36]:


sns.pairplot(df,hue='target')


# In[37]:


sns.pairplot(df,hue='target')

3. Data Preprocessing 
Lower case
Tokenization
Removing special characters
Removing stop words and punctuation
Stemming
# In[38]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[39]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[40]:


df['text'][10]


# In[41]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[42]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[43]:


df.head()


# In[44]:


pip install wordcloud


# In[45]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[46]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[47]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[48]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[49]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[50]:


df.head()


# In[51]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[52]:


len(spam_corpus)


# In[53]:


import pandas as pd
import seaborn as sns
from collections import Counter

spam_corpus = ["spam", "spam", "spam", "ham", "ham", "egg", "spam"]
most_common_words = Counter(spam_corpus).most_common(30)

data = pd.DataFrame(most_common_words)

sns.barplot(x=data[0], y=data[1])

plt.xticks(rotation="vertical")

plt.show()


# In[54]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[55]:


len(ham_corpus)


# In[56]:


import pandas as pd
import seaborn as sns

from collections import Counter

# Create a list of the most common words in the spam corpus
spam_corpus = ["spam", "spam", "spam", "ham", "ham", "egg", "spam"]
most_common_words = Counter(spam_corpus).most_common(30)

# Create a Pandas DataFrame from the most common words
data = pd.DataFrame(most_common_words)

# Create a bar plot of the most common words
sns.barplot(x=data[0], y=data[1])

# Rotate the x-axis labels to prevent overlapping
plt.xticks(rotation="vertical")

# Show the plot
plt.show()


# In[57]:


df.head()


# 4. Model Building

# In[58]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[59]:


X = tfidf.fit_transform(df['transformed_text']).toarray()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                


# In[60]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)


# In[61]:


# appending the num_character col to X
#X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))


# In[62]:


X.shape


# In[63]:


y = df['target'].values


# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[66]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[67]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[68]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[69]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[70]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[71]:


# tfidf --> MNB


# In[72]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[73]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[74]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[75]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[76]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[77]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[78]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[79]:


performance_df


# In[80]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[81]:


performance_df1


# In[82]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[83]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[84]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[85]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[86]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[87]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[88]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[89]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[90]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[91]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[92]:


voting.fit(X_train,y_train)


# In[93]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[97]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[98]:


from sklearn.ensemble import StackingClassifier


# In[99]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[100]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[101]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




