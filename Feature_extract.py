# Data PLot feature extraction pipeline for MovieLens dataset
import pandas as pd
from empath import Empath
lexicon = Empath()
filename="links.csv"
df=pd.read_csv(filename,encoding="utf-8")
from imdb import IMDb
ia = IMDb()

increments=100
num_movies=len(df)

elems=[]
for i in range(0,num_movies,increments):
    elems.append(i)

def plotFunc(id):
    items=['languages','rating','plot outline']
    response = ia.get_movie((id))
    d={}
    for i in items:
        if i not in response.keys():
            d={}
            return d
        d[i]=response[i]
    return d

def getPlot(elem):
    print("started ", elem)
    for j in range(elem,elem+increments,1):
        id=df.loc[j,'imdbId']
        temp=plotFunc(id)    
        if len(temp)==0:
            continue;
        for k in temp.keys():
            if k=='languages':
                df.loc[j,k]=temp[k][0]
            else:
                df.loc[j,k]=temp[k]

import threading
threads = [threading.Thread(target=getPlot, args=(elem,)) for elem in elems]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

import numpy as np
items=['languages','rating','plot outline']
for i in items:
    df[i]=""

def cleanPlot(elem):
    print("started ", elem)
    for j in range(elem,elem+increments,1):
        id=df2.loc[j,'plot outline']
        temp=cleaning_function(id)[0]
        df2.loc[j,'plot']=temp
        
threads = [threading.Thread(target=fetch_url, args=(elem,)) for elem in elems]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

def plotAnalyse(ind):
    print("started",ind)
    for index in range(ind,ind+increments,1):
        s=df.loc[index,'plot outline']
        out=lexicon.analyze(s,normalize=True)
        for i in out.keys():
            df.loc[index,i]=out[i]

threads = [threading.Thread(target=plotAnalyse, args=(elem,)) for elem in elems]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.utils import lemmatize
stop_words = stopwords.words('english')

import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
appos = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not"
}

df2['valence']=0
df2['arousal']=0
df2['dominance']=0
df2['ones']=1

# VAD attributes file
file=r"NRC-VAD-Lexicon.txt"

data=pd.read_csv(file,encoding="utf-8",sep='\t')
words={}
for i,val in enumerate(data['Word']):
    words[val]=i

for index in range(0,len(df2)):
    s=df2.loc[index,'plot outline']
    if isinstance(s,float):
        print("here")
        continue
    cnt=0
    v=0
    a=0
    d=0
    for word in s.split():
        if word in words.keys():
#             print(word)
            v+=data.loc[words[word],'Valence']
            a+=data.loc[words[word],'Arousal']
            d+=data.loc[words[word],'Dominance']
            cnt+=1
    print(cnt)
    if cnt!=0:
        df2.loc[index,'valence']=v/cnt
        df2.loc[index,'arousal']=a/cnt
        df2.loc[index,'dominance']=d/cnt

df4=pd.pivot_table(df2,index='movieId',columns='languages',values='ones')
df3=df4.join(df2,on='movieId',how='inner',lsuffix='_caller', rsuffix='_other')
df3=df3.dropna(thresh=int(len(df3)*0.8), axis=1)
res=df2.groupby(['languages'])['movieId'].count()[lambda x: x >= 10]
df3=df3.replace(0, pd.np.nan)
df3=df3.dropna(thresh=int(len(df3)*0.1), axis=1)
df3=df3.replace(pd.np.nan,0)
langs=list(res.keys())
df3 = df2[df2['languages'].isin(langs)]
df3.to_csv("modified_links_Plot.csv",index=False)