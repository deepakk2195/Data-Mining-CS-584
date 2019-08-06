from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import re
import numpy as np
from collections import Counter
import glob
import os
from sklearn.metrics.pairwise import cosine_similarity

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
#lemma = PorterStemmer()
 
def cleanline(doc):#this function cleans the data by removing stopwords,punctuation and lemmatizing the data. This is done on both test and training data.
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    #normalized = " ".join(lemma.stem(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    #processed = re.sub(r"\d+","",punc_free)
    y = processed.split()
    return y

def listtodict(k,l):#This function assigns class labels and for a given 'k' value finds the frequency of 1's and 0's and prints accordingly.
    d={}
    k.sort()
    l.sort()

    for i in range(len(k)-1,len(k)-50,-1):#putting the cosine similarities of truthful into dictionary and its label
        d[k[i]]=0
    for i in range(len(l)-1,len(l)-50,-1):#putting the cosine similarities of deceptive into dictionary and its label
        d[l[i]]=1 
    d=sorted(d.items())#sorting the dictionary containing the cosine similarities and the class labels.
    #print(d)
    final=0
    for i in range(1,96):#here k is 95
        final=d[-i][1]+final
    #final=d[-1][1]+d[-2][1]+d[-3][1]+d[-4][1]+d[-5][1]+d[-6][1]+d[-7][1]+d[-8][1]+d[-9][1]+d[-10][1]+d[-11][1]+d[-12][1]+d[-13][1]+d[-14][1]+d[-15][1]+d[-16][1]+d[-17][1]+d[-18][1]+d[-19][1]#Here 'k' is '5'
    #print(final)
    if final>47:
        print('1')
        return(1)
    else:
        print('0')
        return(0)
        
    
directory='E:\CS584\Training'
truthful=[]
deceptive=[]
#reading data from training set and storing in truthful and deceptive lists respectively
for dirs in os.listdir(directory):
    for file in os.listdir(directory+"\\"+dirs):
        if file == "truthful":
                for f in os.listdir(directory+"\\"+dirs+"\\"+file):
                    temp=directory+"\\"+dirs+"\\"+file+"\\"+f
                    with open(temp, 'r') as filehandle:
                        t=filehandle.readlines()
                        for line in t:
                            line = line.strip()
                            cleaned = cleanline(line)
                            cleaned = ' '.join(cleaned)
                            truthful.append(cleaned)
        else:
                for f in os.listdir(directory+"\\"+dirs+"\\"+file):
                    temp=directory+"\\"+dirs+"\\"+file+"\\"+f
                    with open(temp, 'r') as filehandle:
                        t=filehandle.readlines()
                        for line in t:
                            line = line.strip()
                            cleaned = cleanline(line)
                            cleaned = ' '.join(cleaned)
                            deceptive.append(cleaned)



vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=5000)#Using TfIdf vectorizer with features of 5000
truthful_x = vectorizer.fit_transform(truthful)#vectorizing,fitting and transforming truthful list
#print(truthful_x)
print(np.shape(truthful_x))


##random.shuffle(truthful)                            
##d1=truthful[-2:]
##d2=truthful[:-2]
##vectorizer = TfidfVectorizer(stop_words='english',max_features=275)
##truthful_x = vectorizer.fit_transform(d2)
###print(truthful_x)
##print(np.shape(truthful_x))
##deceptive_x = vectorizer.fit_transform(deceptive)
##print(np.shape(deceptive_x))
##test = vectorizer.transform(d1)
##print(np.shape(test))
##k=cosine_similarity(test,truthful_x)
##l=cosine_similarity(test,deceptive_x)
###print(len(k))
###print(len(l))
##for i in range(0,len(k)):
##    fin=listtodict(k[i],l[i])

##   The above commented code is aprt of testing the training data.
##   I manually partitioned deceptive and truthful data and tested.
##


directory='E:\CS584\\test'#reading test data
test_clean_sentence =[]
#for f in os.listdir(directory):
for f in range(1,161):
    temp=directory+"\\"+str(f)+".txt"
    file=open(temp,'r')
    for line in file:
        line=line.strip()
        cleaned_test  = cleanline(line)
        cleaned = ' '.join(cleaned_test)
        cleaned = re.sub(r"\d+","",cleaned)
        test_clean_sentence.append(cleaned)

test = vectorizer.transform(test_clean_sentence)#vectorizing and transforming test data
print(np.shape(test))
k=cosine_similarity(test,truthful_x)#finding cosine similarity between truthful and test data
#print(k)
print(np.shape(k))
deceptive_x = vectorizer.fit_transform(deceptive)#vectorizing,fitting and transforming deceptive list
print(np.shape(deceptive_x))
#print(deceptive_x)
test = vectorizer.transform(test_clean_sentence)
print(np.shape(test))
l=cosine_similarity(test,deceptive_x)#finding cosine similarity between deceptive and test data
print(np.shape(l))
#print(l)

f=open("output1.txt","w")#writing output to'output' file
for i in range(0,len(k)):
    fin=listtodict(k[i],l[i])#sending the cosine similarities of truthful and deceptive data for each test data
    f.write(str(fin) + '\n')

f.close()

