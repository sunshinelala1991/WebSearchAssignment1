import json
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from MyLogesticRegression import MyLogisticRegression
#nltk.download('words')

word_set=set(nltk.corpus.words.words())
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
sent_segmenter = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

def preprocess(tweet):


    #print(tweet)

    tweet = re.sub("@[^ ]+", "", tweet).strip()
    tweet=re.sub("http[^ ]+","",tweet).strip()
    hashtag = re.findall('#[^ ]*', tweet)

    tag_list=[]
    if hashtag:
        #print hashtag,' hashtag----'
        for tag in hashtag:
            tag=re.sub('#','',tag)
            #print tag,' tag----'
            # if contains capital letters
            if re.search('.*[A-Z].*',tag):
                tag_list=tag_list+re.findall('[A-Z][^A-Z]*',tag)

            else:
                 # if not contain capital letters
                i=0
                #print tag,'tag'
                while i<len(tag):
                    for j in range(len(tag),i,-1):
                        lemma=lemmatize(tag[i:j])

                        if lemma in word_set:
                            #print(lemma)
                            tag_list.append(tag[i:j])
                            i=j-1
                    i+=1
    if tag_list:
        for i in range(len(tag_list)):
            tag_list[i]=tag_list[i].lower()
        #print tag_list



    tweet = re.sub('#[^ ]*','',tweet).lower()

    tweet = sent_segmenter.tokenize(tweet)

    words = []
    for sentence in tweet:
        words= words+ word_tokenizer.tokenize(sentence)

    if tag_list:
        words=words+tag_list
    #print words




    return words

def preprocess_file(filename):
    tweets = []
    labels=[]
    f = open(filename)
    for line in f:
        tweet_dict = json.loads(line)
        #print tweet_dict
        tweets.append(preprocess(tweet_dict["text"]))
        labels.append(int(tweet_dict["label"]))
    print 'done preprocessing'
    return tweets,labels



def lemmatize(word):

    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def convert_to_feature_dicts(tweets,remove_stop_words,n):
    feature_dicts = []
    if remove_stop_words:
        stop_word_set=set(stopwords.words('english'))
    if n>0:
        small_set=set()
        whole_feature_dic={}
        for tweet in tweets:
            for w in tweet:
                whole_feature_dic[w]=whole_feature_dic.get(w,0)+1

        for w in whole_feature_dic:
            if whole_feature_dic[w]<=n:
                small_set.add(w)



    for tweet in tweets:
        # build feature dictionary for tweet
        feature_dict={}
        for w in tweet:


            if remove_stop_words and n<=0:
                if w not in stop_word_set:
                    feature_dict[w]=feature_dict.get(w,0)+1
            elif remove_stop_words and n>0:
                if w not in stop_word_set and w not in small_set:
                    feature_dict[w]=feature_dict.get(w,0)+1
            elif n>0:
                if w not in small_set:
                    feature_dict[w]=feature_dict.get(w,0)+1
            else:
                feature_dict[w]=feature_dict.get(w,0)+1

        feature_dicts.append(feature_dict)

    #print feature_dicts
    return feature_dicts


def prepare_data(trn_feature_dicts,dev_feature_dicts):
    vectorizer = DictVectorizer()
    trn_feature_dicts = vectorizer.fit_transform(trn_feature_dicts)
    dev_feature_dicts = vectorizer.transform(dev_feature_dicts)
    return trn_feature_dicts,dev_feature_dicts



def do_multiple_10foldcrossvalidation(clf,data,classifications):
    predictions = cross_validation.cross_val_predict(clf, data,classifications, cv=10)
    print clf
    print "accuracy"
    print accuracy_score(classifications,predictions)
    print classification_report(classifications,predictions)


def test(clf,training_data,training_classifications,test_data,test_classifications):

    clf.fit(training_data,training_classifications)
    predictions = clf.predict(test_data)
    accuracy = accuracy_score(test_classifications,predictions)
    print accuracy



trn_tweets,trn_labels=preprocess_file('train.json')

trn_feature_dicts=convert_to_feature_dicts(trn_tweets,True,1)

dev_tweets,dev_labels=preprocess_file('dev.json')
dev_feature_dicts=convert_to_feature_dicts(dev_tweets,True,0)
#print 'shit' in set(nltk.corpus.words.words())
trn_feature_dicts,dev_feature_dicts=prepare_data(trn_feature_dicts,dev_feature_dicts)

clf=DecisionTreeClassifier()

test(clf,trn_feature_dicts,trn_labels,dev_feature_dicts,dev_labels)
#do_multiple_10foldcrossvalidation(clf,trn_feature_dicts,trn_labels)

clf2=LogisticRegression()

test(clf2,trn_feature_dicts,trn_labels,dev_feature_dicts,dev_labels)

clf3 = LogisticRegression(solver='lbfgs', multi_class='multinomial')

clf3.fit(trn_feature_dicts,trn_labels)



print clf3.predict(dev_feature_dicts)[0:10]

print trn_feature_dicts.shape,len(trn_labels)
print type(trn_feature_dicts)

for coef in clf3.coef_:
    print coef,len(coef)

print clf3.intercept_

print clf3.classes_


myLR=MyLogisticRegression(clf3.coef_, clf3.intercept_, clf3.classes_)

print myLR.predict(dev_feature_dicts)[0:10]