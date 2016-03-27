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
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import cosine as cos_distance
import gensim
from nltk.data import find
from nltk.corpus import brown
import math
import logging
from nltk.corpus import opinion_lexicon
#nltk.download('sentiwordnet')

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
    return accuracy





def get_polarity_type(synset_name):
    swn_synset =  swn.senti_synset(synset_name)
    if not swn_synset:
        return None
    elif swn_synset.pos_score() > swn_synset.neg_score() and swn_synset.pos_score() > swn_synset.obj_score():
        return 1
    elif swn_synset.neg_score() > swn_synset.pos_score() and swn_synset.neg_score() > swn_synset.obj_score():
        return -1
    else:
        return 0



def second_lexicon(positive_seeds,negative_seeds):

    word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
    model = gensim.models.Word2Vec.load_word2vec_format(word2vec_sample, binary=False)
    positive_list=[]
    negative_list=[]

    for aword in model.vocab:
        score=0
        for pseed in positive_seeds:
            score+=model.similarity(aword, pseed)
        for nseed in negative_seeds:
            score-=model.similarity(aword,nseed)

        score=score/16.0
        if score>0.03:
            positive_list.append(aword)
        elif score<-0.03:
            negative_list.append(aword)

    return positive_list,negative_list






def get_BOW(text):
    BOW = {}
    for word in text:
        BOW[word.lower()] = BOW.get(word.lower(),0) + 1
    return BOW

def third_lexicon(positive_seeds,negative_seeds):
    positive_list=[]
    negative_list=[]

    all_dic={}
    seed_total_dic={}
    for fileid in brown.fileids():
        bow=get_BOW(brown.words(fileid))
        for aword in bow:
            all_dic[aword]=all_dic.get(aword,{})
            all_dic[aword]['word_count']=all_dic[aword].get('word_count',0)+1
            for pseed in positive_seeds:
                if pseed in bow:
                    all_dic[aword][pseed]=all_dic[aword].get(pseed,0)+1
            for nseed in negative_seeds:
                if nseed in bow:
                    all_dic[aword][nseed]=all_dic[aword].get(nseed,0)+1

        for pseed in positive_seeds:
            if pseed in bow:
                seed_total_dic[pseed]=seed_total_dic.get(pseed,0)+1
        for nseed in negative_seeds:
            if nseed in bow:
                seed_total_dic[nseed]=seed_total_dic.get(nseed,0)+1

    total_count=float(len(brown.fileids()))

    for aword in all_dic:
        score=0
        for pseed in positive_seeds:
            if all_dic[aword].get(pseed) != None:
                a_score=math.log((all_dic[aword][pseed]/total_count)/((all_dic[aword]['word_count']/total_count)*(seed_total_dic[pseed]/total_count)), 2)
                if a_score>0:
                    score+=a_score

        for nseed in negative_seeds:
            if all_dic[aword].get(nseed) != None:
                a_score=math.log((all_dic[aword][nseed]/total_count)/((all_dic[aword]['word_count']/total_count)*(seed_total_dic[nseed]/total_count)), 2)
                if a_score>0:
                    score-=a_score


        score=score/16.0


        if score>0.3:
            positive_list.append(aword)
        elif score<-0.3:
            negative_list.append(aword)

    return positive_list,negative_list

def calculate_percentage(manual,automatic):
    automatic=set(automatic)
    count=0
    for word in manual:
        if word in automatic:
            count+=1

    return float(count)/len(manual)

def my_polarity(tweet,pset,nset,):
    score=0
    for word in tweet:
        if word in pset:
            score+=1
        elif word in nset:
            score-=1
    if score>0:
        return 1
    elif score<0:
        return -1
    else:
        return 0

def accuracy_of_lexicon(tweets,labels,plist,nlist):
    count=0
    pset=set(plist)
    nset=set(nlist)

    for tweet,label in zip(tweets,labels):
        if label==my_polarity(tweet,pset,nset):
            count+=1

    return float(count)/len(labels)




trn_tweets,trn_labels=preprocess_file('train.json')


trn_feature_dicts=convert_to_feature_dicts(trn_tweets,True,1)


dev_tweets,dev_labels=preprocess_file('dev.json')


dev_feature_dicts=convert_to_feature_dicts(dev_tweets,True,0)

trn_feature_dicts,dev_feature_dicts=prepare_data(trn_feature_dicts,dev_feature_dicts)

clf=DecisionTreeClassifier()

print test(clf,trn_feature_dicts,trn_labels,dev_feature_dicts,dev_labels)
#do_multiple_10foldcrossvalidation(clf,trn_feature_dicts,trn_labels)

clf2=LogisticRegression()

print test(clf2,trn_feature_dicts,trn_labels,dev_feature_dicts,dev_labels)

clf3 = LogisticRegression(solver='lbfgs', multi_class='multinomial')

clf3.fit(trn_feature_dicts,trn_labels)





myLR=MyLogisticRegression(clf3.coef_, clf3.intercept_, clf3.classes_)

print myLR.predict(dev_feature_dicts)[0:10]




'''
positive_list1=[]
negative_list1=[]

count = 0
for synset in wn.all_synsets():
    count += 1
    if count % 1000 == 0:
        print count
    # count synset polarity for each lemma
    name=synset.name()


    polarity_type=get_polarity_type(name)
    if polarity_type is not None:
        if polarity_type ==1:
            positive_list1+=synset.lemma_names()
        elif polarity_type== -1:
            negative_list1+=synset.lemma_names()

print 'positive list negative list 1'
print positive_list1[0:10],negative_list1[0:10]


positive_seeds = ["good","nice","excellent","positive","fortunate","correct","superior","great"]
negative_seeds = ["bad","nasty","poor","negative","unfortunate","wrong","inferior","awful"]
positive_list2, negative_list2=second_lexicon(positive_seeds, negative_seeds)
print 'positive list negative list 2'
print positive_list2[0:10], negative_list2[0:10]


positive_list3, negative_list3=third_lexicon(positive_seeds, negative_seeds)
print 'positive list negative list 3'
print positive_list3[0:10], negative_list3[0:10]


positive_words = opinion_lexicon.positive()
negative_words = opinion_lexicon.negative()


percentage1p= calculate_percentage(positive_words,positive_list1)
percentage1n=calculate_percentage(negative_words,negative_list1)

percentage2p=calculate_percentage(positive_words, positive_list2)
percentage2n=calculate_percentage(negative_words, negative_list2)

percentage3p=calculate_percentage(positive_words, positive_list3)
percentage3n=calculate_percentage(negative_words, negative_list3)


print(percentage1p,percentage1n,percentage2p,percentage2n,percentage3p,percentage3n)
print 'accuracy of lexicon'
print accuracy_of_lexicon(dev_tweets,dev_labels,positive_words,negative_words)
print accuracy_of_lexicon(dev_tweets,dev_labels,positive_list1,negative_list1)
print accuracy_of_lexicon(dev_tweets,dev_labels,positive_list2,negative_list2)
print accuracy_of_lexicon(dev_tweets,dev_labels,positive_list3,negative_list3)
'''