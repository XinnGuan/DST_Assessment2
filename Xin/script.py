import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
import joblib
import pickle
import time
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import PCA
import xgboost as xgb

train=pd.read_csv()

def myprocess(thisdoc):
    return(thisdoc.strip('[]').replace("u'", '').replace("'", '').replace(' ', '').split(','))

train['text']=train['text'].map(myprocess) 

vector_size=100
window=3
min_count=3
workers=3
sg=1

w2vmodel=Word2Vec(df['text'],min_count=min_count,vector_size=vector_size,workers=workers,sg=sg)

word2vec_model_file = 'C:/Users/haile/OneDrive - University of Bristol/assessment2/data_cleaned_1' + 'word2vec_1' + '.model'
w2vmodel.save(word2vec_model_file)

w2v_model = Word2Vec.load(word2vec_model_file)

pretrained_path = "C:/Users/haile/OneDrive - University of Bristol/assessment2/glove_model2.txt"
model_2 = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)
model_2.save('C:/Users/haile/OneDrive - University of Bristol/assessment2/glove_model.model')

w2v_model.build_vocab([list(model_2.index_to_key)], update=True)

training_examples_count = w2v_model.corpus_count
w2v_model.train(df['text'],total_examples=training_examples_count, epochs=w2v_model.epochs)

emb_df = (
    pd.DataFrame(
        [w2v_model.wv.get_vector(str(n)) for n in w2vmodel.wv.key_to_index],
        index = w2vmodel.wv.key_to_index
    )
)

X_train=train['text']

y_train=train['sentiment']

def featureVecMethod(words, model, num_features):
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    index2word_set = set(model.wv.index_to_key)
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model.wv[word])
    
    if nwords ==0:
        featureVec = np.zeros(num_features)
    else:
        featureVec = np.divide(featureVec, nwords)

    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        if counter%1000 == 0:
            print("Comment %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
            
        counter = counter+1
        
    return reviewFeatureVecs

trainDataVecs = getAvgFeatureVecs(X_train['text'], w2v_model, 100)


def remove_unuseful_rows(trainDataVecs,y_train):
    list=[]
    y_train.reset_index()
    for i in range(trainDataVecs.shape[0]):
        if np.all(trainDataVecs[i,]==0):
            list.append(i)
    if len(list)!=0:
        trainDataVecs=np.delete(trainDataVecs,list,axis=0)
        y_train.drop(index=list,inplace=True)
    return trainDataVecs,y_train


trainDataVecs1,y_train1=remove_unuseful_rows(trainDataVecs,y_train)

param_grid = {
        'min_child_weight': [1, 5,10],
        'max_depth': [7,9],
        }

estimator = xgb.XGBClassifier(n_estimators=300, objective='binary:logistic',learning_rate =0.2, gamma=0.4,
    subsample=0.9, 
    colsample_bytree = 0.9,)











