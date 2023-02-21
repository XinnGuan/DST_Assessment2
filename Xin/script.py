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

train=pd.read_csv('data_cleaned_full_text_and_summaries')

def myprocess(thisdoc):
    return(thisdoc.strip('[]').replace("u'", '').replace("'", '').replace(' ', '').split(','))

train['text']=train['text'].map(myprocess) 

vector_size=100
window=3
min_count=3
workers=3
sg=1

w2vmodel=Word2Vec(train['text'],min_count=min_count,vector_size=vector_size,workers=workers,sg=sg)

#word2vec_model_file = 'C:/Users/haile/OneDrive - University of Bristol/assessment2/data_cleaned_1' + 'word2vec_1' + '.model'
#w2vmodel.save(word2vec_model_file)

w2v_model = w2vmodel

pretrained_path = "glove_model2.txt"
model_2 = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)
#model_2.save('C:/Users/haile/OneDrive - University of Bristol/assessment2/glove_model.model')

w2v_model.build_vocab([list(model_2.index_to_key)], update=True)

training_examples_count = w2v_model.corpus_count
w2v_model.train(train['text'],total_examples=training_examples_count, epochs=w2v_model.epochs)

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

scale_pos_weight = (sum(y_train1==0))/(sum(y_train1==1))
print(scale_pos_weight)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    n_estimators=150)

start_time = time.time()
xgb_model.fit(trainDataVecs1, y_train1)


X_test=pd.read_csv("test_cleaned_text_and_summaries")
X_test['text']=X_test['text'].map(myprocess)

testDataVecs = getAvgFeatureVecs(X_test['text'], w2v_model, 100)

y_test=pd.read_csv("test_sentiments_50k.csv")

testDataVecs1,y_test1=remove_unuseful_rows(testDataVecs,y_test['sentiment'])
xgb_prediction=xgb_model.predict(testDataVecs1)
print(classification_report(y_test1,xgb_prediction))


y_pred = [x[1] for x in xgb_model.predict_proba(testDataVecs1)]
fpr, tpr, thresholds = roc_curve(y_test1, y_pred, pos_label = 1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize = (5, 3))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

df=xgb_model.predict_proba(testDataVecs1)
df=pd.DataFrame(df)
df.to_csv('summary_proba_results.csv')






