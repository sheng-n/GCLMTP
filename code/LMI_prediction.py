from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn import svm
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



'''Read dataset1'''
def read_file1():

    train_id = np.loadtxt("data/mi_lnc_train_id1.txt")
    test_id = np.loadtxt("data/mi_lnc_test_id1.txt")
    neg_id = np.loadtxt("data/MLI_negtive_id.txt")

    low_A = np.loadtxt("dataset1_result/low_A_256.txt")

    mi_lnc = np.loadtxt("data/yuguoxian_lnc_mi.txt").T

    mi_feature = low_A[645: ]
    lnc_feature = low_A[ :240]
    return train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, neg_id

'''Read  dataset2'''
def read_file2():

    train_id = np.loadtxt("dataset2/mi_lnc_train_id1.txt")
    test_id = np.loadtxt("dataset2/mi_lnc_test_id1.txt")
    neg_id = np.loadtxt("dataset2/MLI_negtive_id.txt")

    low_A = np.loadtxt("dataset2_result/low_A_256.txt")

    mi_lnc = pd.read_csv('dataset2/mi_lnc_intersection.csv', index_col='Unnamed: 0')
    mi_lnc = mi_lnc.values

    mi_feature = low_A[981: ]
    # print(mi_feature.shape)
    lnc_feature = low_A[:665]
    return train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, neg_id

def get_feature(A_feature, B_feature, index, adi_matrix):
    input = []
    output = []
    for i in range(index.shape[0]):
        A_i = int(index[i][0])
        B_j = int(index[i][1])
        feature = np.hstack((A_feature[A_i], B_feature[B_j]))
        input.append(feature.tolist())
        label = adi_matrix[[A_i],[B_j]].tolist()
        # print(type(label))
        # label = label.tolist()
        # print(label)
        output.append(label)
    output = np.array(output)
    output = output.ravel()
    return np.array(input), output


'''miRNA-lncRNA'''
# train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, negtive_id = read_file1()
train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, negtive_id = read_file2()
train_input, train_output = get_feature(mi_feature, lnc_feature, train_id, mi_lnc)  # (2328, dim)
test_input, test_output = get_feature(mi_feature, lnc_feature,test_id, mi_lnc)
case_study_input,case_study_output = get_feature(mi_feature,lnc_feature,negtive_id,mi_lnc)


# --------------------------------- Exploring the performance of different classifiers------------------------------
'''AdaBoost'''
flag = 0
# flag = 1
if flag:
    ada = AdaBoostClassifier(n_estimators=40)
    ada.fit(train_input,train_output)
    y_pred = ada.predict_proba(test_input)[:,1]
    print(y_pred)

'''XGBoost'''
flag = 0
# flag = 1
if flag:
    xgb = XGBClassifier(n_estimators = 200, eta = 0.1, max_depth = 10)
    xgb.fit(train_input,train_output)
    y_pred = xgb.predict_proba(test_input)[:,1]
    print(y_pred)


'''LigheGBM'''
flag = 0
# flag = 1
if flag:
    lgb_train = lgb.Dataset(train_input, train_output)
    lgb_eval = lgb.Dataset(test_input, test_output, reference=lgb_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'max_depth': 4,
        'metric': {'l2', 'auc'},
        'is_unbalance': 'true',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round = 100,
                    valid_sets = lgb_eval,
                    early_stopping_rounds = 300)
    y_pred = gbm.predict(test_input, num_iteration = gbm.best_iteration)



'''RandomForest'''
# flag = 0
flag = 1
if flag:

    rf = RandomForestClassifier(n_estimators = 500, max_depth = 7)
    rf.fit(train_input,train_output)
    y_pred = rf.predict_proba(test_input)[:,1]
    print(y_pred)


'''MLP'''
flag = 0
# flag = 1
if flag:
    mlp = MLPClassifier(solver='adam',hidden_layer_sizes=(512,2),activation='relu', learning_rate_init=0.0001,max_iter=1000)
    mlp.fit(train_input,train_output)
    y_pred = mlp.predict_proba(test_input)[:,1]
    print(y_pred)


'''GBDT'''
flag = 0
# flag = 1
if flag:

    gbdt = GradientBoostingClassifier(n_estimators=100,max_depth = 10)
    gbdt.fit(train_input, train_output)

    y_pred = gbdt.predict_proba(test_input)[:,1]
    print(y_pred)


