# アンサンブル学習

'''
ariable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex
Age	Age in years
sibsp	# of siblings / spouses aboard the Titanic
parch	# of parents / children aboard the Titanic
ticket	Ticket number
fare	Passenger fare
cabin	Cabin number
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
'''
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier

'''
def predict(df_key, df_data, classifier, subsets):
    stdsc = StandardScaler()
    df_std = stdsc.fit_transform(df_data)
    df_predict = pd.DataFrame(classifier.predict(df_std[:, subsets]))
    df_result = pd.concat([df_key, df_predict], axis=1)
    df_result.rename(columns={0: 'Survived'}, inplace=True)

    return df_result
'''

class MultivoteClassifier:
    '''
        アンサンブル分類器。
        ◆初期変数
            ・clf -> 分類器(array)

        ◆fit(grid)
            ・grid -> チューニングするパラメータ(array)
                      ※clfの順番と合わせる
            ・各分類器のベストパラメータを検索し、fitさせる。

        ◆predict(X, y)
            ・与えられたテストデータに対して、答えを返す。
            return: キー+答え
    '''
    def __init__(self, clf, test_size=0.25):
        self.clf = clf

    def fit(self, X, y, param_grid, test_size=0.25):
        self.best_clf = []
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.25, random_state=1)
        for i, clf, grid in zip(range(len(self.clf)), self.clf, param_grid):
            gs = GridSearchCV(estimator=clf,
                              param_grid=grid,
                              scoring='accuracy',
                              cv=5,
                              n_jobs=1)
            gs = gs.fit(X_train, y_train)
#            print("i:", i, "results:", gs.cv_results_)
            self.best_clf.append(gs.best_estimator_)
            self.best_clf[i].fit(X_train, y_train)
            print("i:", i, " estimator:", gs.best_estimator_)
            print("i:", i, " score:", gs.best_score_)
            print("i:", i, " score:", self.best_clf[i].score(X_test, y_test))

        # voting=hard: 2値分類する。 (※voting=soft: 確率を割り出す(らしい))
        best_clf_vote = []
        for i, clf in zip(range(len(self.best_clf)), self.best_clf):
            best_clf_vote.append((i, clf))
            

        self.voteclf_ = VotingClassifier(estimators=best_clf_vote, voting='hard', weights=None, n_jobs=1)
        self.voteclf_.fit(X_train, y_train)

        print("VoteScore\n")
        print(self.voteclf_.score(X_train, y_train))
        print(self.voteclf_.score(X_test, y_test))
        return self

    def predict(self, X, key):
        predict = pd.DataFrame(self.voteclf_.predict(X))
        result = pd.concat([predict, key], axis=1)
        result.rename(columns={0: 'Survived'}, inplace=True)
        return result

def edit_data(df, train_flg=False, test_flg=False):
    if train_flg:
        df_ret = pd.get_dummies(df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin']]).fillna(0)
    if test_flg:
        df_ret = pd.get_dummies(df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin']]).fillna(0)

    df_ageisnull = []
    for age in df['Age']:
        if age == 0:
            df_ageisnull.append(1)
        else:
            df_ageisnull.append(0)

    df_ret["Ageisnull"] = df_ageisnull

    return df_ret


#1.前処理
#   データの取得
df_train = pd.read_csv("dataset/train.csv")
df_test = pd.read_csv("dataset/test.csv")
#print(df)
# 共通の編集
df_train['Cabin'] = df_train['Cabin'].where(df_train['Cabin'].isnull(), 1)
df_test['Cabin'] = df_test['Cabin'].where(df_test['Cabin'].isnull(), 1)
# 分類器の設定
clf_xgb = xgb.XGBClassifier()

clf = [clf_xgb]

# チューニング値の設定
params = {'learning_rate': [0.01, 0.05, 0.1, 0.2],
          'max_depth': [2, 3, 4, 5, 6],
          'subsample': [0.9, 0.95],
          'colsample_bytree': [0.5, 1.0]
          }

param_grid = [paramｓ]

# 年齢ありデータ

df_train_edited = edit_data(df_train, train_flg=True)
X, y = df_train_edited.iloc[:, 1:].values, df_train_edited.iloc[:, 0].values
mlvote_age = MultivoteClassifier(clf)
mlvote_age.fit(X, y, param_grid)

df_test_edited = edit_data(df_test, test_flg=True)
df_key = pd.get_dummies(df_test[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin']])
df_key = df_key['PassengerId']
result = mlvote_age.predict(df_test_edited.values, df_key)

result = result.sort_values(by='PassengerId')
result.to_csv('result/xgb.csv')
