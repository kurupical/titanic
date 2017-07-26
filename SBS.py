# 逐次後退選択(Sequential Backward Selection)によりデータ分析を行う

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
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X_train, X_test, y_train, y_test):
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)

            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)

        return score

def fit_sbs(X, y, classifier):
    '''
    X : テストデータ
    y : 教師データ
    classifier : 分類器のオブジェクト
    '''
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=1)
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)
    sbs = SBS(classifier, k_features=1)
    sbs.fit(X_train_std, X_test_std, y_train, y_test)

    # スコアが最大になるindexを検索(同値がある場合は要素数少ないほう)
    maxscore_index = max([i for i, x in enumerate(sbs.scores_) if x == max(sbs.scores_)])


    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.6, 0.9])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.show()

    classifier.fit(X_train_std[:, sbs.subsets_[maxscore_index]], y_train)

    return classifier, sbs.subsets_[maxscore_index]

def predict(df_key, df_data, classifier, subsets):
    stdsc = StandardScaler()
    df_std = stdsc.fit_transform(df_data)
    df_predict = pd.DataFrame(classifier.predict(df_std[:, subsets]))
    df_result = pd.concat([df_key, df_predict], axis=1)
    df_result.rename(columns={0: 'Survived'}, inplace=True)

    return df_result


#1.前処理
#   データの取得
df_train = pd.read_csv("dataset/train.csv")
df_test = pd.read_csv("dataset/test.csv")
#print(df)
#　共通の編集
df_train['Cabin'] = df_train['Cabin'].where(df_train['Cabin'].isnull(), 1)
#   データ解析
lr_age = LogisticRegression(penalty='l2', C=0.1)
lr_noage = LogisticRegression(penalty='l2', C=0.1)


#   年齢データありの学習モデル、年齢データなしの学習モデル　の２つを作成。

df_agedata = pd.get_dummies(df_train[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin']]).dropna().reset_index(drop=True)
X, y = df_agedata.iloc[:, 1:].values, df_agedata.iloc[:, 0].values
print(df_agedata)
lr_agedata, maxscore_subsets_agedata = fit_sbs(X, y, lr_age)

df_noagedata = pd.get_dummies(df_train[['Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked','Cabin']])
X, y = df_noagedata.iloc[:, 1:].values, df_noagedata.iloc[:, 0].values
lr_noagedata, maxscore_subsets_noagedata = fit_sbs(X, y, lr_noage)

# データ推測
df_key = pd.get_dummies(df_test[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin']]).dropna(subset=['Age']).reset_index(drop=True)
df_key = df_key['PassengerId']
df_test_agedata = pd.get_dummies(df_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin']]).dropna(subset=['Age']).reset_index(drop=True)
df_test_agedata = df_test_agedata.fillna(df_test_agedata['Fare'].mean()) # Fareがnullのレコードが1個だけあるので平均値を埋めておく
df_result_agedata = predict(df_key, df_test_agedata, lr_agedata, maxscore_subsets_agedata)

df_test_noagedata = df_test.ix[df_test["Age"].isnull()]
df_key = pd.get_dummies(df_test_noagedata[['PassengerId','Pclass','Sex','SibSp','Parch','Fare','Embarked','Cabin']]).reset_index(drop=True)
df_key = df_key['PassengerId']
df_test_noagedata = pd.get_dummies(df_test_noagedata[['Pclass','Sex','SibSp','Parch','Fare','Embarked','Cabin']]).reset_index(drop=True)
df_result_noagedata = predict(df_key, df_test_noagedata, lr_noagedata, maxscore_subsets_noagedata)

df_result = pd.concat([df_result_agedata.loc[:,['PassengerId','Survived']], df_result_noagedata.loc[:, ['PassengerId','Survived']]])
df_result = df_result.sort_values(by='PassengerId')
df_result.to_csv('result/sbs.csv')



#2.解析処理
#   モデル作成
