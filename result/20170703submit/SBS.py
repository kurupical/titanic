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

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        print("test:" ,self.subsets_)
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]
        print("test:" , self.scores_)

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


#1.前処理
#   データの取得
df = pd.read_csv("dataset/train.csv")
df_test = pd.read_csv("dataset/test.csv")
#print(df)
#   データの編集
#   passenger_id,name,ticket,cabinは関係なさそうなので要素を削除
df = pd.get_dummies(df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])
print(df.isnull().sum())
df = df.dropna() # Age=Nanになっているデータを取り除く
#   データをtest/trainに分割する(※再現性あるようにする)
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.25, random_state=1)# random_stateはシード値。固定値を指定することで毎回同じ分割になる。

#   データのスケーリング(平均=0,標準偏差=1)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#2.解析処理
#   モデル作成
lr = LogisticRegression(penalty='l2', C=0.1)
sbs = SBS(lr, k_features=1)
X_std = stdsc.fit_transform(X)
sbs.fit(X_std, y)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.6, 0.9])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
# plt.show()

k5 = list(sbs.subsets_[2])
print(df.columns[1:][k5])
lr.fit(X_train_std, y_train)
print('training accuracy:', lr.score(X_train_std, y_train))
print('test accuracy:', lr.score(X_test_std, y_test))

lr.fit(X_train_std[:, k5], y_train)
print('training accuracy(k5):', lr.score(X_train_std[:, k5], y_train))
print('test accuracy(k5):', lr.score(X_test_std[:, k5], y_test))
# print(df_test)
df_test_std = pd.get_dummies(df_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])
df_test_std = df_test_std.fillna(df_test_std['Age'].mean())
print(df_test_std)
df_test_std = stdsc.fit_transform(df_test_std)
#print(df_test_std)
df_test_predict = pd.DataFrame(lr.predict(df_test_std[:, k5]))
df_test_std = pd.DataFrame(df_test_std)

df_result = pd.concat([df_test_predict, df_test], axis=1)
df_result.rename(columns={0: 'Survived'}, inplace=True)
print(df_result)
df_result.to_csv('result/sbs.csv')
