import pandas as pd
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

def run():
    df = pd.read_csv('sepsis1.csv')
    df = subset(df)
    decision_tree_model(df)
    #print df

def subset(df_combined):
    df1 = df_combined[df_combined.columns[0:25]] #select first elements up and until diagnose
    return df1

def decision_tree_model(df_combined):
    #create feature set and target set
    features = df_combined.drop('diagnose', axis=1)
    target = df_combined.loc[:,'diagnose']
    features_train, features_test, target_train, target_test = train_test_split(features,
                                                                                target, test_size = 0.20, random_state = 10)
    #Decision Tree with Gini IndexPython

    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=10, min_samples_leaf=5)

    clf_gini.fit(features_train, target_train)
    prediction = clf_gini.predict(features_test)

    acc_score = accuracy_score(target_test,prediction,normalize=True)

    print "Accuracy of Decision Tree for",gender,"and",classifier," is", acc_score

run()
