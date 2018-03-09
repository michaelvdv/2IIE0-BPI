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
    #count = df.groupby('duration').count()
    #print count
    #print df['category']
    #print df

def condition(row):
    if row['duration'] == 0:
        val = 0
    elif 0 < row['duration'] < 5:
        val = 1
    elif 4 < row['duration'] < 15:
        val = 2
    else:
        val = 3
    return val


def subset(df_combined):
    #df1 = df_combined.reset_index()
    #df1.columns[0] = 'New_ID'
    #df1['New_ID'] = df1.index + 10000
    df_combined.insert(0,'New_ID',range(10000, 10000 + len(df_combined)))
    #df1 = df_combined[df_combined.columns[0:26]] #select first elements up and until diagnose
    #selected = ['Age', 'Admission IC', 'Admission NC', 'CRP', 'ER Registration', 'ER Triage', 'IV Antibiotics', 'IV Liquid', 'LacticAcid', 'Leucocytes', 'Release A', 'Release B', 'Release C', 'Release D', 'Release E', 'Return ER', 'duration', 'total_nr_events']
    selected = ['case_id', 'DiagnosticArtAstrup', 'InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie', 'SIRSCritHeartRate', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie', 'Hypoxie', 'DiagnosticUrinarySediment', 'DiagnosticECG', 'DiagnosticLacticAcid', 'Infusion', 'Age', 'Diagnose']
    df1 = df_combined[selected]
    df1['category'] = df1.apply(condition, axis=1)
    return df1

def decision_tree_model(df_combined):
    #df_combined = df_combined.drop("case_id", axis=1)
    #create feature set and target set
    features = df_combined.drop('duration', axis=1).drop('category', axis=1)
    target = df_combined.loc[:,'category']
    features_train, features_test, target_train, target_test = train_test_split(features,
                                                                                target, test_size = 0.20, random_state = 0)
    #Decision Tree with Gini IndexPython

    #build naive bayes model, based on normal distribution
    gnb = GaussianNB()
    gnb.fit(features_train,target_train)
    prediction = gnb.predict(features_test)

    acc_score = accuracy_score(target_test,prediction,normalize=True)

    print "Accuracy of Naive Bayes is", acc_score

run()
