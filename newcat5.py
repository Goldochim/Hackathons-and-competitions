import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import catboost as cat_
import pandas as pd
from sklearn.model_selection import StratifiedKFold

#loading the datasets
train= pd.read_csv('Train loan defaults.csv')
test=pd.read_csv('Test loan defalts.csv')
sample_sub=pd.read_csv('SampleSubmission loan defaults.csv')

#replacing empty cells with an outlier 
train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)

#converting text data to numerical data in the train dataset
def handle_non_numerical_data (train):
    columns=train.columns.values
    
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if train[column].dtype !=np.int64 and train[column].dtype!=np.float64:
            column_contents=train[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1                    
            train[column]=list(map(convert_to_int, train[column]))
    return train
train=handle_non_numerical_data(train)

#converting text data to numerical data in the test dataset
def handle_non_numerical_data (test):
    columns=test.columns.values
    
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if test[column].dtype !=np.int64 and test[column].dtype!=np.float64:
            column_contents=test[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1
                    
            test[column]=list(map(convert_to_int, test[column]))
    return test
test=handle_non_numerical_data(test)

#dropping off columns not making a difference the model
train.drop(['Applicant_ID'],1, inplace=True)
test.drop(['Applicant_ID'],1, inplace=True)
train.drop(['form_field49'], 1, inplace=True)
test.drop(['form_field49'], 1, inplace=True)
train.drop(['form_field48'], 1, inplace=True)
test.drop(['form_field48'], 1, inplace=True)

#creating the target and attribute dataset and preprocessing the attribute dataset
x=np.array(train.drop(['default_status'], 1))
x=preprocessing.scale(x)
y=np.array(train['default_status'])

#applying the model on the general dataset; target and attribute
clf= cat_.CatBoostClassifier(n_estimators=5000, max_depth=6, eval_metric='AUC', reg_lambda=370)
clf=clf.fit(x, y, use_best_model=True)
#splitiing the model into training and testing
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20)
 
#applying the stratified kfold cross validation
skf=StratifiedKFold(n_splits=2)
skf.get_n_splits(x, y)    
model=clf.fit(x_train, y_train)
for train_index, test_index in skf.split(x,y):
  print("TRAIN:", train_index, "TEST:", test_index)
  x_train, x_test= x[train_index], x[test_index]
  y_train, y_test= y[train_index], y[test_index]

#getting the score of the model
accuracy=clf.score(x_test,y_test)

#applying the model to prediction
example_measures=np.array(test)
example_measures=example_measures.reshape(len(example_measures), -1)

sample_sub=clf.predict_proba(example_measures)


#setting the result for conversion to csv
submission=pd.read_csv("SampleSubmission loan defaults.csv")
submission['default_status']=sample_sub

#saving the file to csv using the  sample submission format

submission.to_csv('new5catboost.csv', index=False)















