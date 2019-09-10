from sklearn import datasets
import pandas as pd;
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

data = pd.read_csv('BRAC36002e.csv')
data.head()

filtered_csv = pd.read_csv('BRAC36002e.csv',
usecols=['mouse_x',
         'mouse_y',
         'left_ear_x',
         'left_ear_y',
         'right_ear_x',
         'right_ear_y',
         'left_shoulder_x',
         'left_shoulder_y',
         'right_shoulder_x',
         'right_shoulder_y',
         'left_back_leg_x',
         'left_back_leg_y',
         'right_back_leg_x',
         'right_back_leg_y'])

roi = pd.read_csv('BRAC36002e.csv',
usecols = ['pupA_head_x',
           'pupA_head_y',
           'pupA_body_x',
           'pupA_body_y',
           'pupB_head_x',
           'pupB_head_y',
           'pupB_body_x',
           'pupB_body_y'])

X = data[['mouse_x',
        'left_ear_x',
        'right_ear_x',
        'left_shoulder_x',
        'right_shoulder_x',
        'left_back_leg_x',
        'right_back_leg_x']]

Y = data[['mouse_y',
          'left_ear_y',
          'right_ear_y',
          'left_shoulder_y',
          'right_shoulder_y',
          'left_back_leg_y',
          'right_back_leg_y']]

X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.3)
clf = RandomForestClassifier(n_estimator = 100)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print('Accuracy:'metrics.accuracy.score(Y_test,Y_pred))
