from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt



#lab_name = sys.argv[1]
lab_name = 'pharmavision'
df = pd.read_excel('../../resource/' + lab_name + '/data/comments_score.xlsx')
df = df.dropna()
X = df['clean_comment']
y = df['score']
dict = {}

##Support Vector Machine :

### Count Vectorizer :

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# Create a Pipeline
CV_Support_vector_machine = make_pipeline(CountVectorizer(), svm.SVC())

# Training the model
CV_Support_vector_machine.fit(X_train, y_train)

# Predicting the labels for test data
y_pred_sv = CV_Support_vector_machine.predict(X_test)

# accuracy_score
accSvm = accuracy_score(y_test, y_pred_sv)
dict[accSvm] = CV_Support_vector_machine
# f1_score
f1 = f1_score(y_test, y_pred_sv, average='macro')

# Affichage
print("Results for Support Vector Machine with CountVectorizer")
print("accuracy : %f f1_measure : %f " % (accSvm, f1))

### TFIDF Vectorizer :

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)
# Create a Pipeline
TFidf_Support_vector_machine = make_pipeline(TfidfVectorizer(), svm.SVC(kernel='rbf'))

# Training the model
TFidf_Support_vector_machine.fit(X_train, y_train)

# Predicting the labels for test data
y_pred_sv = TFidf_Support_vector_machine.predict(X_test)

#evaluation

# accuracy_score
accSvm1 = accuracy_score(y_test, y_pred_sv)
print(accSvm1)
# f1_score
f1 = f1_score(y_test, y_pred_sv, average='macro')
print(f1)

#classification_report
report = classification_report(y_test, y_pred_sv)
print(report)

#confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_sv)
sns.set_palette(sns.color_palette("Reds"))
# plot the heatmap
sns.heatmap(confusion_matrix, annot=True)
print(confusion_matrix)


dict[accSvm1] = TFidf_Support_vector_machine


# Affichage
print("Results for Support Vector Machine with tfidf")
print("accuracy : %f f1_measure : %f " % (accSvm1, f1))



# Model saving
model = dict[max([accSvm, accSvm1, accLr, accLr1])]
model_path = '../../resource/' + lab_name + '/models/comment/comment_rating3'
pickle.dump(model, open(model_path, 'wb'))
