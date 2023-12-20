import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_selection import RFECV
from keras.layers import Input

data = pd.read_excel('Electric Vehicles.xls')

print(data.info())
print(data.columns)

print(data.isnull().sum())

print("Duplicated value:")
print(data.duplicated().value_counts())
# Check for duplicate rows based on all columns
duplicate_rows = data[data.duplicated()]
# Display duplicate rows
print("Duplicate Rows:")
print(duplicate_rows)

data.drop_duplicates(inplace=True)
print("Duplicated value:")
print(data.duplicated().value_counts())

# Categorical columns
cat_col = [col for col in data.columns if data[col].dtype == 'object']
print('Categorical columns :', cat_col)
# Numerical columns
num_col = [col for col in data.columns if data[col].dtype != 'object']
print('Numerical columns :', num_col)

data.replace('?', np.nan, inplace=True)

for col in data.columns:
    missing_count = data[col].isnull().sum()
    print(f"Number of '?' in '{col}' column: {missing_count}")

selected_columns = data.loc[:, 'Q14':'Q18_23']
print(selected_columns)

mode_values = selected_columns.mode().iloc[0]

for col in selected_columns.columns:
    data[col].fillna(mode_values[col], inplace=True)

print(data)
print(data.isnull().values.any())

print(data.info())

# Categorical columns
cat_col = [col for col in data.columns if data[col].dtype == 'object']
print('Categorical columns :', cat_col)
# Numerical columns
num_col = [col for col in data.columns if data[col].dtype != 'object']
print('Numerical columns :', num_col)

label_encoder = LabelEncoder()
data['Q16_new'] = label_encoder.fit_transform(data['Q16'])

print(data.info())


new_data = data
new_data = new_data.drop("Q16", axis=1)
print(new_data.info())

for column in new_data.columns:
    new_data[column] = new_data[column].astype("float64")

print(new_data.info())
print(new_data)

# threshold = 3
# columns_to_convert = new_data.loc[:, 'Q14':'Q21'].columns
# new_data[columns_to_convert] = new_data[columns_to_convert].map(lambda x: 1 if x >= threshold else 0)
# new_data[columns_to_convert] = new_data[columns_to_convert].astype("float64")

new_data['Q16_new'] = new_data['Q16_new'].map(lambda x: 1 if x >= 1.5 else 0)
new_data['Q16_new'] = new_data['Q16_new'].astype("float64")

X = new_data.drop('Q16_new', axis=1)
y = new_data['Q16_new']

# Logistic Regression

stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=56)

auc_scores = []
accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
mcc_scores = []

for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = LogisticRegression(C=2, max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    auc_scores.append(auc)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)
    mcc_scores.append(mcc)

print("Logistic Regression: ")
print(f"AUC Mean: {sum(auc_scores) / len(auc_scores)}")
print(f"Accuracy Mean: {sum(accuracy_scores) / len(accuracy_scores)}")
print(f"F1 Mean: {sum(f1_scores) / len(f1_scores)}")
print(f"Precision Mean: {sum(precision_scores) / len(precision_scores)}")
print(f"Recall Mean: {sum(recall_scores) / len(recall_scores)}")
print(f"MCC Mean: {sum(mcc_scores) / len(mcc_scores)}")

# Support Vector Machine

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

roc_auc = metrics.roc_auc_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
mcc = metrics.matthews_corrcoef(y_test, y_pred)

print()
print("Support Vector Machine: ")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("AUC:", roc_auc)
print("F1 Score:", f1)
print("MCC:", mcc)

# Neural Network

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

classifier = Sequential()

classifier.add(Input(shape=(X_train.shape[1],)))

classifier.add(Dense(units=64, kernel_initializer="uniform", activation='relu'))
classifier.add(Dense(units=32, kernel_initializer="uniform", activation='relu'))
classifier.add(Dense(units=1, kernel_initializer="uniform", activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=8, validation_split=0.2, verbose=2)

predictions = classifier.predict(X_test)

roc_auc = roc_auc_score(y_test, predictions)

accuracy = accuracy_score(y_test, (predictions > 0.5).astype(int))

f1 = f1_score(y_test, (predictions > 0.5).astype(int))

precision = precision_score(y_test, (predictions > 0.5).astype(int))

recall = recall_score(y_test, (predictions > 0.5).astype(int))

mcc = matthews_corrcoef(y_test, (predictions > 0.5).astype(int))

print("Neural Network: ")
print("AUC:", roc_auc)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("MCC:", mcc)



# Models with Feature Selection

# Feature Selection Random Forest

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# The "accuracy" scoring is proportional to the number of correct classifications
rf = RandomForestClassifier()
rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy')
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])

cv_results = rfecv.cv_results_

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(range(1, len(cv_results['mean_test_score']) + 1), cv_results['mean_test_score'])
plt.show()

selected_columns1 = ['Q11_1', 'Q11_2', 'Q11_3', 'Q11_4', 'Q11_5', 'Q11_7', 'Q11_8', 'Q11_10',
       'Q11_11', 'Q11_Dont_Know', 'Q12_1', 'Q12_2', 'Q12_3', 'Q12_4', 'Q12_5',
       'Q12_7', 'Q12_8', 'Q12_10', 'Q12_11', 'Q12_14', 'Q12_Dont_Know',
       'Q13_1', 'Q13_2', 'Q13_3', 'Q13_5', 'Q13_6', 'Q13_7', 'Q13_8', 'Q13_9',
       'Q13_10', 'Q13_11', 'Q13_12', 'Q13_13', 'Q13_14', 'Q13_16',
       'Q13_Dont_Know', 'Q14', 'Q15', 'Q17', 'Q18_1', 'Q18_2', 'Q18_3',
       'Q18_4', 'Q18_5', 'Q18_6', 'Q18_7', 'Q18_8', 'Q18_9', 'Q18_10',
       'Q18_11', 'Q18_12', 'Q18_13', 'Q18_14', 'Q18_15', 'Q18_16', 'Q18_17',
       'Q18_18', 'Q18_19', 'Q18_20', 'Q18_21', 'Q18_22', 'Q18_23', 'Q20',
       'Q21','Q16_new']

# selected_columns1 = ['Q11_1', 'Q11_2', 'Q11_4', 'Q11_5', 'Q11_7', 'Q11_8', 'Q11_10',
#        'Q11_11', 'Q11_Dont_Know', 'Q12_1', 'Q12_2', 'Q12_3', 'Q12_4', 'Q12_5',
#        'Q12_7', 'Q12_10', 'Q12_14', 'Q13_1', 'Q13_2', 'Q13_3', 'Q13_5',
#        'Q13_6', 'Q13_7', 'Q13_9', 'Q13_11', 'Q13_13', 'Q13_14', 'Q13_16',
#        'Q13_Dont_Know', 'Q14', 'Q15', 'Q17', 'Q18_1', 'Q18_2', 'Q18_3',
#        'Q18_4', 'Q18_5', 'Q18_6', 'Q18_7', 'Q18_8', 'Q18_9', 'Q18_10',
#        'Q18_11', 'Q18_12', 'Q18_13', 'Q18_14', 'Q18_15', 'Q18_16', 'Q18_17',
#        'Q18_18', 'Q18_19', 'Q18_20', 'Q18_21', 'Q18_22', 'Q18_23', 'Q20',
#        'Q21','Q16_new']


# #Feature Selection Logistic Regression
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# logreg_model = LogisticRegression(C=2, max_iter=700)
# rfecv_logreg = RFECV(estimator=logreg_model, step=1, cv=5, scoring='accuracy')
# rfecv_logreg = rfecv_logreg.fit(X_train, y_train)
#
# print('Optimal number of features (Logistic Regression):', rfecv_logreg.n_features_)
# print('Best features (Logistic Regression):', X_train.columns[rfecv_logreg.support_])

# selected_columns1 = ['Q11_3', 'Q11_Dont_Know', 'Q12_14', 'Q12_Dont_Know', 'Q13_7', 'Q13_15',
#       'Q13_16', 'Q13_Dont_Know', 'Q15', 'Q17', 'Q21','Q16_new']

# selected_columns1 =['Q11_1', 'Q11_3', 'Q11_4', 'Q11_5', 'Q11_6', 'Q11_7', 'Q11_9', 'Q11_11',
#        'Q11_12', 'Q11_13', 'Q11_Dont_Know', 'Q12_2', 'Q12_3', 'Q12_4', 'Q12_5',
#        'Q12_6', 'Q12_8', 'Q12_9', 'Q12_11', 'Q12_12', 'Q12_13', 'Q12_14',
#        'Q12_Dont_Know', 'Q13_4', 'Q13_7', 'Q13_8', 'Q13_10', 'Q13_11',
#        'Q13_12', 'Q13_14', 'Q13_15', 'Q13_16', 'Q13_Dont_Know', 'Q14', 'Q15',
#        'Q17', 'Q18_1', 'Q18_3', 'Q18_5', 'Q18_6', 'Q18_9', 'Q18_11', 'Q18_14',
#        'Q18_22', 'Q18_23', 'Q21','Q16_new']


# #Feature Selection Support Vector Machine
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# svm_model = SVC(kernel='linear')
# rfecv_svm = RFECV(estimator=svm_model, step=1, cv=5, scoring='accuracy')
# rfecv_svm = rfecv_svm.fit(X_train, y_train)
#
# print('Optimal number of features (SVM):', rfecv_svm.n_features_)
# print('Best features (SVM):', X_train.columns[rfecv_svm.support_])


selected_data = new_data[selected_columns1]
print(selected_data)

X = selected_data.drop('Q16_new', axis=1)
y = selected_data['Q16_new']

#Logistic Regression with Feature Selection

print("\nLogistic Regression with Feature Selection")

stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=56)

auc_scores = []
accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
mcc_scores = []

for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = LogisticRegression(C=2,max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    auc_scores.append(auc)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)
    mcc_scores.append(mcc)

print(f"AUC Mean: {sum(auc_scores) / len(auc_scores)}")
print(f"Accuracy Mean: {sum(accuracy_scores) / len(accuracy_scores)}")
print(f"F1 Mean: {sum(f1_scores) / len(f1_scores)}")
print(f"Precision Mean: {sum(precision_scores) / len(precision_scores)}")
print(f"Recall Mean: {sum(recall_scores) / len(recall_scores)}")
print(f"MCC Mean: {sum(mcc_scores) / len(mcc_scores)}")

#Support Vector Machine with Feature Selection

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

roc_auc = metrics.roc_auc_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
mcc = metrics.matthews_corrcoef(y_test, y_pred)

print()
print("Support Vector Machine with Feature Selection: ")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("AUC:", roc_auc)
print("F1 Score:", f1)
print("MCC:", mcc)

#Neural Network with Feature Selection

print("\nNeural Network with Feature Selection: ")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

classifier = Sequential()

classifier.add(Input(shape=(X_train.shape[1],)))

classifier.add(Dense(units=64, kernel_initializer="uniform", activation='relu'))
classifier.add(Dense(units=32, kernel_initializer="uniform", activation='relu'))
classifier.add(Dense(units=1, kernel_initializer="uniform", activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=8,validation_data=(X_test, y_test),verbose=2)

predictions = classifier.predict(X_test)

roc_auc = roc_auc_score(y_test, predictions)

accuracy = accuracy_score(y_test, (predictions > 0.5).astype(int))

f1 = f1_score(y_test, (predictions > 0.5).astype(int))

precision = precision_score(y_test, (predictions > 0.5).astype(int))

recall = recall_score(y_test, (predictions > 0.5).astype(int))

mcc = matthews_corrcoef(y_test, (predictions > 0.5).astype(int))

print("AUC:", roc_auc)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("MCC:", mcc)
