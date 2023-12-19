import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

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

data.drop_duplicates(inplace= True)
print("Duplicated value:")
print(data.duplicated().value_counts())

# Categorical columns
cat_col = [col for col in data.columns if data[col].dtype == 'object']
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in data.columns if data[col].dtype != 'object']
print('Numerical columns :',num_col)

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
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in data.columns if data[col].dtype != 'object']
print('Numerical columns :',num_col)


label_encoder = LabelEncoder()
data['Q16_new'] = label_encoder.fit_transform(data['Q16'])

print(data.info())

new_data = data
new_data = new_data.drop("Q16", axis=1)
print(new_data.info())

for column in new_data.columns:
    new_data[column]=new_data[column].astype("int64")


print(new_data.info())
print(new_data)

# Feature Selection
#
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.feature_selection import RFECV
#
# # Özellikler ve etiketleri ayırma
# X = new_data.drop('Q16_new', axis=1)
# y = new_data['Q16_new']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # The "accuracy" scoring is proportional to the number of correct classifications
# rf = RandomForestClassifier()
# rfecv = RFECV(estimator=rf, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
# rfecv = rfecv.fit(X_train, y_train)
#
# print('Optimal number of features :', rfecv.n_features_)
# print('Best features :', X_train.columns[rfecv.support_])
#
# cv_results = rfecv.cv_results_
#
# # Grafikleme
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (accuracy)")
# plt.plot(range(1, len(cv_results['mean_test_score']) + 1), cv_results['mean_test_score'])
# plt.show()

selected_columns1 = ['Q11_1', 'Q11_2', 'Q11_4', 'Q11_5', 'Q11_7', 'Q11_8', 'Q11_10',
       'Q11_11', 'Q11_Dont_Know', 'Q12_1', 'Q12_2', 'Q12_3', 'Q12_4', 'Q12_5',
       'Q12_7', 'Q12_10', 'Q12_14', 'Q13_1', 'Q13_2', 'Q13_3', 'Q13_5',
       'Q13_6', 'Q13_7', 'Q13_9', 'Q13_11', 'Q13_13', 'Q13_14', 'Q13_16',
       'Q13_Dont_Know', 'Q14', 'Q15', 'Q17', 'Q18_1', 'Q18_2', 'Q18_3',
       'Q18_4', 'Q18_5', 'Q18_6', 'Q18_7', 'Q18_8', 'Q18_9', 'Q18_10',
       'Q18_11', 'Q18_12', 'Q18_13', 'Q18_14', 'Q18_15', 'Q18_16', 'Q18_17',
       'Q18_18', 'Q18_19', 'Q18_20', 'Q18_21', 'Q18_22', 'Q18_23', 'Q20',
       'Q21','Q16_new']

threshold = 3

selected_data = new_data[selected_columns1]
columns_to_convert = selected_data.loc[:, 'Q14':'Q21'].columns
selected_data['Q16_new'] = selected_data['Q16_new'].map(lambda x: 1 if x >= 1.5 else 0)
print(selected_data)

X = selected_data.drop('Q16_new', axis=1)
y = selected_data['Q16_new']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))


roc_auc = metrics.roc_auc_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
mcc = metrics.matthews_corrcoef(y_test, y_pred)

print("AUC:", roc_auc)
print("F1 Score:", f1)
print("MCC:", mcc)