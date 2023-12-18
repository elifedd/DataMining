import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

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
    new_data[column] = new_data[column].astype("int64")


print(new_data.info())
print(new_data)

threshold = 3
columns_to_convert = new_data.loc[:, 'Q14':'Q21'].columns
new_data[columns_to_convert] = new_data[columns_to_convert].map(lambda x: 1 if x >= threshold else 0)
new_data['Q16_new'] = new_data['Q16_new'].map(lambda x: 1 if x >= 1.5 else 0)

X = new_data.drop('Q16_new', axis=1)
y = new_data['Q16_new']

X = new_data.drop('Q16_new', axis=1)
y = new_data['Q16_new']

# Convert all columns to int64
for column in new_data.columns:
    new_data[column] = new_data[column].astype("int64")

# Split the data into features (X) and target variable (y)
X = new_data.drop('Q16_new', axis=1)
y = new_data['Q16_new']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize the Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Fit the classifier on the training data
nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Calculate additional metrics
roc_auc = metrics.roc_auc_score(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)
mcc = metrics.matthews_corrcoef(y_test, y_pred)

# Print the metrics
print(f"AUC: {roc_auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"MCC: {mcc}")
