import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef,accuracy_score

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

threshold = 3
columns_to_convert = new_data.loc[:, 'Q14':'Q21'].columns
new_data[columns_to_convert] = new_data[columns_to_convert].map(lambda x: 1 if x >= threshold else 0)
new_data[columns_to_convert] = new_data[columns_to_convert].astype("float64")

new_data['Q16_new'] = new_data['Q16_new'].map(lambda x: 1 if x >= 1.5 else 0)
new_data['Q16_new'] = new_data['Q16_new'].astype("float64")

X = new_data.drop('Q16_new', axis=1)
y = new_data['Q16_new']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

classifier = Sequential()

classifier.add(Dense(kernel_initializer="uniform", activation='relu',units=64, input_dim=X_train.shape[1]))

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
