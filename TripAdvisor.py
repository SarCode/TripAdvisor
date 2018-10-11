# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Trip.csv')

dataset.replace({"3,5":np.NaN}, inplace=True)
dataset.replace({np.NaN:3}, inplace=True)

dataset.replace({"4,5":np.NaN}, inplace=True)
dataset.replace({np.NaN:4}, inplace=True)

dataset=dataset.drop(columns=['User country','User continent','Review month','Review weekday','Period of stay'])
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_2 = LabelEncoder()

X.iloc[:, 3] = labelencoder_X_2.fit_transform(X.iloc[:, 3])
X.iloc[:, 4] = labelencoder_X_2.fit_transform(X.iloc[:, 4])
X.iloc[:, 5] = labelencoder_X_2.fit_transform(X.iloc[:, 5])
X.iloc[:, 6] = labelencoder_X_2.fit_transform(X.iloc[:, 6])
X.iloc[:, 7] = labelencoder_X_2.fit_transform(X.iloc[:, 7])
X.iloc[:, 8] = labelencoder_X_2.fit_transform(X.iloc[:, 8])
X.iloc[:, 9] = labelencoder_X_2.fit_transform(X.iloc[:, 9])
X.iloc[:, 10] = labelencoder_X_2.fit_transform(X.iloc[:, 10])



onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###################################################################

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

x=cm.sum()
n=5
sum_diagonal=sum([cm[i][j] for i in range(n) for j in range(n) if i==j]) 

accuracy=sum_diagonal/x

print("Accuracy by KNN : ",accuracy*100)