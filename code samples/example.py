# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# load the iris datasets
dataset = datasets.load_iris()
# fit a Naive Bayes model to the data
model = GaussianNB()
# features
X = dataset.data
# labels
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.33, random_state=42)
model.fit(X_train, y_train)
print dataset.data
print dataset.target
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(X_test)
print len(predicted)
print len (y_test)
print accuracy_score(y_test, predicted)
# summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))

