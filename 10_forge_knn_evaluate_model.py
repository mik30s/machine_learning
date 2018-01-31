#code provided to students: text p. 39
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors       import KNeighborsClassifier

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state = 0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print clf.predict(X_test)
print clf.score(X_test, y_test)

