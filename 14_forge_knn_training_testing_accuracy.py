#code provided to students: text p. 41
from sklearn.datasets        import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors       import KNeighborsClassifier
from matplotlib import pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target,
        stratify=cancer.target, random_state = 66)
training_accuracy = []
testing_accuracy  = []

for n in range(1, 11):
    clf = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    testing_accuracy.append(clf.score(X_test, y_test))
plt.plot(range(1, 11), training_accuracy, label='training')
plt.plot(range(1, 11), testing_accuracy,  label='testing')
plt.legend()

