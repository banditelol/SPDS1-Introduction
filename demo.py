from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors

DTclf = tree.DecisionTreeClassifier()
SVMclf = svm.LinearSVC()
SGDclf = linear_model.SGDClassifier()
KNNclf = neighbors.KNeighborsClassifier()
# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female']

X_test =  [[159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y_test = ['female', 'male', 'male']

# CHALLENGE - ...and train them on our data
DTclf = DTclf.fit(X, Y)
SVMclf = SVMclf.fit(X, Y)
SGDclf = SGDclf.fit(X, Y)
KNNclf = KNNclf.fit(X, Y)


DTprediction = DTclf.predict(X_test)
SVMprediction = SVMclf.predict(X_test)
SGDprediction = SGDclf.predict(X_test)
KNNprediction = KNNclf.predict(X_test)

# CHALLENGE compare their reusults and print the best one!

print("\nDecision Tree : ")
print("Prediction : " + str(DTprediction))
print("Accuracy Score : " + str(accuracy_score(Y_test, DTprediction)))
print("\nLinear SVC : ") 
print("Prediction : " + str(SVMprediction))
print("Accuracy Score : " + str(accuracy_score(Y_test, SVMprediction)))
print("\nStochastic Gradient Descent : ") 
print("Prediction : " + str(SGDprediction))
print("Accuracy Score : " + str(accuracy_score(Y_test, SGDprediction)))
print("\nK-Nearest Neighbors : ") 
print("Prediction : " + str(KNNprediction))
print("Accuracy Score : " + str(accuracy_score(Y_test, KNNprediction)))