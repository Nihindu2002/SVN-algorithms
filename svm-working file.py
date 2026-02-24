import sklearn
from sklearn import datasets
from sklearn import svm

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)


print(x_train,y_train)

"""malignant(1) -  that grow uncontrollably, invade nearby tissues, and can spread to other parts of the body
benign (0) = a non-cancerous growth that does not spread"""

classes = ['benign', 'malignant']

