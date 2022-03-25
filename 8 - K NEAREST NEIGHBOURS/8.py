from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Load dataset
iris=datasets.load_iris()
print("Iris Data set loaded...")


# Split the data into train and test samples
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.1)
print("Dataset is split into training and testing...")
print("Size of training data and its label",x_train.shape,y_train.shape)
print("Size of testing data and its label",x_test.shape, y_test.shape)


# Prints Label no. and their names
for i in range(len(iris.target_names)):
    print("Label", i , "-",str(iris.target_names[i]))
    
    
# Create object of KNN classifier
model = KNeighborsClassifier(n_neighbors=1)

# Perform Training
model.fit(x_train, y_train)

# Perform testing
y_pred=model.predict(x_test)


# Display the results
print("Results of Classification using K-nn with K=1 ")
for i in range(len(x_test)):
    print(" Sample:", str(x_test[i]), " Actual-label:", str(y_test[i]), " Predicted-label:", str(y_pred[i]))
print("Classification Accuracy :" , model.score(x_test,y_test));
print('Confusion Matrix :\n',confusion_matrix(y_test,y_pred))
print('Accuracy Metrics : \n',classification_report(y_test,y_pred))
