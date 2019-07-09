import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
from sklearn import preprocessing

#Loading the data into a dataframe
dataset = pd.read_csv('C:/Users/Administrator/Desktop/Code/Churn_Modelling.csv')

#Encoding String Column Values
ls=preprocessing.LabelEncoder()
dataset['Geography']=ls.fit_transform(dataset['Geography'])
dataset['Gender']=ls.fit_transform(dataset['Gender'])

#Finding Correlation among different columns
print(dataset.corr())

#Plotting the heatmap for getting a better visual of correlation
import seaborn as sns
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,15))
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Selecting the KBest features using SelectKBest 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X=dataset[['RowNumber', 'CustomerId', 'CreditScore', 'Geography','Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard','IsActiveMember', 'EstimatedSalary']]
Y=dataset[['Exited']]   
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#deriving the feature importance of each column
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(20,'Score'))

#dropping unnecessary columns
dataset.drop(['CustomerId','Surname','Geography'],axis=1,inplace=True)

#Plotting the graphs between a column and the column "Exited" to visualize any relation possible
import matplotlib.pyplot as plt
import seaborn as sns
print("Graph between 'Number of products' and 'Exited'")
sns.barplot(x="NumOfProducts",y='Exited',data=dataset)
plt.show()
print("Graph between 'Tenure' and 'Exited'")
sns.barplot(x="Tenure",y='Exited',data=dataset)
plt.show()
print("Graph between 'Has Cr Card' and 'Exited'")
sns.barplot(x="HasCrCard",y='Exited',data=dataset)
plt.show()
print("Graph between 'Is active member' and 'Exited'")
sns.barplot(x="IsActiveMember",y='Exited',data=dataset)
plt.show()
dataset.drop(['HasCrCard'],axis=1,inplace=True)
print("Graph between 'Credit Score' and 'Exited'")
sns.barplot(x="CreditScore",y='Exited',data=dataset)
plt.show()
print("Graph between 'Gender' and 'Exited'")
sns.barplot(x="Gender",y='Exited',data=dataset)
plt.show()

#dropping less weightage columns
dataset.drop(['RowNumber','Tenure'],axis=1,inplace=True)

#Splitting the dataset into X and Y corresponding to INPUT and OUTPUT respectively
X=dataset[['CreditScore', 'Gender', 'Age', 'Balance', 'NumOfProducts',
       'IsActiveMember', 'EstimatedSalary']]
Y=dataset[['Exited']]

#Again Splitting into train and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Applying SVM and finding out result
clf = svm.SVC(kernel='rbf', gamma=100)
clf.fit(X_train, Y_train)
clf.predict(X_test)
print("Applying Support Vector Machine we get:-")
print("Accuracy Score :",accuracy_score(Y_test,clf.predict(X_test))*100)
from sklearn.metrics import classification_report
print("Confusion Matrix :",confusion_matrix(Y_test,clf.predict(X_test))) 
print("\n")

# Applying KNN and finding out result
print("Applying KNN we get:-")
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 2) 
knn.fit(X_train, Y_train) 
pred = knn.predict(X_test)  
from sklearn.metrics import confusion_matrix 
print("Confusion Matrix :",confusion_matrix(Y_test, pred)) 
print(classification_report(Y_test, pred)) 
print ("Accuracy : ", accuracy_score(Y_test, pred)*100)
print("\n")

# Applying Logistic Regression and finding out result
print("Applying Logistic Regression we get:-")
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test, y_pred) 
print ("Confusion Matrix : \n", cm)
print ("Accuracy : ", accuracy_score(Y_test, y_pred)*100)
print("\n")

# Applying Gaussian and finding out result
print("Applying Gaussian we get")
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, Y_train)  
y_pred = gnb.predict(X_test) 
print ("Accuracy : ", accuracy_score(Y_test, y_pred)*100)
gm = confusion_matrix(Y_test, y_pred) 
print ("Confusion Matrix : \n", gm)
print("\n")

# Applying Decision Tree and finding out result
print("Applying Decision Tree we get:-")
from sklearn.tree import DecisionTreeClassifier 
clf= DecisionTreeClassifier(random_state =0,max_depth=3, min_samples_leaf=3) 
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test) 
print("Accuracy Score :",accuracy_score(Y_test,y_pred)*100)
print("Confusion Matrix :",confusion_matrix(Y_test,clf.predict(X_test))) 
print("\n")

# Applying Random Forest Classifier and finding out result
print("Applying Random Forest Classifier we get:-")
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=4)
clf.fit(X_test, Y_test)
predictions = clf.predict(X_test)
print(accuracy_score(Y_test,predictions)*100)
print("Confusion Matrix \n",confusion_matrix(Y_test,clf.predict(X_test))) 
print("\n")

# Applying Gradient Boosting Classifier and finding out result
print("Applying Gradient Boosting Classifier we get:-")
from sklearn.ensemble import GradientBoostingClassifier
scikit_model = GradientBoostingClassifier(random_state = 1)
scikit_model.fit(X_test, Y_test)
predictions = scikit_model.predict(X_test)
print("Accuracy Score :",accuracy_score(Y_test,predictions)*100)
print("Confusion Matrix \n",confusion_matrix(Y_test,clf.predict(X_test))) 

print("So for our model Decision Tree worked well with accuracy score of 84.25")
print("I have choosen Decision Tree because neither it was overfitting data nor the accuracy was too low")