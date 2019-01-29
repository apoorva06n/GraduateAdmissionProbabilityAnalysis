import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
	#import data into a dataframe and set serial number as index
	df_profile = pd.read_csv("./Admission_Predict_Ver1.1.csv",index_col=0)
	df_profile.index.name = None
	sns.set()
	#display heatmap to check is there are any null values in the dataset
	sns.heatmap(df_profile.isnull(),yticklabels=False,cbar=False,cmap='viridis')
	plt.title("Heat map to display any null values")
	plt.show()
	df_profile['GRE Score'].hist(bins=30,color='darkred',alpha=0.7)
	plt.title('Histogram to show GRE Score distribution')
	plt.show()
	sns.countplot(x='University Rating',data=df_profile,palette='rainbow')
	plt.title("University rating count plot")
	plt.show()
	#clean the data by getting rid of continuous probable values and adding categorial values
	df_profile['Admission Chance'] = df_profile[['Chance of Admit ']].apply(admission_probability,axis=1)
	print(df_profile.head())
	new_df = df_profile.drop('Chance of Admit ',axis=1)
	print(new_df.head())
	#divide the data set into training and test data set and train the model
	df_profile.drop('Admission Chance',axis=1,inplace=True)
	Logistic_Reg_Module(new_df.drop('Admission Chance',axis=1),new_df['Admission Chance'])
	Linear_Reg_Module(df_profile.drop('Chance of Admit ',axis=1),df_profile['Chance of Admit '])
	SVM(new_df.drop('Admission Chance',axis=1),new_df['Admission Chance'])
	decision_tree(new_df.drop('Admission Chance',axis=1),new_df['Admission Chance'])
	random_forest(new_df.drop('Admission Chance',axis=1),new_df['Admission Chance'])

def Logistic_Reg_Module(X,y):
	X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.3)
	print("----------- Logistic Regression Prediction ------------")
	LR_model = LogisticRegression()
	LR_model.fit(X_train,y_train)
	#predict the test data based on training set
	predictions = LR_model.predict(X_test)
	#print the evaluation report
	print("\n---------- Prediction evaluation using Confusion Matrix -----------\n")
	print(confusion_matrix(y_test,predictions))
	print("\n---------- Prediction evaluation using Classification Report -----------\n")
	print(classification_report(y_test,predictions))

def Linear_Reg_Module(X,y):	
	X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.3)
	print("----------- Linear Regression Prediction ------------")
	lm = LinearRegression()
	lm.fit(X_train,y_train)
	lm_pred = lm.predict(X_test)
	print("MAE :",metrics.mean_absolute_error(y_test,lm_pred))
	print("MSE :",metrics.mean_squared_error(y_test,lm_pred))
	print("RMSE :",np.sqrt(metrics.mean_squared_error(y_test,lm_pred)))
	sns.distplot((y_test - lm_pred ),bins=50)
	plt.show()

def SVM(X,y):
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
	print("----------- SVM Prediction ------------")
	param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}
	#Create a grid object and fit it to the training data
	grid_obj= GridSearchCV(SVC(),param_grid,verbose=2)
	grid_obj.fit(X_train,y_train)
	pred = grid_obj.predict(X_test)
	print("\n---------- Prediction evaluation using Confusion Matrix -----------\n")
	print(confusion_matrix(y_test,pred))
	print("\n---------- Prediction evaluation using Classification Report -----------\n")
	print(classification_report(y_test,pred))

def decision_tree(X,y):
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
	print("--------- Deicision Tree Classifier -------")
	#train decision tree classifier
	dtree = DecisionTreeClassifier()
	dtree.fit(X_train,y_train)
	#predict results
	predictions = dtree.predict(X_test)
	#print evaluation results
	print("\n---------- Prediction evaluation using Confusion Matrix -----------\n")
	print(confusion_matrix(y_test,predictions))
	print("\n---------- Prediction evaluation using Classification Report -----------\n")
	print(classification_report(y_test,predictions))

def random_forest(X,y):
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
	#train random forest classifier
	print("----- Random Forest Classifier -----")
	rfClass = RandomForestClassifier()
	rfClass.fit(X_train,y_train)
	#predict results on test data
	pred_rfc = rfClass.predict(X_test)
	#print evaluation results
	print("\n---------- Prediction evaluation using Confusion Matrix -----------\n")
	print(confusion_matrix(y_test,pred_rfc))
	print("\n---------- Prediction evaluation using Classification Report -----------\n")
	print(classification_report(y_test,pred_rfc))


def admission_probability(cols):
	prob_val = cols[0]
	if prob_val <= 0.50:
		return 0
	elif prob_val <=0.75: 
		return 1
	else: return 2

if __name__ == '__main__':
	main()