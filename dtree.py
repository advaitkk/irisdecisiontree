"""
Created on Sun Jul 26 13:09:09 2020
@author: Advait
"""
import sklearn.datasets as datasets
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

# Loading dataset
iris=datasets.load_iris()

#dataframe
df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head(5))

y=iris.target
#print(y)

#decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(df,y)
print('Decision Tree Classifer Created')

#PLOT 
from sklearn.tree import plot_tree
model_all_params = DecisionTreeClassifier().fit(iris.data, iris.target)
plt.figure(figsize = (20,10)) #set size
plot_tree(model_all_params, 
          filled=True      )
plt.show()

#accuracy
y_pred = dtree.predict(df)
print('\nAccuracy: {0:.4f}'.format(accuracy_score(y, y_pred)))