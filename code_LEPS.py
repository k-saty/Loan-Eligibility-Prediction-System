import numpy as np 
import pandas as pd  
import seaborn as sns 
import matplotlib.pyplot as plt  
%matplotlib inline
train_data = pd.read_csv("dataset/train_u6lujuX_CVtuZ9i.csv") 
train_data.head()
print(train_data.shape)
train_data.describe()
train_data.info()
def missing_values(df): 
    a = num_null_values = df.isnull().sum()  
    return a
missing_values(train_data)
train_data.drop(["Loan_ID","Dependents"], axis=1, inplace=True)
train_data
# Dealing with null values [ categorical ]

cols = train_data[["Gender", "Married", "Self_Employed"]] 
for  i in cols: 
    train_data[i].fillna(train_data[i].mode().iloc[0], inplace=True)
train_data.isnull().sum()
# Dealing with Numerical Values missig_data 

n_cols = train_data[["LoanAmount", "Loan_Amount_Term", "Credit_History"]] 
for i in n_cols: 
    train_data[i].fillna(train_data[i].mean(axis=0), inplace=True)
# Visualization
def bar_chart(col): 
    Approved = train_data[train_data["Loan_Status"]=="Y"][col].value_counts() 
    Disapproved = train_data[train_data["Loan_Status"]=="N"][col].value_counts() 
    
    df1 = pd.DataFrame([Approved, Disapproved]) 
    df1.index = ["Approved", "Disapproved"] 
    df1.plot(kind="bar")
bar_chart("Gender")
bar_chart("Married")
bar_chart("Education")
bar_chart("Self_Employed")
train_data.head()
from sklearn.preprocessing import OrdinalEncoder 

ord_enc = OrdinalEncoder() 
train_data[["Gender",'Married','Education','Self_Employed','Property_Area','Loan_Status']] = ord_enc.fit_transform(train_data[["Gender",'Married','Education','Self_Employed','Property_Area','Loan_Status']])
train_data.head()
# Accessing the categories_ attribute
categories_mapping = {}
for i, column in enumerate(["Gender", 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']):
    categories_mapping[column] = list(ord_enc.categories_[i])

print("Categories Mapping:")
print(categories_mapping)
train_data[["Gender",'Married','Education','Self_Employed','Property_Area','Loan_Status']] = train_data[["Gender",'Married','Education','Self_Employed','Property_Area','Loan_Status']].astype('int')
train_data
from sklearn.model_selection import train_test_split  
X = train_data.drop("Loan_Status", axis=1) 
y = train_data["Loan_Status"] 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2) 
print(X_train.shape) 
print(y_train.shape)
print(X_test.shape) 
print(y_test.shape)
X_train

from sklearn.naive_bayes import GaussianNB 

gfc = GaussianNB() 
gfc.fit(X_train, y_train) 
pred1 = gfc.predict(X_test)
from sklearn.metrics import precision_score, recall_score, accuracy_score 

def loss(y_true, y_pred): 
    pre=  precision_score(y_true, y_pred) 
    rec = recall_score(y_true, y_pred) 
    acc = accuracy_score(y_true, y_pred) 
    
    print(f"precision {pre}") 
    print(f"recall {rec}") 
    print(f"accuracy {acc}")
loss(y_test, pred1)
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, pred1)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
scaler = StandardScaler()



# Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
pred1 = model_lr.predict(X_test)

# Decision Tree Classifier
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
pred2 = model_dt.predict(X_test)

# Random Forest Classifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
pred3 = model_rf.predict(X_test)

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# K-Nearest Neighbors Classifier
param_grid_knn = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(X_train_sc, y_train)
best_knn = grid_knn.best_estimator_
pred4 = best_knn.predict(X_test_sc)


print("Logistic Regression Accuracy:", accuracy_score(y_test, pred1))
print("Decision Tree Accuracy:", accuracy_score(y_test, pred2))
print("Random Forest Accuracy:", accuracy_score(y_test, pred3))
print("K-Nearest Neighbors Accuracy:", accuracy_score(y_test, pred4))
import joblib 
joblib.dump(gfc, "model.pkl") 
model = joblib.load('model.pkl' ) 
model.predict(X_test)
X_train.iloc[[0]]
y_train.iloc[[0]]


input_data = {
    'Gender': 1,
    'Married': 0,
    'Education': 0,
    'Self_Employed': 0,
    'ApplicantIncome': 2971,
    'CoapplicantIncome': 2791.0,
    'LoanAmount': 144.0,
    'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0,
    'Property_Area': 1
}

input_df = pd.DataFrame([input_data])


prediction = gfc.predict(input_df) 

# Print the prediction
print("Prediction:", prediction)

%history -f code_LEPS.py
