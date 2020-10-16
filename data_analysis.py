# %%

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import json
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.feature_extraction import FeatureHasher

# %%
# Import settings
with open("settings.json", "r") as inFile:
    settings = json.load(inFile)
    
# Import data
data = pd.read_csv(settings['file']['data'])    

# Insight of the data 
data

# %%
# Data types 
data.dtypes

# Convert data types
data['PatientId'] = data['PatientId'].astype('category')
data['AppointmentID'] = data['AppointmentID'].astype('category')
data['Gender'] = data['Gender'] == 'F'
data['Gender'] = data['Gender'].astype('int')
data['TimeAppointment'] = data['ScheduledDay'].astype('datetime64[ns]').dt.hour
data['ScheduledDay'] = data['ScheduledDay'].astype('datetime64[ns]').dt.date
data['AppointmentDay'] = data['AppointmentDay'].astype('datetime64[ns]').dt.date
data['Age'] = data['Age'].astype('int')
data['Neighbourhood'] = data['Neighbourhood'].astype('category')
data['No-show'] = data['No-show'] == 'No'
data['No-show'] = data['No-show'].astype('int')


# %%
# Crete OHE of categorical variables 
Neigh_pd = pd.get_dummies(data['Neighbourhood'], prefix = 'NB')

# Hashing trick for Neighbourhood variable


# Create new variables 
data['IntervalDay'] = data['AppointmentDay'].sub(data['ScheduledDay'], axis =0).dt.days

# Create final dataframe 
data_rd = pd.concat([data[['Gender', 'Age', 'Scholarship', 'Hipertension', 
    'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'IntervalDay',
    'TimeAppointment', 'No-show']], Neigh_pd], axis = 1)
data_rd = data[['Gender', 'Age', 'Scholarship', 'Hipertension', 
    'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'IntervalDay',
    'TimeAppointment', 'No-show']]

# Show data frame 
data_rd.head()

# %% 

### RANDOM FOREST 

# If balance:
if settings['balance'] == True:

    # Get Show/No Show
    data_show = data_rd[data_rd['No-show'] == 0]
    data_NS = data_rd[data_rd['No-show'] == 1]

    # Get max length 
    min_L = min([len(data_show), len(data_NS)])

    # Get balanced subsets 
    data_show = data_show.sample(frac=min_L/len(data_show))
    data_NS = data_NS.sample(frac= min_L/len(data_NS))

    # Concate row-wise
    data_rd = pd.concat([data_show, data_NS]).sample(frac = 1)

# %%

# Get data
X, y = data_rd.drop(columns=['No-show']), data_rd['No-show']
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Create classifier 
classifier = RandomForestClassifier(n_estimators=500, 
    criterion="gini", 
    min_samples_split=2,
    min_samples_leaf=1
    )
classifier = AdaBoostClassifier(n_estimators=300, learning_rate=0.5)

classifier = GradientBoostingClassifier(learning_rate=0.1,   
    n_estimators= 500,
    subsample=1.0,
    min_samples_split=2,
    min_samples_leaf=2,
    max_depth=6,
    min_weight_fraction_leaf=0.1,
    verbose=1)

classifier.fit(X_train, y_train)

# Predict test score 
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
acc

# %%
prec = precision_score(y_test, y_pred)
prec 

# %%

# Compute ROC curve
fpr, tpr, thr = roc_curve(y_test, y_pred)

# Compute Area Under the Curve
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()





# %%
