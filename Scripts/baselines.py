import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os

#os.chdir('OneDrive/PhD/TephraDataBase/Scripts')
from funciones import simbologia

np.random.seed(0)

# Data preprocessing
####################

# Load the data
df = pd.read_excel('../Data/TephraDataBase.xlsx')

# 1 - Convert geochemical columns to float type

# Identify the non numerical entries in the geochemical columns
# tmp = df.loc[:, 'SiO2':'U']
# for j in range(tmp.shape[1]):
#     print(j)
#     pd.to_numeric(tmp.iloc[:, j], errors = 'raise')

# Pb line 4410 for variable Fe0: what is 10..3?
df = df.replace(to_replace='10..3', value=np.nan)
df = df.replace(to_replace='Not analyzed', value=np.nan)
df = df.replace(to_replace='-', value=np.nan)
df = df.replace(to_replace='Over range', value=np.nan)
df = df.replace(to_replace='bdl', value=np.nan)
df = df.replace(to_replace='<1', value=1)
df = df.replace(to_replace='Not determined', value=np.nan)
df = df.replace(to_replace='<0.01', value=0.01)
df = df.replace(to_replace='<0.1', value=0.1)
df = df.replace(to_replace='<5', value=np.nan)
df = df.replace(to_replace='<10', value=np.nan)
df = df.replace(to_replace='<4', value=np.nan)
df = df.replace(to_replace='<6', value=np.nan)
df = df.replace(to_replace='n.a.', value=np.nan)

# 1.2 - Drop FeO, FeO2, LOI and Total
df = df.drop(['Fe2O3','FeO','LOI','Total'], axis=1)

# 1.3 - Drop Samples labeled as Prueba (in Spanish "test"), Subsidiary Vcha dome which is an unidentified Volcano
#        and samples measured by 'Accelerator Mass Spectrometry' which is used to measure age and not geochemistry
df = df[(df.Volcan != 'Prueba')&(df.Volcan != 'Subsidiary Vcha dome')]
df = df[(df.TecnicaDeMedicion != 'Accelerator Mass Spectrometry')]

# 1.4 - Filter samples with Flag 4 == problems in geochemistry 
print('Number of samples with doutfull geochemistry: {}'.format(df[df.Flag == 4].shape[0]))
df = df[df.Flag != 4]

# here additionally the samples for which a problem in the classification of the sample has been indicated in the 
# literature might also be done. corresponding to Flag == 3
print('Number of samples with doutfull classification: {}'.format(df[df.Flag == 3].shape[0]))

df.loc[:, 'SiO2':'U'] = df.loc[:, 'SiO2':'U'].astype('float')

# 2 - Filter samples for which volcano id is unknown
df_unknown = df.query('Volcan == "Unknown"')
df = df.query('Volcan != "Unknown"').copy()
df['Volcan'] = df['Volcan'].astype("category")

# The list of volcanoes and associated codes
df['Volcan'].cat.categories
y = np.array(df['Volcan'].cat.codes)

# Print nb of samples per class
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print('id: {}, volcán: \033[1m{}\033[0m, count: {}'.format(u,df.Volcan.cat.categories[u],c))
n_classes = len(unique)

# 3 - Retrieve the geochemical data
X_major = df.loc[:, 'SiO2':'Cl'] 
X_traces = df.loc[:, 'Rb':'U']
X = pd.concat([X_major, X_traces], axis=1)

# 4 - Check how missing values are distributed: in two blocks.
Xmask = X.isna()
plt.imshow(Xmask, aspect='auto')

X1 = Xmask.loc[:, 'SiO2':'P2O5']
X2 = Xmask.loc[:, 'Rb':'U']

missing_traces = X1.sum(axis=1) < X2.sum(axis=1)
missing_traces = np.array(missing_traces, dtype=int)


# Learn a model, with stratified Kfold cross val.
# Metric used is the fraction of correctly classified samples.
#############################################################

# Choose the model you want to train
# Multinomial logistic regression
est = LogisticRegression(penalty='l2', multi_class='multinomial',
                         solver='saga', class_weight='balanced')
grid = {'logisticregression__C': [1e-2, 1e-1, 1, 1e1, 1e2]}

# K nearest neihbours
est = KNeighborsClassifier()
grid = {'kneighborsclassifier__n_neighbors': [2, 5, 10],
        'kneighborsclassifier__weights': ['uniform', 'distance']}

# Random Forest
est = RandomForestClassifier(n_estimators=3, min_samples_leaf=3,
                             random_state=0, class_weight='balanced')
grid = {'randomforestclassifier__n_estimators': [2, 10, 50, 100],
        'randomforestclassifier__min_samples_leaf': [1, 2, 5]}

# Cross-validation without grid search
# (i.e we take teh default hyperparameters of the models)
# ------------------------------------------------------
scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    imp = IterativeImputer(random_state=0)
    T_train = imp.fit_transform(X_train)
    T_test = imp.transform(X_test)

    sc = StandardScaler()
    T_train = sc.fit_transform(T_train)
    T_test = sc.transform(T_test)

    est.fit(T_train, y_train)
    ac = est.score(T_test, y_test)
    scores.append(ac)
print('Mean test set accuracy: {}'.format(sum(scores)/len(scores)))


# Cross-validation with grid search
# ---------------------------------
n_samples = X.shape[0]
n_rep = 5

X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)

clf = make_pipeline(IterativeImputer(random_state=0), StandardScaler(), est)

skf = StratifiedKFold(n_splits=3, shuffle=True)

gs = GridSearchCV(estimator=clf, param_grid=grid, cv=skf)
gs.fit(X_train_out, y_train_out)
res = pd.DataFrame(gs.cv_results_)
print(res)

# Compute accuracy
ac = gs.score(X_test_out, y_test_out)
print('Grid search CV accuracy: {}'.format(ac))

# Plot prediction in Ti02-K2O plane
y_pred = gs.predict(X_test_out)
hue = y_pred//2
style = y_pred % 2
pal = sns.color_palette("hls", len(np.unique(y_test_out)))
dpal = {}
for i, ID in enumerate(np.unique(y_test_out)):
    volcan = df.Volcan.cat.categories[ID]
    #print(volcan)
    color, marker = simbologia(volcan,'Unknown')
    dpal[volcan] = color

y_test_out_Volcan = df.Volcan.cat.categories[y_test_out]
y_pred_Volcan = df.Volcan.cat.categories[y_pred]

fig, axes = plt.subplots(1, 2)
ind_wrong = np.where(y_pred != y_test_out)[0]

sns.scatterplot(X_test_out.loc[:, 'TiO2'], X_test_out.loc[:, 'K2O'],
                hue=y_test_out_Volcan, alpha=0.7, palette=dpal, ax=axes[0])
sns.scatterplot(X_test_out.loc[:, 'TiO2'], X_test_out.loc[:, 'K2O'],
                hue=y_pred_Volcan, alpha=0.7, palette=dpal, ax=axes[1])

sns.scatterplot(X_test_out.TiO2.iloc[ind_wrong],
                X_test_out.K2O.iloc[ind_wrong],
                alpha=1, ax=axes[0], marker='x', color='k')
sns.scatterplot(X_test_out.TiO2.iloc[ind_wrong],
                X_test_out.K2O.iloc[ind_wrong],
                alpha=1,  ax=axes[1], marker='x', color='k')

axes[0].set_title("Original data")
axes[0].legend(loc='center left', bbox_to_anchor=(0, -0.4), ncol=3)
axes[1].set_title("Predicted data")
axes[1].legend(loc='center left', bbox_to_anchor=(0, -0.4), ncol=3)

fig, axes = plt.subplots(1, 2, figsize=(15,5))
ind_wrong = np.where(y_pred != y_test_out)[0]

sns.scatterplot(X_test_out.loc[:, 'SiO2'], X_test_out.loc[:, 'K2O'],
                hue=y_test_out_Volcan, alpha=0.7, palette=dpal, ax=axes[0])
sns.scatterplot(X_test_out.loc[:, 'SiO2'], X_test_out.loc[:, 'K2O'],
                hue=y_pred_Volcan, alpha=0.7, palette=dpal, ax=axes[1])

sns.scatterplot(X_test_out.SiO2.iloc[ind_wrong],
                X_test_out.K2O.iloc[ind_wrong],
                alpha=1, ax=axes[0], marker='x', color='k')
sns.scatterplot(X_test_out.SiO2.iloc[ind_wrong],
                X_test_out.K2O.iloc[ind_wrong],
                alpha=1,  ax=axes[1], marker='x', color='k')

axes[0].legend(loc='center left', bbox_to_anchor=(0, -0.4), ncol=3)
axes[1].legend(loc='center left', bbox_to_anchor=(0, -0.4), ncol=3)


# Plot confusion matrix
cm = confusion_matrix(y_test_out, y_pred)
cm = (cm.T/cm.sum(axis=1)).T
plt.imshow(cm)
plt.colorbar()

########################### Classifying new samples ########################
#Load data
Data_cores = pd.read_excel('../Data/DataCores.xlsx')

#Filtering the tephra I want to identify
Data_cores = Data_cores[Data_cores.Label.isin(["T9/100"])]

#Pre processing data
Data_cores.loc[:, 'SiO2':'U'] = Data_cores.loc[:, 'SiO2':'U'].astype('float')
Data_cores_mayor = Data_cores.loc[:, 'SiO2':'K2O']
Data_cores_trace = Data_cores.loc[:, 'Rb':'U']
X_cores = pd.concat([Data_cores_mayor, Data_cores_trace], axis=1)

#Predict
y_pred = gs.predict(X_cores)
(unique, counts) = np.unique(y_pred, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)




############################# SECOND PART ##################################
# We now do the same analysis as before, but forcing the samples for which both
# traces and major elmements are present to always be in the train set, so as
# to have more data to learn to impute well.
# Conclusion: doing this does not change the results

is_both = (X1.sum(axis=1) < 5) & (X2.sum(axis=1) < 13)
is_both = np.array(is_both)

n_both_per_volcan = np.zeros(n_classes)
for c, b in zip(y, is_both):
    n_both_per_volcan[c] += b
n_both_per_volcan = n_both_per_volcan.astype(int)

for nb, c in zip(n_both_per_volcan, counts):
    print('{}/{}'.format(nb, c))

# For two volcanos, more than half of the samples have both major and traces
# elements. In this case, we will only keep half of them in the permanent
# training subgroup, so that some samples of theses classes remain in the
# test sets.
is_permanent_train = is_both.copy()

vol_to_remove = np.where(n_both_per_volcan > 0.5*counts)[0]
for v in vol_to_remove:
    candidates = np.where((y == v) & is_both)[0]
    n_to_remove = n_both_per_volcan[v] - int(0.5*counts[v])
    ind = np.random.choice(candidates, size=n_to_remove, replace=False)
    is_permanent_train[ind] = False

# X_perm contains the samples for which both major elements and traces are
# available. X_other contains the rest.
X_perm = X.iloc[is_permanent_train]
X_other = X.iloc[~is_permanent_train]
y_perm = y[is_permanent_train]
y_other = y[~is_permanent_train]


# Cross-validation without grid search
# ------------------------------------
n_samples = X.shape[0]
n_rep = 5
scores = np.zeros(n_rep)
for i in range(n_rep):
    X_train, X_test, y_train, y_test = train_test_split(
        X_other, y_other, test_size=int(0.3*n_samples), random_state=i,
        stratify=y_other)
    X_train = pd.concat([X_train, X_perm], axis=0)
    y_train = np.concatenate([y_train, y_perm], axis=0)

    imp = IterativeImputer(random_state=0)
    T_train = imp.fit_transform(X_train)
    T_test = imp.transform(X_test)

    sc = StandardScaler()
    T_train = sc.fit_transform(T_train)
    T_test = sc.transform(T_test)

    est.fit(T_train, y_train)
    scores[i] = est.score(T_test, y_test)
print('Mean test set accuracy: {}'.format(scores.mean()))


# Cross-validation with grid search
# ---------------------------------
n_samples = X.shape[0]
n_perm = X_perm.shape[0]
n_rep = 5

X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(
    X_other, y_other, test_size=int(0.3*n_samples), random_state=0,
    stratify=y_other)

cv_splits = []
n_train_out = X_train_out.shape[0]
indices = np.arange(n_train_out)
for i in range(n_rep):
    ind_train, ind_test = train_test_split(
        indices, test_size=0.3, random_state=i, stratify=y_train_out)
    n_train = len(ind_train)
    ind_perm = np.arange(n_train, n_train + n_perm)
    ind_train = np.concatenate([ind_train, ind_perm])
    cv_splits.append((ind_train, ind_test))

grid = {'randomforestclassifier__n_estimators': [2, 10, 50, 100],
        'randomforestclassifier__min_samples_leaf': [1, 2, 5]}

clf = make_pipeline(IterativeImputer(random_state=0), StandardScaler(), est)
gs = GridSearchCV(estimator=clf, param_grid=grid, cv=cv_splits)
gs.fit(X_train_out, y_train_out)
res = pd.DataFrame(gs.cv_results_)
print(res)

# Compute accuracy
ac = gs.score(X_test_out, y_test_out)
print('Grid search CV accuracy: {}'.format(ac))

# Plot prediction in Ti02-K2O plane
y_pred = gs.predict(X_test_out)
hue = y_pred//2
style = y_pred % 2
pal = sns.color_palette("hls", len(np.unique(y_test_out)))
dpal = {}
for i, col in enumerate(pal):
    dpal[i] = col
fig, axes = plt.subplots(1, 2)
ind_wrong = np.where(y_pred != y_test_out)[0]

sns.scatterplot(X_test_out.loc[:, 'TiO2'], X_test_out.loc[:, 'K2O'],
                hue=y_test_out, alpha=0.8, palette=dpal, ax=axes[0])
sns.scatterplot(X_test_out.loc[:, 'TiO2'], X_test_out.loc[:, 'K2O'],
                hue=y_pred, alpha=0.8, palette=dpal, ax=axes[1])

sns.scatterplot(X_test_out.TiO2.iloc[ind_wrong],
                X_test_out.K2O.iloc[ind_wrong],
                alpha=1, ax=axes[0], marker='x', color='k')
sns.scatterplot(X_test_out.TiO2.iloc[ind_wrong],
                X_test_out.K2O.iloc[ind_wrong],
                alpha=1,  ax=axes[1], marker='x', color='k')

# Plot confusion matrix
cm = confusion_matrix(y_test_out, y_pred)
cm = (cm.T/cm.sum(axis=1)).T
plt.imshow(cm)
plt.colorbar()
