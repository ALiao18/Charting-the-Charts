import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import random as random
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.special import expit 
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.metrics import roc_curve

random.seed(16339968)
data = pd.read_csv("/Users/Andrew/Desktop/Code/ds102/spotify52kData.csv")

#%% Q1
#Question 1
#duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, 
#liveness, valence, and tempo

# Sample data extraction from a DataFrame named 'data'
duration = data['duration']
danceability = data['danceability']
energy = data['energy']
loudness = data['loudness']
speechiness = data['speechiness']
acousticness = data['acousticness']
instrumentalness = data['instrumentalness']
liveness = data['liveness']
valence = data['valence']
tempo = data['tempo']

fig, axs = plt.subplots(5, 2, figsize=(15, 20))  
fig.subplots_adjust(hspace=0.5, wspace=0.4) 

plot_config = [
    {'data': duration, 'label': "Duration", 'bins': 50, 'range': (1, 450000)},
    {'data': danceability, 'label': "Danceability", 'bins': 30, 'range': None},
    {'data': energy, 'label': "Energy", 'bins': 30, 'range': None},
    {'data': loudness, 'label': "Loudness", 'bins': 30, 'range': None},
    {'data': speechiness, 'label': "Speechiness", 'bins': 30, 'range': None},
    {'data': acousticness, 'label': "Acousticness", 'bins': 30, 'range': None},
    {'data': instrumentalness, 'label': "Instrumentalness", 'bins': 30, 'range': None},
    {'data': liveness, 'label': "Liveness", 'bins': 30, 'range': None},
    {'data': valence, 'label': "Valence", 'bins': 30, 'range': None},
    {'data': tempo, 'label': "Tempo", 'bins': 30, 'range': None}
]

for ax, config in zip(axs.flatten(), plot_config):
    ax.hist(config['data'], bins = config['bins'], range = config['range'])
    ax.set_title(config['label'])
    ax.set_xlabel(config['label'])
    ax.set_ylabel('Count')

plt.show()

#%% Q2
#Question 2 test relationship between duration and popularity
def countZeroOrNan(col):
    return sum((data[col] == 0) | pd.isnull(data[col]))

durationNanCount = countZeroOrNan("duration") #0
popularityNanCount = countZeroOrNan("popularity") #6374

dataQ2 = data.dropna(subset = ["popularity", "duration"])
dataQ2 = data.loc[(data['popularity'] != 0) & (data['duration'] != 0)]
dataQ2['duration_log'] = np.log(data['duration'] + 1)

plt.scatter(dataQ2["duration"], dataQ2["popularity"])
plt.show()
durationPopularityCorrelation = dataQ2['popularity'].corr(dataQ2['duration']) #-0.1

dataQ2b = dataQ2.loc[(dataQ2["duration"] < 1000000)]
plt.scatter(dataQ2b["duration"], dataQ2b["popularity"])
plt.show()
durationPopularityCorrelationB = dataQ2b['popularity'].corr(dataQ2b['duration']) #-0.11

plt.scatter(dataQ2["duration_log"], dataQ2["popularity"])
plt.show()
durationPopulationCorrelationC = dataQ2['popularity'].corr(dataQ2['duration_log']) #-0.04

#%% Q3
#Question 3 comparing popularity between explicit and non-explicit songs
#first, split the dataset into 2, one where explicit is true and one where explicit is false
dataExplicit = data[data['explicit'] == True]
dataNonExplicit = data[data['explicit'] == False]
dataExplicitClean = dataExplicit[(dataExplicit['popularity'] > 0) & (dataExplicit['popularity'].notna())]
dataNonExplicitClean = dataNonExplicit[(dataNonExplicit['popularity'] > 0) & (dataNonExplicit['popularity'].notna())]

#plotting 
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].hist(dataExplicit['popularity'], bins=40)
axs[0, 0].set_title("Popularity Distribution For Explicit Songs")
axs[0, 0].set_xlabel("Popularity")
axs[0, 0].set_ylabel("Count")

axs[0, 1].hist(dataNonExplicit['popularity'], bins=40)
axs[0, 1].set_title("Popularity Distribution For non-Explicit Songs")
axs[0, 1].set_xlabel("Popularity")
axs[0, 1].set_ylabel("Count")

axs[1, 0].hist(dataExplicitClean['popularity'], bins=40)
axs[1, 0].set_title("Popularity Distribution For Explicit Songs (without 0s)")
axs[1, 0].set_xlabel("Popularity")
axs[1, 0].set_ylabel("Count")

axs[1, 1].hist(dataNonExplicitClean['popularity'], bins=40)
axs[1, 1].set_title("Popularity Distribution For non-Explicit Songs (without 0s)")
axs[1, 1].set_xlabel("Popularity")
axs[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.show()

uStatistic, pValue = mannwhitneyu(dataExplicitClean['popularity'], dataNonExplicitClean['popularity'], alternative='greater')
#uStatistic: 108874531.5
#pValue: 2.9626001463178775e-18

#I use a difference in proportions, being songs with popularity rating over 50
#Ho: there is no difference explicit and non-explicit songs in popularity
#Ha: explicit songs are more popular than non-explicit songs

#since we are only measuring if it is greater, we should use a 2 tail
#here I caluclate the necessary numbers
#x1 = (dataExplicitClean['popularity'] > 50).sum() # 1628
#x2 = (dataNonExplicitClean['popularity'] > 50).sum() # 11026
#n1 = len(dataExplicitClean) # 4985
#n2 = len(dataNonExplicitClean) # 40641
#explicitProportion = x1 / n1 # 0.3266
#nonexplicitProportion = x2 / n2 # 0.2713
#pPooled = (x1 + x2) / (n1 + n2) # 0.2773418664796388
#se = np.sqrt(pPooled * (1 - pPooled) * ((1 / n1) + (1 / n2))) # 0.006718395841684671
#z = ((explicitProportion - nonexplicitProportion) / se) # 8.227761678479112
#pValue = 1 - stats.norm.cdf(z) # 1.1102230246251565e-16

#However, after more consideration, I decided to go with the mann whitney u test since
#I cannot assume popularity to be normally distributed. 
uStatistic, pValue = mannwhitneyu(dataExplicitClean['popularity'], dataNonExplicitClean['popularity'], alternative='greater')
#uStatistic: 108874531.5
#pValue: 2.9626001463178775e-18
#based on the above p value, we can confidently reject the null hypothesis and accept the 
#alternative hypothesis that explicit songs are more popular than non explicit songs

#%% Q4
#Question 4: comparing popularity between major key songs and minor key songs
#first, split dataset into major key and minor key songs, 1 is major, 0 minor
dataMajorKey = data[data['mode'] == 1]
dataMinorKey = data[data['mode'] == 0]
dataMajorKeyClean = dataMajorKey[(dataMajorKey['popularity'] > 0) & (dataMajorKey['popularity'].notna())]
dataMinorKeyClean = dataMinorKey[(dataMinorKey['popularity'] > 0) & (dataMinorKey['popularity'].notna())]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].hist(dataMajorKey['popularity'], bins=40)
axs[0, 0].set_title("Popularity Distribution For Songs in Major")
axs[0, 0].set_xlabel("Popularity")
axs[0, 0].set_ylabel("Count")

axs[0, 1].hist(dataMinorKey['popularity'], bins=40)
axs[0, 1].set_title("Popularity Distribution For Songs in Minor")
axs[0, 1].set_xlabel("Popularity")
axs[0, 1].set_ylabel("Count")

axs[1, 0].hist(dataMajorKeyClean['popularity'], bins=40)
axs[1, 0].set_title("Popularity Distribution For Songs in Major (without 0s)")
axs[1, 0].set_xlabel("Popularity")
axs[1, 0].set_ylabel("Count")

axs[1, 1].hist(dataMinorKeyClean['popularity'], bins=40)
axs[1, 1].set_title("Popularity Distribution For Songs in Minor (without 0s)")
axs[1, 1].set_xlabel("Popularity")
axs[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.show()

uStatistic, pValue = mannwhitneyu(dataMajorKeyClean['popularity'], dataMinorKeyClean['popularity'], alternative='greater')
#uStatistic: 243630068.5
#pValue: 0.936642073259969

# we can observe that major and minor songs have similar probability densities both
# with and without zero values and there seems to be a peak in songs rated around 20 for both 
# as well. 

#I use a difference in proportions, being songs with popularity rating over 50
#Ho: there is no difference major key and minor key songs in popularity
#Ha: major key songs are more popular than minor key songs
#since we are only measuring if it is greater, we should use a 2 tail
#here I caluclate the necessary numbers

#x1 = (dataMajorKeyClean['popularity'] > 50).sum() # 7479
#x2 = (dataMinorKeyClean['popularity'] > 50).sum() # 5175
#n1 = len(dataMajorKeyClean) # 28198
#n2 = len(dataMinorKeyClean) # 17428

#majorKeyProportion = x1 / n1 # 0.2652315767075679
#minorKeyProportion = x2 / n2 # 0.29693596511361026
#pPooled = (x1 + x2) / (n1 + n2) # 0.2773418664796388
#se = np.sqrt(pPooled * (1 - pPooled) * ((1 / n1) + (1 / n2))) # 0.004313675122854755
#z = ((majorKeyProportion - minorKeyProportion) / se) # 
#pValue = 1 - stats.norm.cdf(z) # 

#After further considering, I am choosing to use the Mann Whitney U test
uStatistic, pValue = mannwhitneyu(dataMajorKeyClean['popularity'], dataMinorKeyClean['popularity'], alternative='greater')
#uStatistic: 243630068.5
#pValue: 0.936642073259969
#based on the high pValue, we fail to reject the null hypothesis and therefore continue on with 
#the assumption that there isn't a difference in popularity between major and minor key songs
#%% Q5
#Q5: determining relationship between energy and loudness
# judging from the distribution charts in Q1, we can reasonably agree that both energy
# and loudness are not normally distributed 
# loudness is actually normlaized with 0 being the greatest value since decibel is on a 
# here I can also handle null and 0 values by dropping the rows because missing values likely 
# happens randomly and isn't because of a specific criteria
# initial scatterplot
dataQ5Clean = data[(data['energy'].notna()) & (data['loudness'].notna())]

plt.scatter(dataQ5Clean['loudness'], dataQ5Clean['energy'])
plt.title("Scatterplot Between Loudness and Energy")
plt.xlabel("loudness")
plt.ylabel("energy")
plt.show()
# The results of this plot is a trumpet shape
# I can apply spearman's rank correlation here since the relationship is monotonic (doesn't decrease)
rho, p = spearmanr(dataQ5Clean['energy'], dataQ5Clean['loudness'])
#rho: 0.7306382054765808
#p = 0.0
    
# However, I am still going to check again after transforming the loudness scale to a linear one
# Energy is a "made up" feature and I'm assuming that it is on a linear scale, 
# but just so happens that most songs are energetic

pRef = 20e-6  # reference pressure in pascals (20 micropascals)
dataQ5Clean['pressure'] = pRef * 10**(dataQ5Clean['loudness'] / 20)

#plot after transforming
plt.scatter(dataQ5Clean['pressure'], dataQ5Clean['energy'])
plt.title("Scatterplot Between Sound Pressure and Energy")
plt.xlabel("pressure")
plt.ylabel("energy")
plt.show()

#now I can use the correlation coefficient r 
r, p = pearsonr(dataQ5Clean['energy'], dataQ5Clean['pressure'])
#r: 0.7666329867098407
#p: 0.0
#r^2: 0.5877261363116508
# I can substantiate that energy largely reflects loudness in a song
#%% Q6
#Q6 Which of the 10 features in Q1 predicts popularity best? How good is the best model
#First I want to check whether removing 0s in popularity changes the distribution of each variable
dataQ6Clean = data[(data['popularity'] > 0) & (data['popularity'].notna())]

features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Normalize the data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(dataQ6Clean[features]), columns=features)

# Check for multicollinearity
plt.figure(figsize=(12, 10))
correlation_matrix = X_scaled.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Correlation analysis
r = []
p = []

for feature in features:
    r_value, p_value = pearsonr(X_scaled[feature], dataQ6Clean['popularity'])
    r.append(r_value)
    p.append(p_value)

# Print correlation results
print("Correlation with popularity:")
for feature, r_value, p_value in zip(features, r, p):
    print(f"{feature}: r = {r_value:.4f}, p = {p_value:.4e}")

# Visualize correlations
plt.figure(figsize=(12, 6))
plt.bar(features, r)
plt.title("Correlation of Features with Popularity")
plt.xlabel("Features")
plt.ylabel("Correlation Coefficient")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Linear regression models
X = X_scaled
y = dataQ6Clean['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16339968)

results = []

# Baseline model
baseline_rmse = np.sqrt(mean_squared_error(y_test, [y_train.mean()] * len(y_test)))
baseline_r2 = r2_score(y_test, [y_train.mean()] * len(y_test))
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Baseline R-squared: {baseline_r2:.4f}")

for feature in features:
    X_train_feature = X_train[[feature]]
    X_test_feature = X_test[[feature]]
    model = LinearRegression()
    model.fit(X_train_feature, y_train)
    y_pred = model.predict(X_test_feature)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append([rmse, r2])

results_array = np.array(results)

# Visualize R-squared values
plt.figure(figsize=(12, 6))
plt.bar(features, results_array[:, 1])
plt.title("R-squared Values for Each Feature")
plt.xlabel("Features")
plt.ylabel("R-squared")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print results
print("\nLinear Regression Results:")
for feature, (rmse, r2) in zip(features, results):
    print(f"{feature}: RMSE = {rmse:.4f}, R-squared = {r2:.4f}")

# Identify best predictor
best_feature_index = np.argmax(results_array[:, 1])
best_feature = features[best_feature_index]
best_rmse, best_r2 = results[best_feature_index]

print(f"\nBest predictor: {best_feature}")
print(f"RMSE: {best_rmse:.4f}")
print(f"R-squared: {best_r2:.4f}")

#r: [-0.10004, 0.08024, -0.114576, 0.05724633, -0.10568152108160818, 0.05095887450089502, -0.24046427297484535, -0.09268998187181707, 0.006896509811715853, -0.04569551014101218]
#p: [8.238062504484929e-102, 4.6584619484991824e-66, 3.909755098563911e-133, 1.9559479171891944e-34, 1.8897528311310124e-113, 1.2623650408395213e-27, 0.0, 1.3184651387445522e-87, 0.140727189530324, 1.5832016614246769e-22]
#maxR: 0.24046. This corresponds to instrumentalness, therefore instrumentalness is the best model, but is still horrible
#it is interesting to note that all models except for valence had a p value of something very close to 0, but valence had
#a p value of 0.14.

#%%6b
dataQ6Clean = data[(data['popularity'] > 0) & (data['popularity'].notna())]

pRef = 20e-6  # code from above
dataQ6Clean['pressure'] = pRef * 10**(dataQ5Clean['loudness'] / 20)

target = dataQ6Clean['popularity']
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataQ6Clean[features], target, test_size = 0.2, random_state = 16339968)

results = []

for feature in features:
    X_train_feature = X_train[[feature]]
    X_test_feature = X_test[[feature]]
    model = LinearRegression()
    model.fit(X_train_feature, y_train)
    y_pred = model.predict(X_test_feature)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results.append([rmse, r2])

results_array = np.array(results)

#array([[ 1.91313818e+01,  1.11858611e-02],
#       [ 1.91963174e+01,  4.46201935e-03],
#       [ 1.91099017e+01,  1.34050289e-02],
#       [ 1.91972726e+01,  4.36294006e-03],
#       [ 1.91368269e+01,  1.06229091e-02],
#       [ 1.92124747e+01,  2.78544819e-03],
#       [ 1.86900748e+01,  5.62780237e-02],
#       [ 1.91480442e+01,  9.46270449e-03],
#       [ 1.92402518e+01, -1.00150117e-04],
#       [ 1.92068917e+01,  3.36492851e-03]])

#%% Q7
#Building a model that uses *all* of the song features from question 1, how well can you 
#predict popularity now? How much (if at all) is this model improved compared to the best
#model in question 6). How do you account for this?

#I will approach this question by conducting both multiple linear regression and ridge regression to see which is better
#MLR
# Data preparation
dataQ7Clean = data[(data['popularity'] > 0) & (data['popularity'].notna())]
features = ["duration", "danceability", "energy", "loudness", "speechiness", 
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

X = dataQ7Clean[features]
y = dataQ7Clean['popularity']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=16339968)

# Multiple Linear Regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)

y_pred_mlr = mlr.predict(X_test)
rmse_mlr = np.sqrt(mean_squared_error(y_test, y_pred_mlr))
r2_mlr = r2_score(y_test, y_pred_mlr)

print("Multiple Linear Regression Results:")
print(f"RMSE: {rmse_mlr:.4f}")
print(f"R-squared: {r2_mlr:.4f}")

# Ridge Regression
ridge = Ridge(alpha=1.0)  # You can experiment with different alpha values
ridge.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression Results:")
print(f"RMSE: {rmse_ridge:.4f}")
print(f"R-squared: {r2_ridge:.4f}")

# Compare coefficients
mlr_coefs = pd.DataFrame({'Feature': features, 'MLR Coefficient': mlr.coef_})
ridge_coefs = pd.DataFrame({'Feature': features, 'Ridge Coefficient': ridge.coef_})
coef_comparison = pd.merge(mlr_coefs, ridge_coefs, on='Feature')

print("\nCoefficient Comparison:")
print(coef_comparison)

# Visualize coefficient comparison
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(features))

plt.bar(index, mlr_coefs['MLR Coefficient'], bar_width, label='MLR')
plt.bar(index + bar_width, ridge_coefs['Ridge Coefficient'], bar_width, label='Ridge')

plt.xlabel('Features')
plt.ylabel('Coefficient Values')
plt.title('Comparison of MLR and Ridge Regression Coefficients')
plt.xticks(index + bar_width / 2, features, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance ranking
mlr_importance = pd.DataFrame({'Feature': features, 'Importance': np.abs(mlr.coef_)})
mlr_importance = mlr_importance.sort_values('Importance', ascending=False)

ridge_importance = pd.DataFrame({'Feature': features, 'Importance': np.abs(ridge.coef_)})
ridge_importance = ridge_importance.sort_values('Importance', ascending=False)

#%% Q8
# how many meaningful principal components can you extract? 
# What proportion of the variance do these principal components account for?

#correlation matrix for all 10 variables
dataQ8Clean = data[(data['popularity'] > 0) & (data['popularity'].notna())]
dataQ8Clean = data[['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

corrMatrix = np.corrcoef(dataQ8Clean,rowvar = False)
plt.imshow(corrMatrix) 
plt.colorbar()
plt.show()

#observation1: collinearity seems to be mostly not an issue, except for 1. 
#Now, since pretty much all variables are not normally distributed, I will z-score all of them
zscoredData = stats.zscore(dataQ8Clean)

pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
#array([2.73393354, 1.61739086, 1.38460532, 0.97960682, 0.87522623,
#       0.8148464 , 0.67828163, 0.4715811 , 0.31313969, 0.13158071])

loadings = pca.components_
rotatedData = pca.fit_transform(zscoredData)

varExplained = eigVals/sum(eigVals)*100

for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))

numSongs = len(dataQ8Clean)
x = np.arange(1, 11)
plt.bar(x, eigVals, color='blue')
plt.plot([0,10],[1,1],color='orange') 
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

#using the kaiser criterion, we retain only the first 3 components
#%% Q9
# Can you predict whether a song is in major or minor key from valence? If so, how good is this
# prediction? If not, is there a better predictor?
# Data preparation
dataQ9Clean = data.dropna(subset=['mode', 'valence'])
X = dataQ9Clean[['valence']]  # We'll start with just valence as in the original question
y = dataQ9Clean['mode']

# Check class distribution
print("Class distribution before balancing:")
print(y.value_counts(normalize=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16339968)

# Create a pipeline with Random Undersampling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('undersampler', RandomUnderSampler(sampling_strategy='majority', random_state=16339968)),
    ('classifier', LogisticRegression())
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluate
print("\nModel Performance (Valence only):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Valence only)')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Minor', 'Major'])
plt.yticks(tick_marks, ['Minor', 'Major'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations to the confusion matrix
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Valence only)')
plt.legend()
plt.show()

# Now let's try with all features
features = ['valence', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'tempo']

X_all = dataQ9Clean[features]

# Split the data
X_train_all, X_test_all, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=16339968)

# Create and fit the pipeline
pipeline_all = Pipeline([
    ('scaler', StandardScaler()),
    ('undersampler', RandomUnderSampler(sampling_strategy='majority', random_state=16339968)),
    ('classifier', LogisticRegression())
])

pipeline_all.fit(X_train_all, y_train)

# Predict
y_pred_all = pipeline_all.predict(X_test_all)
y_pred_proba_all = pipeline_all.predict_proba(X_test_all)[:, 1]

# Evaluate
print("\nModel Performance (All Features):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_all):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_all):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_all):.4f}")

# Feature Importance
classifier = pipeline_all.named_steps['classifier']
feature_importance = pd.DataFrame({'Feature': features, 'Importance': np.abs(classifier.coef_[0])})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Feature Importance for Major/Minor Key Prediction')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nFeature Importance Ranking:")
print(feature_importance)


#%% Q10
# I don't see any reason why classical music would be missing data for a particular reason
# so I remove nans by row removal
# Data preparation
dataQ10Clean = data.copy()
dataQ10Clean['classical'] = ((data['track_genre'] == 'classical') & (data['duration'] <= 1000000)).astype(int)

# Features
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = dataQ10Clean[features]
y = dataQ10Clean['classical']

# Check class distribution
print("Class distribution before balancing:")
print(y.value_counts(normalize=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16339968)

# Function to create and evaluate model
def create_evaluate_model(X_train, y_train, X_test, y_test, use_pca=False):
    steps = [
        ('scaler', StandardScaler()),
        ('undersampler', RandomUnderSampler(sampling_strategy='majority', random_state=16339968))
    ]
    
    if use_pca:
        steps.append(('pca', PCA(n_components=3)))
    
    steps.append(('classifier', LogisticRegression()))
    
    pipeline = Pipeline(steps)
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print(f"\nModel Performance ({'PCA' if use_pca else 'All Features'}):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({("PCA" if use_pca else "All Features")})')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-Classical', 'Classical'])
    plt.yticks(tick_marks, ['Non-Classical', 'Classical'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({("PCA" if use_pca else "All Features")})')
    plt.legend()
    plt.show()
    
    return pipeline

# Create and evaluate model with all features
model_all = create_evaluate_model(X_train, y_train, X_test, y_test, use_pca=False)

# Create and evaluate model with PCA
model_pca = create_evaluate_model(X_train, y_train, X_test, y_test, use_pca=True)

# Feature Importance for all features model
classifier = model_all.named_steps['classifier']
feature_importance = pd.DataFrame({'Feature': features, 'Coefficient': classifier.coef_[0]})
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Coefficient'])
plt.title('Feature Coefficients for Classical Music Prediction')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

print("\nFeature Coefficients:")
print(feature_importance[['Feature', 'Coefficient']])

#%% Q10b
X_pca = rotatedData[:, :3]  # Select the first three principal components
Y = dataQ10Clean['classical'].values

model = LogisticRegression()
model.fit(X_pca, Y)

x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
x_values = np.linspace(x_min, x_max, 500).reshape(-1, 1)

mean_pc2 = np.mean(X_pca[:, 1])  # Mean of the second principal component
mean_pc3 = np.mean(X_pca[:, 2])  # Mean of the third principal component
x_full = np.column_stack([x_values, np.full(x_values.shape, mean_pc2), np.full(x_values.shape, mean_pc3)])

y_pred = model.predict(X_pca)
y_pred_proba = model.predict_proba(X_pca)[:, 1] 

#x_full = np.column_stack([x_values, np.full((len(x_values), 2), np.mean(X_pca[:, 1:3], axis=0))])
y_values = model.predict_proba(x_full)[:, 1]

accuracy = accuracy_score(Y, y_pred) #0.9795769230769231
precision = precision_score(Y, y_pred, average='binary') #0.3106796116504854
roc_auc = roc_auc_score(Y, y_pred_proba) #0.9476993448412805

plt.figure(figsize=(8, 5))
fpr, tpr, thresholds = roc_curve(Y, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], Y, color='blue', alpha=0.2, label='Data Points')  
x_values = np.linspace(min(X_pca[:, 0]), max(X_pca[:, 0]), 300)
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]
plt.plot(x_values, y_values, color='red', linewidth=2, label='Sigmoid Curve')
plt.xlabel('First Principal Component')
plt.ylabel('Probability of Being Classical')
plt.title('Logistic Regression Sigmoid Curve on First Principal Component')
plt.legend()
plt.show()


