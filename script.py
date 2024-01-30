from scipy.stats import wilcoxon
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

# Load the dataset
X = load_iris().data
y = load_iris().target

# Prepare models and select your CV method
model1 = ExtraTreesClassifier()
model2 = RandomForestClassifier()
kf = KFold(n_splits=20, random_state=42)

# Extract results for each model on the same folds
results_model1 = cross_val_score(model1, X, y, cv=kf)
results_model2 = cross_val_score(model2, X, y, cv=kf)

# Calculate p value
stat, p = wilcoxon(results_model1, results_model2, zero_method='zsplit'); p