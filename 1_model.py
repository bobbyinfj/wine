# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from matplotlib import pyplot as plt

# %%
df = pd.read_pickle("df.pkl")
df.info()
X = df.drop('rating', axis=1)
y = df.pop('rating')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#compute sample weights
sw_train = compute_sample_weight("balanced", y_train)

print(f"train sample weights are {sw_train}")
#compute full dataset weights
sw_full = compute_sample_weight("balanced", y)
print(f"full sample weights are {sw_full}")
# %%
# xgb.plot_tree(xg_clf,num_trees=10)
# xgb.to_graphviz(bst, num_trees=2)

# plt.rcParams['figure.figsize'] = [50, 10]
# plt.show()
# plt.savefig('tree.png')

# %% CV training to optimize hyperparameters

clf = xgb.XGBClassifier()

#WORK ON TUNING LATER

parameters = {
     'learning_rate': [.01],
     'max_depth': [2, 5],
     'min_child_weight' : [1, 3],
     'gamma': [ 0.0 ],
     'colsample_bytree': [0.5, 1],
     'n_estimators': [25, 50]
}

# parameters = {
#      'learning_rate': [.01, .30],
#      'max_depth': [2, 3, 4,  6, 9],
#      'min_child_weight' : [ 1, 3, 5, 7 ],
#      'gamma': [ 0.0, 0.2 ],
#      'colsample_bytree': [0.5, 0.75, 1],
#      'n_estimators': [25, 50, 100],

# }

# parameters = {
#      "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#      "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
#      "min_child_weight" : [ 1, 3, 5, 7 ],
#      "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#      "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
#      "n_estimators"     : [150]
#      }

grid = GridSearchCV(clf,
                    parameters, n_jobs=3,
                    scoring="neg_log_loss",
                    cv=5, verbose=10,
                    )

grid.fit(X_train, y_train, sample_weight = sw_train )

# %% 

# model.dump_model('dump.raw.txt')
bst = grid.best_estimator_

bst.save_model('best_xgb_cv.json')

xgb_latest = xgb.XGBClassifier(n_jobs=3) # or which ever sklearn booster you're are using
xgb_latest.load_model("best_xgb_cv.json") # or model.bin if you are using binary format and not the json

# %% Validate model on holdout data
preds = xgb_latest.predict(X_test)
report = classification_report(y_test, preds)
print(report)

xgb.plot_importance(xgb_latest)
plt.show()

# %% Retrain on all data and export model
print(xgb_latest.get_params())
xgb_latest.fit(X, y, sample_weight= sw_full)
print(xgb_latest.get_params())
xgb_latest.save_model('xgb_full_train.json')
# %% Sanity check
xgb.plot_importance(xgb_latest)
plt.show()
# %%
