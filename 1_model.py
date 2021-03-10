# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

#best 

# With a model made, create a service which responds to HTTP requests. We only need one endpoint:...
# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.5, gamma=0.0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.01, max_delta_step=0, max_depth=2,
#               min_child_weight=1, missing=nan, monotone_constraints='()',
#               n_estimators=25, n_jobs=4, num_parallel_tree=1,
#               objective='multi:softprob', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='auto', validate_parameters=1, verbosity=None)

# {'objective': 'multi:softprob', 'use_label_encoder': True, 'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.4, 'gamma': 0.1, 'gpu_id': -1, 'importance_type': 'gain', 'interaction_constraints': '', 'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 4, 'min_child_weight': 3, 'missing': nan, 'monotone_constraints': '()', 'n_estimators': 50, 'n_jobs': 3, 'num_parallel_tree': 1, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': None, 'subsample': 1, 'tree_method': 'auto', 'validate_parameters': 1, 'verbosity': None}

parameters = {
     "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
     "n_estimators"     : [25, 50, 100, 150]
     }

cv = RandomizedSearchCV(clf,
                    parameters,n_iter=10, n_jobs=3,
                    scoring="neg_log_loss",
                    cv=5, verbose=10,
                    )

cv.fit(X_train, y_train, sample_weight = sw_train )


# %% 

# model.dump_model('dump.raw.txt')
bst = cv.best_estimator_

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
