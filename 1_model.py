"""
Trains and saves an XGBoost model on the wine data. 

The final model fit on all data is saved as xgb_full_train.json for use in the microservice.
"""
# %% Load Modules
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from matplotlib import pyplot as plt
import shap 
from datetime import datetime

# %% Load Dataframe
df = pd.read_pickle("df.pkl")
df.info()
X = df.drop('rating', axis=1)
y = df.pop('rating')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute sample weights
sw_train = compute_sample_weight("balanced", y_train)
print(f"train sample weights are {sw_train}")

# Compute full dataset weights
sw_full = compute_sample_weight("balanced", y)
print(f"full sample weights are {sw_full}")

# %% CV training to optimize hyperparameters

clf = xgb.XGBClassifier()

# best params for quick rerun of code
parameters = {
     "learning_rate"    : [0.1],
     "max_depth"        : [4],
     "min_child_weight" : [3],
     "gamma"            : [0.1],
     "colsample_bytree" : [1],
     "n_estimators"     : [50]
     }

# uncomment for actual search
# parameters = {
#      "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#      "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12],
#      "min_child_weight" : [ 1, 3, 5, 7 ],
#      "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#      "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7, 1 ],
#      "n_estimators"     : [25, 50, 100, 150]
#      }

cv = RandomizedSearchCV(clf,
                    parameters,n_iter=150, n_jobs=3,
                    scoring="neg_log_loss",
                    cv=5, verbose=10,
                    )

start_time = datetime.now() 
cv.fit(X_train, y_train, sample_weight = sw_train )
time_elapsed = datetime.now() - start_time 
print('Time elapsed for cv fit is (hh:mm:ss.ms) {}'.format(time_elapsed))

# %% Save the best estimator model 
bst = cv.best_estimator_
bst.save_model('best_xgb_cv.json')
xgb_latest = xgb.XGBClassifier(n_jobs=3)
xgb_latest.load_model("best_xgb_cv.json")

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
# %% Sanity check for changes from full train
xgb.plot_importance(xgb_latest)
plt.show()

#  %% Shap interpretation of final fit
explainer = shap.TreeExplainer(xgb_latest)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
 