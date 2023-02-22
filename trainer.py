import pandas as pd
import numpy as np
import json
from joblib import dump
from scipy.sparse import data
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine
from datetime import datetime

from targeter import get_target_up_down, get_target_regression

all_pairs = {
"AUDCAD": [0.0040, 0.0018],
"AUDCHF": [0.0040, 0.0015],
"AUDJPY": [0.55, 0.20],
"AUDNZD": [0.0025, 0.0010],
"AUDUSD": [0.004, 0.002],
"CADCHF": [0.0033, 0.0015],
"CADJPY": [0.70, 0.25],
"CHFJPY": [0.60, 0.25],
"EURAUD": [0.008, 0.003],
"EURCAD": [0.006, 0.0025],
"EURCHF": [0.0025, 0.001],
"EURGBP": [0.0025, 0.001],
"EURJPY": [0.4, 0.15],
"EURUSD": [0.0035, 0.0015],
"GBPAUD": [0.006, 0.0025],
"GBPCAD": [0.005, 0.002],
"GBPCHF": [0.006, 0.0025],
"GBPJPY": [0.6, 0.25],
"GBPNZD": [0.006, 0.0025],
"GBPUSD": [0.004, 0.002],
"NZDCAD": [0.003, 0.0015],
"NZDCHF": [0.0035, 0.0015],
"NZDJPY": [0.45, 0.15],
"NZDUSD": [0.004, 0.0015],
"USDCAD": [0.0042, 0.0018],
"USDCHF": [0.003, 0.0012],
"USDJPY": [0.32, 0.1],
}

engine = create_engine("mysql+pymysql://rex:#Pass123@localhost/new_ml")



if __name__ == "__main__":
  model_metrics = {}
  h_grid = {"n_estimators": [200, 250, 300],
            "max_depth": [None, 5],
            "min_samples_split": [2],
            "min_samples_leaf": [1, 2, 4]}
  for pair in all_pairs.keys():

    print(f"Working on {pair} at: {datetime.now().strftime('%H:%M:%S')}")
    
    #Training for Hourly classifier
    df_h = pd.read_sql(f"SELECT * FROM `{pair.lower()}_1h` ORDER BY `index` ASC", engine, index_col="index")
    
    print("Targeting...")
    df_h = get_target_up_down(df_h, uptick=all_pairs[pair][0], downtick=all_pairs[pair][1])
    X = df_h.drop(["target_up", "target_down"], axis=1)
    y_up = df_h["target_up"]
    y_down = df_h["target_down"]

    Xhup_train, Xhup_test, yhup_train, yhup_test = train_test_split(X, y_up, test_size=0.2) 
    Xhdown_train, Xhdown_test, yhdown_train, yhdown_test = train_test_split(X, y_down, test_size=0.2)

    print("Training up")
    clf = RandomForestClassifier(n_jobs=-1)
    cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    up_clf = GridSearchCV(estimator=clf, scoring='roc_auc', param_grid=h_grid,
                          cv=cross_val, return_train_score=True, n_jobs=-1, verbose=2)

    up_clf.fit(Xhup_train, yhup_train)
    up_final_clf = RandomForestClassifier(n_jobs=-1,
                                          n_estimators = up_clf.best_params_["n_estimators"],
                                          max_depth = up_clf.best_params_["max_depth"],
                                          min_samples_split = up_clf.best_params_["min_samples_split"],
                                          min_samples_leaf = up_clf.best_params_["min_samples_leaf"])

    up_final_clf.fit(Xhup_train, yhup_train)
    
    print("Training down")
    down_clf = GridSearchCV(estimator=clf, scoring='roc_auc', param_grid=h_grid,
                          cv=cross_val, return_train_score=True, n_jobs=-1, verbose=2)

    down_clf.fit(Xhdown_train, yhdown_train)

    down_final_clf = RandomForestClassifier(n_jobs=-1,
                                          n_estimators = down_clf.best_params_["n_estimators"],
                                          max_depth = down_clf.best_params_["max_depth"],
                                          min_samples_split = down_clf.best_params_["min_samples_split"],
                                          min_samples_leaf = down_clf.best_params_["min_samples_leaf"])

    down_final_clf.fit(Xhdown_train, yhdown_train)

    #Getting Confusion matrices
    #For up
    up_cms = {}
    down_cms = {}
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
    for threshold in thresholds:
      predicted_proba = up_final_clf.predict_proba(Xhup_test)
      predictions = (predicted_proba [:,1] >= threshold).astype('int')

      up_cm = confusion_matrix(yhup_test, predictions)

      up_cms[threshold] = {"positive": str(up_cm[1, 1]),
                           "false_positive": str(up_cm[0, 1]),
                           "missed_positive": str(up_cm[1, 0])}

      predicted_proba_down = down_final_clf.predict_proba(Xhdown_test)
      predictions_down = (predicted_proba_down [:,1] >= threshold).astype('int')

      down_cm = confusion_matrix(yhdown_test, predictions_down)

      down_cms[threshold] = {"positive": str(down_cm[1, 1]),
                             "false_positive": str(down_cm[0, 1]),
                             "missed_positive": str(down_cm[1, 0])}

    print("Training 4h")
    #Training regression for 4 hour periods
    df_4h = pd.read_sql(f"SELECT * FROM `{pair.lower()}_4h` ORDER BY `index` ASC", engine, index_col="index")
    df_4h = get_target_regression(df_4h)

    X_4h = df_4h.drop(["target_regression"], axis=1)
    y_4h = df_4h["target_regression"]

    X4h_train, X4h_test, y4h_train, y4h_test = train_test_split(X_4h, y_4h, test_size=0.2)

    model_4h = RandomForestRegressor(n_jobs=-1)
    model_4h.fit(X4h_train, y4h_train)
    y4h_preds = model_4h.predict(X4h_test)

    metrics_4h = {"Average candle": str(df_4h["bodysize"].abs().mean()),
                  "Mean Absolute Error": str(mean_absolute_error(y4h_test, y4h_preds))}


    #Training regression for daily periods
    print("Training D")
    df_d = pd.read_sql(f"SELECT * FROM `{pair.lower()}_d` ORDER BY `index` ASC", engine, index_col="index")
    df_d = get_target_regression(df_d)

    X_d = df_d.drop(["target_regression"], axis=1)
    y_d = df_d["target_regression"]

    Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_d, y_d, test_size=0.2)

    model_d = RandomForestRegressor(n_jobs=-1)
    model_d.fit(Xd_train, yd_train)
    yd_preds = model_d.predict(Xd_test)

    metrics_d = {"Average candle": str(df_d["bodysize"].abs().mean()),
                  "Mean Absolute Error": str(mean_absolute_error(yd_test, yd_preds))}

    model_metrics[pair] = {"Hourly UP": up_cms,
                           "Hourly DOWN": down_cms,
                           "4 Hours": metrics_4h,
                           "Daily": metrics_d}

    # save models
    dump(up_final_clf, filename=f"./models/{pair.lower()}_h_up.joblib")
    dump(down_final_clf, filename=f"./models/{pair.lower()}_h_down.joblib")
    dump(model_4h, filename=f"./models/{pair.lower()}_4h.joblib")
    dump(model_d, filename=f"./models/{pair.lower()}_d.joblib")
    print(f"Models for {pair} saved")

  with open('./models/model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(model_metrics, f, ensure_ascii=False, indent=4)
  print("Performance for each model recorded")