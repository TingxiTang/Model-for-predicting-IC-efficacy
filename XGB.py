import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def objective(trial):
    # 定义超参数搜索空间
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 1.0, 0.1),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 0.01, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
    }

    # 创建XGBoost分类器
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    return auc

#导入数据
data = pd.read_csv(r'C:\Users\唐挺玺\Desktop\文章\投稿(Clinical Cancer Research)\Revise重分析\Fig5需修改\三组治疗前-normalized.csv', header=None)
data = data.T
data.columns = data.iloc[0,:]
data = data[1:]
X = np.array(data.drop(['Label','Sample'],axis=1))
y = np.array(data['Label'])
y[y=='no-response']="0"
y[y=='resistant']="0"
y[y=='sensitive']="1"
y=y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建Optuna Study对象
study = optuna.create_study(direction='maximize')

# 运行优化
study.optimize(objective, n_trials=700, timeout=600)
# 打印最佳超参数和对应的目标值
best_params = study.best_params
best_value = study.best_value
print("Best hyperparameters:", best_params)
print("Best value:", best_value)



