import pandas as pd
import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def objective(trial):
    # 定义超参数搜索空间
    n_estimators = trial.suggest_int('n_estimators', 20, 150, step=5)
    max_depth = trial.suggest_int('max_depth', 2, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    # 创建随机森林分类器
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=42
    )
    model.fit(X_train, y_train)
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
y[y=='no-response']='0'
y[y=='resistant']='0'
y[y=='sensitive']='1'
y=y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建Optuna Study对象
study = optuna.create_study(direction='maximize')

# 运行优化
study.optimize(objective, n_trials=300)

# 打印最佳超参数和对应的目标值
best_params = study.best_params
best_value = study.best_value
print("Best hyperparameters:", best_params)
print("Best value:", best_value)