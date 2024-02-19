import pandas as pd
import optuna
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
def objective(trial):
    # 定义超参数搜索空间
    C = trial.suggest_loguniform('C', 0.1, 10.0)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    degree = trial.suggest_int('degree', 2, 5)
    gamma = trial.suggest_loguniform('gamma', 0.001, 0.1)
    # 创建SVM分类器
    model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, random_state=42,probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred)

    return score

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
study.optimize(objective, n_trials=100)

# 打印最佳超参数和对应的目标值
best_params = study.best_params
best_value = study.best_value
print("Best hyperparameters:", best_params)
print("Best value:", best_value)