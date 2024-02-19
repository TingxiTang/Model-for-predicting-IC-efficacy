import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.utils import resample

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
feature_names = np.array(data.columns)[2:] #特征名称

#分为训练和测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Lasso建模
lasso_model = Lasso(alpha=0.0010070077621547207,random_state=42)
lasso_model.fit(X_train, y_train)
#SVM建模
svm_model = SVC(C=5.424570027106945, kernel='poly', degree=4, gamma=0.005583527236348648, random_state=42, probability=True)
svm_model.fit(X_train, y_train)
#RF建模
rf_model = RandomForestClassifier(n_estimators=125, max_depth=3, min_samples_split=10, max_features='sqrt', random_state=42)
rf_model.fit(X_train, y_train)
#XGB建模
para={'booster': 'gbtree', 'max_depth': 10, 'subsample': 0.7, 'colsample_bytree': 0.6, 'learning_rate': 0.3355020700783656, 'reg_alpha': 0.003537520178148478, 'reg_lambda': 0.0018767104846311063, 'gamma': 0.07351445833656385, 'min_child_weight': 4, 'n_estimators': 264}
xgb_model = XGBClassifier(**para)
xgb_model.fit(X_train, y_train)

# 预测概率
svm_probs = svm_model.predict_proba(X_train)[:, 1]
lasso_probs = lasso_model.predict(X_train)
rf_probs = rf_model.predict_proba(X_train)[:, 1]
xgb_probs = xgb_model.predict_proba(X_train)[:, 1]

# 计算AUC值
svm_auc = roc_auc_score(y_train, svm_probs)
lasso_auc = roc_auc_score(y_train, lasso_probs)
rf_auc = roc_auc_score(y_train, rf_probs)
xgb_auc = roc_auc_score(y_train, xgb_probs)

# 绘制Train_ROC曲线
svm_fpr, svm_tpr, _ = roc_curve(y_train, svm_probs)
lasso_fpr, lasso_tpr, _ = roc_curve(y_train, lasso_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_train, rf_probs)
xgb_fpr, xgb_tpr, _ = roc_curve(y_train, xgb_probs)

plt.plot(svm_fpr, svm_tpr, label=f"SVM (AUC = {svm_auc:.3f}")
plt.plot(lasso_fpr, lasso_tpr, label=f"Lasso (AUC = {lasso_auc:.3f}])")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.3f}])")
plt.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC = {xgb_auc:.3f}])")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
# plt.savefig(r'C:\Users\唐挺玺\Desktop\文章\投稿(Clinical Cancer Research)\Revise重分析\Fig5需修改\train_AUC.pdf', format='PDF')
plt.close()

# 预测概率
svm_probs = svm_model.predict_proba(X_test)[:, 1]
lasso_probs = lasso_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# 计算AUC值
svm_auc = roc_auc_score(y_test, svm_probs)
lasso_auc = roc_auc_score(y_test, lasso_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
xgb_auc = roc_auc_score(y_test, xgb_probs)

# 计算置信区间
def bootstrap_auc(y_true, y_pred, n_bootstrap=1000):
    aucs = []
    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred[indices]
        aucs.append(roc_auc_score(bootstrap_true, bootstrap_pred))
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    return lower, upper

svm_lower, svm_upper = bootstrap_auc(y_test, svm_probs)
lasso_lower, lasso_upper = bootstrap_auc(y_test, lasso_probs)
rf_lower, rf_upper = bootstrap_auc(y_test, rf_probs)
xgb_lower, xgb_upper = bootstrap_auc(y_test, xgb_probs)

# 绘制ROC曲线
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
lasso_fpr, lasso_tpr, _ = roc_curve(y_test, lasso_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)

plt.plot(svm_fpr, svm_tpr, label=f"SVM (AUC = {svm_auc:.3f}, [{svm_lower:.3f}, {svm_upper:.3f}])")
plt.plot(lasso_fpr, lasso_tpr, label=f"Lasso (AUC = {lasso_auc:.3f}, [{lasso_lower:.3f}, {lasso_upper:.3f}])")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.3f}, [{rf_lower:.3f}, {rf_upper:.3f}])")
plt.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC = {xgb_auc:.3f}, [{xgb_lower:.3f}, {xgb_upper:.3f}])")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# plt.savefig(r'C:\Users\唐挺玺\Desktop\文章\投稿(Clinical Cancer Research)\Revise重分析\Fig5需修改\test_AUC.pdf', format='PDF')

# 获取特征重要性
importances = xgb_model.feature_importances_

# 创建一个包含特征重要性和特征名称的DataFrame
df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})


# 将DataFrame导出为CSV文件
df.to_csv(r'C:\Users\唐挺玺\Desktop\文章\投稿(Clinical Cancer Research)\Revise重分析\Fig5需修改\feature_importance.csv', index=False)

data_v = pd.read_csv(r'C:\Users\唐挺玺\Desktop\文章\投稿(Clinical Cancer Research)\Revise重分析\Fig5需修改\validation_normalized.csv')
data_v = data_v.T
data_v.columns = data_v.iloc[0,:]
data_v = data_v[1:]
X_v = np.array(data_v.drop(['Label'],axis=1))
y_v=xgb_model.predict(X_v)
df = pd.DataFrame({'label': y_v})
df.index=data_v.index
