import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# --- 1. 加载数据 ---
file_path = 'cleaned_crash_data_v1.csv' # 请确保文件名正确
df = pd.read_csv(file_path)

features = ['AGE', 'LATITUDE_X', 'LONGITUDE_X', 'HOUR', 'MONTH', 'DAY_OF_WEEK']
target = 'SEVERITY_TARGET'

X = df[features].fillna(0)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# --- 2. 标准化 ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. 逻辑回归优化 (Grid Search) ---
print("\n[正在进行 GridSearchCV 优化计算...]")

# 我们不仅优化 C，还引入了权重平衡来处理不平衡数据
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs']
}

# 使用 f1_weighted 作为评分标准，这在不平衡数据中比 accuracy 更专业
grid_log = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=2000),
    param_grid,
    cv=5,
    scoring='f1_weighted'
)
grid_log.fit(X_train_scaled, y_train)


# --- 4. 打印 证据表  ---
print("\n" + "="*60)
print("   SECCIÓN 7: EVIDENCIA DE OPTIMIZACIÓN (LOGISTIC REGRESSION)")
print("="*60)
results = pd.DataFrame(grid_log.cv_results_)
evidence = results[['param_C', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score')
print(evidence.to_string(index=False))
print("="*60)

# --- 5. 最终结果展示 ---
best_model = grid_log.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print(f"\nMejor parámetro C: {grid_log.best_params_['C']}")
print(f"Precisión Final (Balanced): {accuracy_score(y_test, y_pred):.4f}")
print("\nInforme de Clasificación Detallado:")
print(classification_report(y_test, y_pred))

# --- 6. 可视化变量重要性 ---
plt.figure(figsize=(10, 6))
coeffs = best_model.coef_[0]
sns.barplot(x=coeffs, y=features, palette='viridis')
plt.title('Influencia de las Variables en la Gravedad del Accidente')
plt.xlabel('Coeficiente (Importancia)')
plt.show()
