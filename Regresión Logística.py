import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Carga y limpieza de datos (Preprocesamiento)
# Seleccionamos las variables predictoras clave
df = pd.read_csv('crash.csv')
features = ['LATITUDE_X', 'LONGITUDE_X', 'HOUR', 'MONTH', 'DAY_OF_WEEK']
X = df[features].fillna(0)  # Tratamiento de valores nulos
y = df['SEVERITY_TARGET']   # Variable objetivo

# División del dataset en conjuntos de entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Estandarización de características
# Paso obligatorio para Regresión Logística: asegura que todas las variables tengan la misma escala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Entrenamiento del modelo de Regresión Logística
# Optimizamos para maximizar la exactitud (Accuracy) global
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# 4. Evaluación del rendimiento
y_pred = model.predict(X_test_scaled)
print(f"--- Exactitud Final (Accuracy): {accuracy_score(y_test, y_pred):.4f} ---")

# 5. Visualización de la importancia de las variables (Coeficientes)
# Analizamos el peso de cada factor en la predicción del modelo
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({'Feature': features, 'Importance': coefficients})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Configuración del gráfico de barras para un análisis visual directo
plt.figure(figsize=(10, 6))

# Usamos 'hue' y 'legend=False' para una visualización limpia y moderna
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance,
    hue='Feature',      # Asigna un color único a cada variable
    palette='viridis',
    legend=False        # Elimina la leyenda redundante
)

# Títulos y etiquetas en español para el reporte final
plt.title('¿Qué factores influyen más en los accidentes? (Regresión Logística)', fontsize=14)
plt.xlabel('Peso en el modelo (Impacto relativo)', fontsize=12)
plt.ylabel('Variables Predictoras', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Ajuste automático del diseño para evitar cortes en el texto
plt.tight_layout()
plt.show()