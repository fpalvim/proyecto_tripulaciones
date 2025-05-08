# modelo_actividades.py

# Librerías
import warnings
warnings.filterwarnings("ignore")
import random

# Datos
import pandas as pd
import numpy as np
import re
import ast
from collections import Counter

# Gráficos
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocesamiento
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, MinMaxScaler

# División de datos y optimización de Modelos
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Modelos
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
from xgboost import XGBRegressor

# Métricas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Guardar Modelo
import joblib

# Configuración de Pandas
pd.set_option('display.max_columns', None)

# Otros (Generación de datos y manipulación de fechas)
from faker import Faker
from datetime import timedelta, datetime

# Variables
fake = Faker('es_ES')
np.random.seed(42)
random.seed(42)

ACTIVIDADES = {
    "ocio": 15,
    "viajes": 30,
    "deportes": 20,
    "salud": 12,
    "educación": 18,
    "cine": 10
}

RANGO_ACTIVIDADES = {
    "ocio": (4, 14),
    "viajes": (6, 14),
    "deportes": (6, 14),
    "salud": (0, 14),
    "educación": (4, 14),
    "cine": (3, 14)
}

def estacion_del_anio(fecha):
    mes = fecha.month
    if mes in [12, 1, 2]:
        return 'Invierno'
    elif mes in [3, 4, 5]:
        return 'Primavera'
    elif mes in [6, 7, 8]:
        return 'Verano'
    else:
        return 'Otoño'

def generar_datos_sinteticos(fecha_inicio="2021-04-01", fecha_fin="2024-04-01"):
    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    registros = []
    regiones = ["Madrid", "Barcelona", "Málaga", "Valencia", "Sevilla", "Zaragoza"]

    for fecha in fechas:
        for _ in range(np.random.randint(20, 40)):
            genero = random.choice(["M", "F"])
            discapacidad = random.choices(
                ["Ninguna", "Sensorial", "Motora", "Cognitiva"],
                weights=[0.85, 0.05, 0.05, 0.05]
            )[0]

            actividad = random.choices(
                list(ACTIVIDADES.keys()),
                weights=[0.20, 0.20, 0.20, 0.15, 0.15, 0.10]
            )[0]

            min_ed, max_ed = RANGO_ACTIVIDADES[actividad]
            edad_min = random.randint(min_ed, max_ed - 1)
            edad_max = random.randint(edad_min + 1, max_ed)
            grupo_edad = f"{edad_min}-{edad_max}"
            edad = random.randint(edad_min, edad_max)

            base = ACTIVIDADES[actividad]
            variacion = np.random.normal(0, 3)
            ajuste_edad = 1.1 if edad >= 10 else 1.0
            gasto = max(5, round((base + variacion) * ajuste_edad, 2))

            registros.append({
                "fecha": fecha,
                "edad": edad,
                "genero": genero,
                "discapacidad": discapacidad,
                "actividad": actividad,
                "region": random.choice(regiones),
                "gasto": gasto,
                "estacion": estacion_del_anio(fecha),
                "edad_minima": edad_min,
                "edad_maxima": edad_max,
                "grupo_edad": grupo_edad
            })

    return pd.DataFrame(registros)

# Generar datos
df = generar_datos_sinteticos()

# Limpieza y transformación
df['fecha'] = pd.to_datetime(df['fecha'])
df['gasto'] = df['gasto'].astype(float)

df['dia_semana'] = df['fecha'].dt.day_name()
df['mes'] = df['fecha'].dt.month
df['anio'] = df['fecha'].dt.year

df_encoded = df.copy()
label_cols = ['genero', 'discapacidad', 'actividad', 'estacion', 'grupo_edad', 'dia_semana']
encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df_encoded[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

agg_edad_estacion = df.groupby(['grupo_edad', 'estacion']).agg({
    'gasto': ['mean', 'count'],
    'actividad': lambda x: x.mode().iloc[0] if not x.mode().empty else None
}).reset_index()

agg_edad_estacion.columns = ['grupo_edad', 'estacion', 'gasto_medio', 'num_registros', 'actividad_mas_comun']

# Preparación para modelo
X = df_encoded[[
    'edad', 'genero_enc', 'discapacidad_enc', 'actividad_enc',
    'estacion_enc', 'grupo_edad_enc', 'dia_semana_enc', 'mes'
]]
y = df_encoded['gasto']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo XGBoost Regressor con optimización
param_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5]
}

random_search_xgb = RandomizedSearchCV(
    estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
    param_distributions=param_xgb,
    n_iter=30,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

random_search_xgb.fit(X_train, y_train)
best_xgb = random_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2 = r2_score(y_test, y_pred_xgb)

# print("XGBoost - Optimizado:")
# print(f"MAE  : {mae:.4f}")
# print(f"RMSE : {rmse:.4f}")
# print(f"R²   : {r2:.4f}")

# Guardar el modelo
joblib.dump(best_xgb, "modelo_xgboost_optimizado.joblib")
# print("Modelo guardado como 'modelo_xgboost_optimizado.joblib'")

# Cargar el modelo
modelo_cargado = joblib.load("modelo_xgboost_optimizado.joblib")

def predecir_gasto(input_data):
    modelo = joblib.load("modelo_xgboost_optimizado.joblib")
    prediccion = modelo.predict(input_data)
    return prediccion[0]

if __name__ == "__main__":
    nuevo = pd.DataFrame({
        'edad': [3],
        'genero_enc': [1],
        'discapacidad_enc': [0],
        'actividad_enc': [2],
        'estacion_enc': [3],
        'grupo_edad_enc': [2],
        'dia_semana_enc': [4],
        'mes': [5]
    })

    resultado = predecir_gasto(nuevo)
    print(f"💸 Predicción del gasto: {resultado:.2f} unidades monetarias")