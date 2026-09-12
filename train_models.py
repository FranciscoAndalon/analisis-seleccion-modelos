import csv
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from main import CANALES, dividir_datos, guardar_dataset, preparar_datos


BASE = Path(__file__).resolve().parent
SALIDA = BASE / "output" / "comparacion_modelos"


def cargar_datos():
    ruta = BASE / "output" / "dataset_ml.csv"
    if not ruta.exists():
        X, y, conteos = preparar_datos()
        guardar_dataset(X, y)
        print("Preparación:", conteos)
    columnas = [f"{canal}_{medida}" for canal in CANALES
                for medida in ["media_abs", "desviacion", "rango"]]
    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        if lector.fieldnames != columnas + ["movimiento"]:
            raise ValueError("El CSV no tiene las columnas esperadas.")
        X, y = [], []
        for fila in lector:
            X.append([float(fila[c]) for c in columnas])
            y.append(int(fila["movimiento"]))
    if not np.isfinite(X).all() or set(y) != set(range(16)):
        raise ValueError("Se necesitan datos finitos y las 16 clases.")
    return X, y, columnas


def crear_modelos():
    return {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=5, weights="uniform", metric="euclidean")),
        "Regresión logística": make_pipeline(StandardScaler(),
            LogisticRegression(C=1, solver="lbfgs", max_iter=2000)),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


def main():
    X, y, columnas = cargar_datos()
    X_ent, y_ent, X_prueba, y_prueba = dividir_datos(X, y)
    print(f"Muestras: {len(X)} | Entrenamiento: {len(X_ent)} | Prueba: {len(X_prueba)}")
    modelos = crear_modelos()
    for nombre, modelo in modelos.items():
        # Una falta de convergencia debe revisarse antes de comparar.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            modelo.fit(X_ent, y_ent)
        print("Entrenado:", nombre)
    SALIDA.mkdir(parents=True, exist_ok=True)
    archivo = SALIDA / "modelos.joblib"
    joblib.dump({"modelos": modelos, "columnas": columnas,
                 "X_prueba": X_prueba, "y_prueba": y_prueba,
                 "n_entrenamiento": len(X_ent), "semilla": 42,
                 "version_sklearn": sklearn.__version__}, archivo)
    cargados = joblib.load(archivo)
    for nombre, modelo in modelos.items():
        assert np.array_equal(modelo.predict(X_prueba),
                              cargados["modelos"][nombre].predict(X_prueba))
    print("Modelos guardados y predicciones verificadas:", archivo)


if __name__ == "__main__":
    main()
