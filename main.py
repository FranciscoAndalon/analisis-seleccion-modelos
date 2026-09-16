import csv
import math
import random
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
DATOS = BASE / "Dataset" / "Rehab_exercise" / "d02_processed_data"
CANALES = ["f1", "f2", "f3", "f4", "f5", "pitch3"]


def preparar_datos():
    X, y = [], []
    vistos = {}
    conteos = {"originales": 0, "no_finitas": 0, "nulas": 0, "duplicadas": 0}
    for movimiento in range(16):
        muestras = np.load(DATOS / f"{movimiento:03d}_2.npy", allow_pickle=False)
        for muestra in muestras:
            conteos["originales"] += 1
            if not np.isfinite(muestra).all():
                conteos["no_finitas"] += 1
                continue
            if not muestra.any():
                conteos["nulas"] += 1
                continue
            clave = muestra.tobytes()
            if clave in vistos:
                if vistos[clave] != movimiento:
                    raise ValueError("Una señal idéntica tiene etiquetas diferentes.")
                conteos["duplicadas"] += 1
                continue
            vistos[clave] = movimiento
            fila = []
            for canal in range(6):
                valores = muestra[:, canal]
                fila.extend([
                    float(np.mean(np.abs(valores))),
                    float(np.std(valores, ddof=0)),
                    float(np.max(valores) - np.min(valores)),
                ])
            X.append(fila)
            y.append(movimiento)
    return X, y, conteos


def guardar_dataset(X, y):
    salida = BASE / "output" / "dataset_ml.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    columnas = [f"{canal}_{medida}" for canal in CANALES
                for medida in ["media_abs", "desviacion", "rango"]]
    with salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(columnas + ["movimiento"])
        for fila, etiqueta in zip(X, y):
            escritor.writerow(fila + [etiqueta])


def dividir_datos(X, y):
    azar = random.Random(42)
    entrenamiento, prueba = [], []
    for clase in sorted(set(y)):
        indices = [i for i, etiqueta in enumerate(y) if etiqueta == clase]
        azar.shuffle(indices)
        corte = int(len(indices) * 0.8)
        entrenamiento.extend(indices[:corte])
        prueba.extend(indices[corte:])
    azar.shuffle(entrenamiento)
    azar.shuffle(prueba)
    return ([X[i] for i in entrenamiento], [y[i] for i in entrenamiento],
            [X[i] for i in prueba], [y[i] for i in prueba])


def normalizar(X, medias, desviaciones):
    return [[(valor - media) / desviacion
             for valor, media, desviacion in zip(fila, medias, desviaciones)]
            for fila in X]


def entrenar(X, y):
    # Los parámetros de normalización se calculan solo con entrenamiento.
    medias, desviaciones = [], []
    for columna in zip(*X):
        media = sum(columna) / len(columna)
        desviacion = math.sqrt(sum((v - media) ** 2 for v in columna) / len(columna))
        medias.append(media)
        desviaciones.append(desviacion if desviacion != 0 else 1)
    return {"X": normalizar(X, medias, desviaciones), "y": list(y),
            "medias": medias, "desviaciones": desviaciones}


def distancia(a, b):
    return math.sqrt(sum((x - z) ** 2 for x, z in zip(a, b)))


def predecir(modelo, X, k=5):
    if not 1 <= k <= len(modelo["X"]):
        raise ValueError("k debe estar entre 1 y la cantidad de muestras de entrenamiento.")
    predicciones = []
    for fila in normalizar(X, modelo["medias"], modelo["desviaciones"]):
        vecinos = [(distancia(fila, ejemplo), etiqueta)
                   for ejemplo, etiqueta in zip(modelo["X"], modelo["y"])]
        vecinos.sort()
        votos, distancias = {}, {}
        for valor, etiqueta in vecinos[:k]:
            votos[etiqueta] = votos.get(etiqueta, 0) + 1
            distancias[etiqueta] = distancias.get(etiqueta, 0) + valor
        ganador = min(votos, key=lambda clase: (-votos[clase], distancias[clase], clase))
        predicciones.append(ganador)
    return predicciones


def evaluar(reales, predicciones):
    matriz = [[0] * 16 for _ in range(16)]
    aciertos = 0
    for real, predicho in zip(reales, predicciones):
        matriz[real][predicho] += 1
        if real == predicho:
            aciertos += 1
    return aciertos / len(reales), matriz


def main():
    X, y, conteos = preparar_datos()
    guardar_dataset(X, y)
    print("Preparación:", conteos)
    print(f"Dataset: {len(X)} muestras, {len(X[0])} características, {len(set(y))} clases")
    X_ent, y_ent, X_prueba, y_prueba = dividir_datos(X, y)
    print(f"Entrenamiento: {len(X_ent)} | Prueba: {len(X_prueba)}")
    modelo = entrenar(X_ent, y_ent)
    predicciones = predecir(modelo, X_prueba)
    print("\nPrimeras diez predicciones:")
    for real, predicho in list(zip(y_prueba, predicciones))[:10]:
        print(f"Real: {real:02d} | Predicho: {predicho:02d}")
    exactitud, matriz = evaluar(y_prueba, predicciones)
    mayoritaria = min(set(y_ent), key=lambda clase: (-y_ent.count(clase), clase))
    referencia, _ = evaluar(y_prueba, [mayoritaria] * len(y_prueba))
    print(f"\nExactitud KNN (k=5): {exactitud:.2%}")
    print(f"Referencia (siempre clase {mayoritaria:02d}): {referencia:.2%}")
    print("\nMatriz de confusión: filas = reales, columnas = predicciones")
    print("     " + " ".join(f"{clase:3d}" for clase in range(16)))
    for clase, fila in enumerate(matriz):
        print(f"{clase:02d} | " + " ".join(f"{cantidad:3d}" for cantidad in fila))
    print("\nDataset guardado en:", BASE / "output" / "dataset_ml.csv")


if __name__ == "__main__":
    main()
