import csv
import json
from pathlib import Path

import joblib
import matplotlib
import sklearn
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SALIDA = Path(__file__).resolve().parent / "output" / "comparacion_modelos"


def evaluar(modelos, X, y):
    resultados, predicciones, matrices = [], {}, {}
    for nombre, modelo in modelos.items():
        predicho = modelo.predict(X)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predicho, labels=list(range(16)), average="macro", zero_division=0)
        resultados.append({"modelo": nombre, "accuracy": accuracy_score(y, predicho),
                           "precision_macro": precision, "recall_macro": recall,
                           "f1_macro": f1})
        predicciones[nombre] = predicho
        matrices[nombre] = confusion_matrix(y, predicho, labels=list(range(16)))
    resultados.sort(key=lambda fila: (-fila["f1_macro"], -fila["accuracy"], fila["modelo"]))
    return resultados, predicciones, matrices


def guardar_resultados(resultados, predicciones, matrices, y):
    with (SALIDA / "resultados.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(resultados[0]))
        escritor.writeheader()
        escritor.writerows(resultados)
    with (SALIDA / "predicciones.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["real"] + list(predicciones))
        escritor.writerows(zip(y, *predicciones.values()))
    ganador = resultados[0]
    resumen = {"ganador": ganador["modelo"], "metricas": ganador,
               "muestras_prueba": len(y), "criterio": "F1 macro; desempate por accuracy y nombre",
               "limite": "Selección sobre esta prueba; no es una evaluación independiente ni por participantes."}
    (SALIDA / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figura, ejes = plt.subplots(2, 2, figsize=(17, 15), layout="constrained")
    maximo = max(m.max() for m in matrices.values())
    for eje, (nombre, matriz) in zip(ejes.flat, matrices.items()):
        dibujo = ConfusionMatrixDisplay(matriz, display_labels=list(range(16))).plot(
            ax=eje, cmap="Blues", colorbar=False, values_format="d",
            im_kw={"vmin": 0, "vmax": maximo},
            text_kw={"fontsize": 7})
        for texto, cantidad in zip(dibujo.text_.flat, matriz.flat):
            texto.set_color("white" if cantidad > maximo / 2 else "#16324f")
        eje.set_title(nombre)
        eje.set_xlabel("Movimiento predicho")
        eje.set_ylabel("Movimiento real")
    figura.suptitle("Comparación de clasificadores — conjunto de prueba")
    figura.savefig(SALIDA / "matrices_confusion.png", dpi=150)
    plt.close(figura)


def main():
    archivo = SALIDA / "modelos.joblib"
    if not archivo.exists():
        raise SystemExit("Primero ejecuta: python train_models.py")
    datos = joblib.load(archivo)
    if datos["version_sklearn"] != sklearn.__version__:
        raise SystemExit("Cambió la versión de scikit-learn. Ejecuta de nuevo: python train_models.py")
    resultados, predicciones, matrices = evaluar(
        datos["modelos"], datos["X_prueba"], datos["y_prueba"])
    guardar_resultados(resultados, predicciones, matrices, datos["y_prueba"])
    print(f"{'Modelo':23s} {'Accuracy':>10s} {'Precisión':>10s} {'Recall':>10s} {'F1 macro':>10s}")
    for fila in resultados:
        print(f"{fila['modelo']:23s} {fila['accuracy']:10.4f} {fila['precision_macro']:10.4f} "
              f"{fila['recall_macro']:10.4f} {fila['f1_macro']:10.4f}")
    ganador = resultados[0]["modelo"]
    print("\nMejor modelo en esta comparación:", ganador)
    print("Criterio: mayor F1 macro; desempate por accuracy y nombre.")
    for real, predicho in list(zip(datos["y_prueba"], predicciones[ganador]))[:10]:
        print(f"Real: {real:02d} | Predicho: {predicho:02d}")
    print("\nResultados guardados en:", SALIDA)


if __name__ == "__main__":
    main()
