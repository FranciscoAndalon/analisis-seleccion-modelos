import csv
import json

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate

from main import dividir_datos
from train_models import BASE, SALIDA, cargar_datos
from evaluate_models import evaluar, plt


DESTINO = BASE / "output" / "refinamiento_random_forest"


def configuraciones():
    base = dict(n_estimators=100, max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features="sqrt", max_samples=None)
    return {
        "Base": base,
        "Más árboles": dict(base, n_estimators=200),
        "Profundidad limitada": dict(base, n_estimators=200, max_depth=20,
                                     min_samples_split=4),
        "Hojas mayores": dict(base, n_estimators=200, max_depth=20,
                             min_samples_split=4, min_samples_leaf=2),
        "Más características": dict(base, n_estimators=200, max_depth=30,
                                    min_samples_split=4, max_features=0.5),
        "Muestreo y hojas mayores": dict(base, n_estimators=200, max_depth=20,
                                         min_samples_split=4, min_samples_leaf=2,
                                         max_features=0.5, max_samples=0.8),
    }


def buscar_configuracion(X, y):
    particiones = list(StratifiedKFold(n_splits=3, shuffle=True,
                                     random_state=42).split(X, y))
    metrica = make_scorer(f1_score, average="macro", labels=list(range(16)),
                         zero_division=0)
    resultados = []
    for nombre, parametros in configuraciones().items():
        modelo = RandomForestClassifier(**parametros, random_state=42)
        valores = cross_validate(modelo, X, y, cv=particiones, scoring=metrica,
                                return_train_score=True, error_score="raise")
        fila = {"configuracion": nombre, **parametros,
                "f1_entrenamiento": float(valores["train_score"].mean()),
                "f1_validacion": float(valores["test_score"].mean()),
                "desviacion_validacion": float(valores["test_score"].std())}
        for i, valor in enumerate(valores["test_score"], 1):
            fila[f"f1_fold_{i}"] = float(valor)
        resultados.append(fila)
        print(f"{nombre}: F1 validación = {fila['f1_validacion']:.4f}", flush=True)
    # En un empate exacto se conserva la primera configuración, incluida la base.
    mejor = max(resultados, key=lambda fila: fila["f1_validacion"])
    return mejor["configuracion"], resultados


def guardar_csv(nombre, filas):
    with (DESTINO / nombre).open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def main():
    archivo = SALIDA / "modelos.joblib"
    if not archivo.exists() or not (SALIDA / "resultados.csv").exists():
        raise SystemExit("Primero ejecuta python train_models.py y python evaluate_models.py")
    anteriores = joblib.load(archivo)
    if anteriores["version_sklearn"] != sklearn.__version__:
        raise SystemExit("Cambió scikit-learn. Vuelve a entrenar y evaluar los modelos iniciales.")
    X, y, columnas = cargar_datos()
    X_ent, y_ent, X_prueba, y_prueba = dividir_datos(X, y)
    if (columnas != anteriores["columnas"] or
            not np.array_equal(X_prueba, anteriores["X_prueba"]) or
            y_prueba != anteriores["y_prueba"]):
        raise ValueError("Los datos cambiaron. Vuelve a entrenar y evaluar los modelos iniciales.")
    print(f"Ajuste con {len(X_ent)} muestras; prueba reservada: {len(X_prueba)}.")
    nombre, validacion = buscar_configuracion(X_ent, y_ent)
    parametros = configuraciones()[nombre]
    elegido = RandomForestClassifier(**parametros, random_state=42).fit(X_ent, y_ent)
    # La elección ya terminó. Ahora se consulta la prueba para informar resultados.
    resultados, predicciones, matrices = evaluar(
        {"RF original": anteriores["modelos"]["Random Forest"],
         "RF seleccionado por CV": elegido}, X_prueba, y_prueba)
    DESTINO.mkdir(parents=True, exist_ok=True)
    guardar_csv("validacion.csv", validacion)
    guardar_csv("comparacion_prueba.csv", resultados)
    with (DESTINO / "predicciones.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["real"] + list(predicciones))
        escritor.writerows(zip(y_prueba, *predicciones.values()))
    resumen = {"configuracion_elegida": nombre, "parametros": parametros,
               "criterio": "Mayor F1 macro promedio de 3 folds sobre entrenamiento",
               "n_entrenamiento": len(X_ent), "n_prueba": len(X_prueba),
               "resultados_prueba": resultados,
               "limite": "La prueba ya se utilizó para seleccionar la familia Random Forest."}
    (DESTINO / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    joblib.dump({"modelo": elegido, "columnas": columnas, "parametros": parametros,
                 "configuracion": nombre, "version_sklearn": sklearn.__version__},
                DESTINO / "modelo_refinado.joblib")
    cargado = joblib.load(DESTINO / "modelo_refinado.joblib")["modelo"]
    assert np.array_equal(cargado.predict(X_prueba), elegido.predict(X_prueba))
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    posiciones = np.arange(len(validacion))
    ax.barh(posiciones - 0.18, [r["f1_entrenamiento"] for r in validacion],
            height=0.36, label="Entrenamiento de cada fold", color="#aec7d8")
    ax.barh(posiciones + 0.18, [r["f1_validacion"] for r in validacion],
            height=0.36, xerr=[r["desviacion_validacion"] for r in validacion],
            label="Validación (media ± desviación)", color="#246a73", capsize=3)
    ax.set_yticks(posiciones, [r["configuracion"] for r in validacion])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 macro")
    ax.set_title("Refinamiento de Random Forest: validación cruzada de 3 folds")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
    fig.savefig(DESTINO / "validacion.png", dpi=150)
    plt.close(fig)
    for etiqueta, matriz in matrices.items():
        assert matriz.sum() == len(y_prueba)
    print("\nElegido por validación:", nombre, parametros)
    for fila in resultados:
        print(f"{fila['modelo']}: accuracy={fila['accuracy']:.4f}, F1={fila['f1_macro']:.4f}")
    print("Resultados:", DESTINO)


if __name__ == "__main__":
    main()
