# Selección de modelo — clasificación de ejercicios de rehabilitación

Comparación de 4 algoritmos clásicos de aprendizaje automático para
clasificar el tipo de movimiento (16 actividades) a partir de señales de
sensores de un guante de rehabilitación, y selección justificada del
modelo más adecuado.

---

## 1. Dataset

`output/dataset_ml.csv` contiene una fila por cada ciclo de movimiento
registrado por el guante de rehabilitación, con:

- **18 características numéricas**: 6 canales de sensor (`f1`-`f5`,
  `pitch3`) × 3 descriptores estadísticos por canal (`media_abs`,
  `desviacion`, `rango`).
- **1 columna de etiqueta** (`movimiento`): un entero de `0` a `15` que
  identifica cuál de las 16 actividades/ejercicios de rehabilitación se
  realizó en ese ciclo.

`train_models.py` separa este dataset en entrenamiento y prueba (785
muestras de prueba, según `resumen.json`) y entrena 4 modelos con esa
misma partición, para que la comparación entre ellos sea justa.

---

## 2. Modelos comparados

| Modelo | Configuración usada |
|---|---|
| KNN | `k=5`, distancia euclidiana, con estandarización previa (`StandardScaler`) |
| Regresión logística | `C=1`, solver `lbfgs`, con estandarización previa |
| Árbol de decisión | `max_depth=10` |
| Random Forest | `n_estimators=100` |

Los 4 se evalúan sobre el **mismo conjunto de prueba**, nunca usado
durante el entrenamiento, con las métricas: accuracy, precision macro,
recall macro y F1 macro (macro = las 16 actividades pesan igual,
independientemente de cuántas muestras tenga cada una).

---

## 3. Resultados

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **Random Forest** | **0.9070** | **0.9095** | **0.9078** | **0.9074** |
| Árbol de decisión | 0.6624 | 0.6613 | 0.6591 | 0.6512 |
| KNN | 0.6331 | 0.6310 | 0.6411 | 0.6273 |
| Regresión logística | 0.3682 | 0.3761 | 0.3618 | 0.3577 |

**Modelo ganador: Random Forest**, según `resumen.json`, con el criterio
"mayor F1 macro; desempate por accuracy y nombre".

---

## 4. Por qué Random Forest es el modelo más adecuado

### 4.1 Ventaja de desempeño clara y consistente

Random Forest no solo gana en F1 macro (criterio de selección), sino en
**las 4 métricas simultáneamente**, y por un margen muy amplio: ~24 puntos
porcentuales de accuracy sobre el segundo lugar (Árbol de decisión,
0.907 vs 0.662) y más del doble de accuracy que la Regresión logística
(0.907 vs 0.368). No es una victoria marginal ni depende de qué métrica
se privilegie — gana de forma robusta en cualquiera de ellas.

### 4.2 Las matrices de confusión muestran por qué

En `matrices_confusion.png` se ve claramente el patrón:

- **Random Forest** tiene una diagonal fuertemente dominante en las 16
  actividades, con muy poca "fuga" de predicciones hacia otras clases.
- **Árbol de decisión** y **KNN** muestran confusión notable entre
  grupos de actividades específicos (por ejemplo, varias de las
  actividades intermedias se confunden sistemáticamente entre sí),
  reflejo de que esas actividades probablemente involucran movimientos
  anatómicamente parecidos que un solo árbol o una distancia simple no
  logran separar bien.
- **Regresión logística** es la que muestra la confusión más extendida y
  dispersa por toda la matriz, consistente con que las clases no se
  separan mediante fronteras lineales.

### 4.3 Por qué tiene sentido, en términos del algoritmo

- **Reduce la varianza de un árbol individual**: un Random Forest es un
  conjunto (ensamble) de muchos árboles de decisión, cada uno entrenado
  con una muestra aleatoria distinta de los datos (bagging) y
  considerando un subconjunto aleatorio de características en cada
  división. Esto ataca justo la debilidad de un árbol individual (alta
  varianza / tendencia a sobreajustar a particularidades del set de
  entrenamiento), que es exactamente lo que se observa aquí: el árbol
  de decisión, solo, se queda muy por debajo del bosque completo.
- **No asume fronteras lineales**: a diferencia de la regresión
  logística, que traza fronteras de decisión lineales entre clases, un
  árbol (y por lo tanto un bosque) puede capturar relaciones no lineales
  y interacciones complejas entre las 18 características — más
  adecuado para señales de sensores biomecánicos, donde la relación
  entre las estadísticas de la señal y el tipo de movimiento no es
  lineal.
- **Menos sensible al ruido/redundancia entre features que KNN**: KNN
  clasifica según distancia entre observaciones; si algunas de las 18
  características son ruidosas o están correlacionadas entre sí, la
  distancia euclidiana pierde poder discriminativo. Un Random Forest,
  al elegir subconjuntos aleatorios de features en cada división, es más
  robusto a ese tipo de redundancia.

### 4.4 Costo razonable

A cambio de ese desempeño, Random Forest es algo menos interpretable que
un árbol individual (no se puede "leer" una sola secuencia de reglas) y
tarda un poco más en entrenar/predecir que un modelo lineal — pero con
solo 18 características y unos miles de observaciones, ese costo es
mínimo y claramente vale la pena frente a la ganancia de casi 25 puntos
de accuracy.

---

## 5. Limitaciones (ya señaladas en `resumen.json`)

> *"Selección sobre esta prueba; no es una evaluación independiente ni
> por participantes."*

Es decir:

- La comparación se hizo sobre **un solo split** de entrenamiento/prueba,
  no sobre validación cruzada (cross-validation). Los números podrían
  variar algo con otra partición aleatoria de los datos.
- No se evaluó específicamente la capacidad de **generalizar a nuevos
  participantes** (si los ciclos de train y test pertenecen a las mismas
  personas, el modelo podría estar aprendiendo patrones específicos de
  esos participantes en vez de patrones generales del movimiento).
- Los hiperparámetros de cada modelo (`k=5`, `max_depth=10`,
  `n_estimators=100`, etc.) son valores razonables pero no se afinaron
  exhaustivamente (por ejemplo, con `random_forest.py` se puede seguir
  probando `--n-estimators`, `--max-depth`, etc. para exprimir un poco
  más de desempeño).

## 6. Recomendaciones para trabajo futuro

- Repetir la comparación con **validación cruzada** (k-fold) para
  confirmar que la ventaja de Random Forest es estable y no un artefacto
  de esta partición específica.
- Si el dataset lo permite, evaluar con una **partición por
  participante** (todos los ciclos de una persona en train o en test,
  nunca mezclados) para medir generalización real a personas nuevas.
- Afinar hiperparámetros del Random Forest con `random_forest.py`
  (`--n-estimators`, `--max-depth`, `--min-samples-leaf`, `--max-features`)
  usando un conjunto de validación separado del de prueba final.
- Explorar si separar cada ciclo en ventanas de tiempo más cortas (en
  vez de un solo resumen estadístico por ciclo completo) mejora aún más
  el desempeño, ya que actualmente se pierde parte de la información
  temporal de cómo evoluciona cada movimiento.
