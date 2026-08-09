from pathlib import Path
import nbformat as nbf


ROOT = Path(r"C:\Users\harol\OneDrive\Documentos\GitHub\MIID-4303-Aprendizaje-No-Supervisado-G12")
OUT = ROOT / "semanas" / "semana-01" / "entrega-taller-01"
OUT.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Caso-taller: ¿Dónde abrir una nueva sede?

## Recomendación ejecutiva

**Grupo 12 - MIID 4303, Aprendizaje no supervisado**

Con la información histórica de 1985, recomendamos que **New York, Chicago, Washington DC-MD-VA, Boston y San Francisco** avancen a *due diligence*. Las cinco ciudades ocupan los mejores lugares al promediar cinco escenarios metodológicos razonables. La recomendación combina amplitud de servicios urbanos, salud, artes, educación y transporte, pero reconoce costos de alojamiento y seguridad como trade-offs importantes.

Este cuaderno es el soporte técnico completo. Puede leerse sin ejecutarlo y reproduce todas las cifras, tablas y gráficas propuestas para una presentación. **No es una recomendación vigente:** los datos tienen más de cuatro décadas y solo sirven como primer filtro cuantitativo.

### Ruta del análisis

1. Validación, exploración y atípicos.
2. Orientación de criterios y estandarización.
3. PCA, selección e interpretación de componentes.
4. Índice y ranking de 329 ciudades.
5. Sensibilidad y selección robusta de cinco finalistas.
6. Fortalezas, debilidades, limitaciones y próximos pasos.
""")

md(r"""
## 1. Reproducibilidad y librerías

El cuaderno usa exclusivamente la ruta relativa exigida: `data/lugares.csv`. Se fija una semilla aunque PCA con `svd_solver="full"` es determinista; así la entrega queda preparada para cualquier extensión aleatoria. Los resultados se escriben en `resultados/`, sin depender de rutas del autor.
""")

code(r"""
from pathlib import Path
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEMILLA = 4303
random.seed(SEMILLA)
np.random.seed(SEMILLA)
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
sns.set_theme(style="whitegrid", context="notebook")

RUTA_DATOS = Path("data/lugares.csv")
RUTA_RESULTADOS = Path("resultados")
RUTA_RESULTADOS.mkdir(exist_ok=True)

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__, "| numpy:", np.__version__, "| sklearn:", sklearn.__version__)
print("Semilla:", SEMILLA, "| Datos:", RUTA_DATOS)
""")

md(r"""
## 2. Carga y controles de calidad

Se esperan 329 ciudades, un identificador y nueve criterios numéricos. Las aserciones detienen la ejecución si el archivo cambia, falta, contiene nulos o no conserva la estructura del caso; esto evita producir silenciosamente un ranking inválido.
""")

code(r"""
assert RUTA_DATOS.exists(), f"No se encontró {RUTA_DATOS}. El cuaderno debe estar junto a la carpeta data/."
datos = pd.read_csv(RUTA_DATOS)
variables = datos.columns.drop("Ciudad").tolist()

assert datos.shape == (329, 10), f"Dimensión inesperada: {datos.shape}"
assert len(variables) == 9
assert datos[variables].apply(pd.api.types.is_numeric_dtype).all()
assert datos.isna().sum().sum() == 0

control = pd.Series({
    "filas": len(datos),
    "columnas": datos.shape[1],
    "criterios_numéricos": len(variables),
    "faltantes": int(datos.isna().sum().sum()),
    "filas_duplicadas": int(datos.duplicated().sum()),
    "ciudades_duplicadas": int(datos["Ciudad"].duplicated().sum()),
}, name="valor").to_frame()
display(control)
display(datos.head())
""")

md(r"""
### Dirección empresarial de los criterios

En siete criterios, mayor puntuación significa mejor evaluación. En **Alojamiento** y **Crimen**, menor puntuación es mejor. Para que una cifra positiva siempre signifique mayor atractivo, esos dos criterios se multiplican por -1 antes de estandarizar. Esta transformación solo invierte sus ejes: no altera distancias ni correlaciones absolutas, pero reduce el riesgo de interpretar al revés las cargas y el índice.
""")

code(r"""
direccion = pd.DataFrame({
    "criterio": variables,
    "mejor_si": ["menor" if v in ["Alojamiento", "Crimen"] else "mayor" for v in variables],
    "transformación": ["multiplicar por -1" if v in ["Alojamiento", "Crimen"] else "sin cambio" for v in variables],
})
display(direccion)
""")

md(r"""
## 3. Exploración de los datos

Con nueve criterios existen $9(9-1)/2=36$ relaciones por pares. Antes de reducir la dimensión se revisan escalas, dispersión, asociaciones y atípicos. No se elimina ninguna ciudad de forma automática: una observación extrema puede ser una ciudad grande con un perfil real y relevante para la decisión.
""")

code(r"""
resumen = datos[variables].describe().T
resumen["rango"] = resumen["max"] - resumen["min"]
resumen["coef_variación"] = resumen["std"] / resumen["mean"]
resumen["varianza"] = datos[variables].var()

print("Relaciones por pares:", len(variables) * (len(variables) - 1) // 2)
print("Razón entre la mayor y la menor varianza original:", f"{resumen['varianza'].max()/resumen['varianza'].min():,.0f} veces")
display(resumen)
""")

code(r"""
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, variable in zip(axes.flat, variables):
    sns.histplot(datos[variable], bins=22, kde=True, ax=ax, color="#5B4B8A")
    ax.set_title(variable, fontsize=10)
fig.suptitle("Figura 1. Distribuciones de los nueve criterios (escala original)", fontsize=15, y=1.01)
plt.tight_layout()
plt.show()
""")

md(r"""
Las varianzas originales difieren por órdenes de magnitud. Sin estandarizar, PCA maximizaría principalmente la varianza de los criterios con unidades numéricas grandes, no necesariamente su información relativa. Por ello el PCA se hará sobre puntajes z, de modo que cada criterio comience con varianza comparable.
""")

code(r"""
corr = datos[variables].corr()
plt.figure(figsize=(11, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1)
plt.title("Figura 2. Matriz de correlación de los criterios")
plt.tight_layout()
plt.show()

pares = []
for i, a in enumerate(variables):
    for b in variables[i+1:]:
        pares.append((a, b, corr.loc[a, b]))
correlaciones_fuertes = pd.DataFrame(pares, columns=["criterio_1", "criterio_2", "correlación"])
correlaciones_fuertes["abs_correlación"] = correlaciones_fuertes["correlación"].abs()
display(correlaciones_fuertes.nlargest(10, "abs_correlación").drop(columns="abs_correlación"))
""")

md(r"""
La relación dominante es Salud/medio ambiente-Artes (aprox. 0.87); también aparecen asociaciones moderadas entre Salud y Educación/Transporte. Una hipótesis prudente es que ciudades con mayor infraestructura concentran simultáneamente servicios sanitarios, culturales y de movilidad. Es una asociación, no evidencia causal, y puede estar influida por tamaño urbano.
""")

code(r"""
def resumen_atipicos_iqr(df, columnas):
    filas, mascara_global = [], pd.Series(False, index=df.index)
    for col in columnas:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mascara = (df[col] < inferior) | (df[col] > superior)
        mascara_global |= mascara
        filas.append([col, int(mascara.sum()), inferior, superior])
    return pd.DataFrame(filas, columns=["criterio", "n_atípicos", "límite_inferior", "límite_superior"]), mascara_global

tabla_atipicos, es_atipica = resumen_atipicos_iqr(datos, variables)
display(tabla_atipicos)
print("Ciudades atípicas en al menos un criterio:", int(es_atipica.sum()))

X_orientado = datos[variables].copy()
X_orientado[["Alojamiento", "Crimen"]] *= -1
Z = pd.DataFrame(StandardScaler().fit_transform(X_orientado), columns=variables, index=datos.index)
datos_extremos = datos[["Ciudad"]].copy()
datos_extremos["máximo_abs_z"] = Z.abs().max(axis=1)
display(datos_extremos.nlargest(12, "máximo_abs_z"))
""")

md(r"""
Se conservan los atípicos en el modelo principal porque representan perfiles urbanos observados y eliminarlos cambiaría la población de interés. Para verificar que no dicten por sí solos el resultado, más adelante se reestimará el escalador y el PCA sin las ciudades marcadas por IQR, pero se puntuarán nuevamente las 329.
""")

md(r"""
## 4. Estandarización

Cada criterio orientado se centra y divide por su desviación estándar poblacional. Así una unidad representa una desviación estándar respecto al conjunto de 329 ciudades. Esta escala también permite describir fortalezas y debilidades con una métrica común.
""")

code(r"""
escalador = StandardScaler()
X_std = escalador.fit_transform(X_orientado)
X_std = pd.DataFrame(X_std, columns=variables, index=datos.index)

verificacion = X_std.agg(["mean", lambda s: s.std(ddof=0)]).T
verificacion.columns = ["media", "desv_estándar_poblacional"]
display(verificacion)
assert np.allclose(X_std.mean(), 0, atol=1e-12)
assert np.allclose(X_std.std(ddof=0), 1, atol=1e-12)
""")

md(r"""
## 5. PCA y número de componentes

Se ajustan los nueve componentes con SVD completa. La selección no aplica una regla mecánicamente: se contrastan Kaiser, sedimentación, varianza acumulada e interpretabilidad.
""")

code(r"""
pca = PCA(svd_solver="full")
scores_total = pca.fit_transform(X_std)
proporcion = pca.explained_variance_ratio_
acumulada = proporcion.cumsum()
eigenvalues = pca.explained_variance_

tabla_varianza = pd.DataFrame({
    "componente": [f"CP{i}" for i in range(1, 10)],
    "eigenvalue": eigenvalues,
    "varianza_explicada": proporcion,
    "varianza_acumulada": acumulada,
})
n_kaiser = int((eigenvalues > 1).sum())
n_80 = int(np.argmax(acumulada >= 0.80) + 1)

display(tabla_varianza)
print("Kaiser (eigenvalue > 1):", n_kaiser, "componentes")
print("Mínimo para explicar al menos 80%:", n_80, "componentes")
""")

code(r"""
fig, ax1 = plt.subplots(figsize=(10, 5.5))
x = np.arange(1, 10)
ax1.plot(x, proporcion, marker="o", linewidth=2, label="Varianza individual")
ax1.plot(x, acumulada, marker="s", linewidth=2, label="Varianza acumulada")
ax1.axhline(0.80, color="gray", linestyle="--", label="Umbral 80%")
ax1.axvline(n_80, color="#C44E52", linestyle=":", label=f"Retención: {n_80} CP")
ax1.set(xticks=x, xlabel="Número de componentes", ylabel="Proporción", ylim=(0, 1.03))
ax1.set_title("Figura 3. Sedimentación y varianza acumulada")
ax1.legend(loc="center right")
plt.tight_layout()
plt.show()
""")

md(r"""
### Decisión: retener cinco componentes

Kaiser retiene 3 CP, pero solo explica cerca de **64.0%**. Cinco CP alcanzan **82.6%**, superan el umbral descriptivo del 80% y conservan dimensiones interpretables de seguridad, vivienda, recreación y transporte que se perderían con tres. Los cinco se usan para describir y construir el índice principal; el modelo de tres CP queda como escenario de sensibilidad.
""")

md(r"""
## 6. Cargas e interpretación

Las cargas siguientes son los coeficientes de las combinaciones lineales. El signo global de una CP es arbitrario; la interpretación depende de qué criterios aparecen juntos o contrapuestos, no de que el signo sea positivo en sí mismo.
""")

code(r"""
n_componentes = n_80
cargas = pd.DataFrame(
    pca.components_.T,
    index=variables,
    columns=[f"CP{i}" for i in range(1, 10)],
)
cargas_retenidas = cargas.iloc[:, :n_componentes]
display(cargas_retenidas.round(3))

plt.figure(figsize=(10, 6.5))
sns.heatmap(cargas_retenidas, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Figura 4. Cargas de los cinco componentes retenidos")
plt.tight_layout()
plt.show()
""")

md(r"""
### Nombres prudentes de las dimensiones

- **CP1 - servicios metropolitanos y cultura vs. costo/seguridad:** Salud, Artes, Transporte y Recreación se mueven juntos; se contraponen a vivienda asequible y seguridad.
- **CP2 - educación/seguridad vs. recreación/economía:** concentra Educación y seguridad, opuestas a Recreación y Economía.
- **CP3 - clima/terreno vs. economía:** es la contraposición más clara entre esos dos criterios.
- **CP4 - seguridad/economía vs. vivienda/transporte:** combina seguridad y economía, contrapuestas a asequibilidad y transporte.
- **CP5 - recreación/seguridad/conectividad:** reúne Recreación, seguridad y Transporte, contrapuestos parcialmente a Clima.

Los nombres describen asociaciones de 1985, no conceptos universales ni relaciones causales.
""")

code(r"""
scores = pd.DataFrame(scores_total[:, :n_componentes], columns=cargas_retenidas.columns)
scores.insert(0, "Ciudad", datos["Ciudad"])

extremos_componentes = []
for cp in cargas_retenidas.columns:
    altos = scores.nlargest(3, cp)[["Ciudad", cp]].assign(extremo="alto")
    bajos = scores.nsmallest(3, cp)[["Ciudad", cp]].assign(extremo="bajo")
    extremos_componentes.append(pd.concat([altos, bajos]).rename(columns={cp: "puntuación"}).assign(componente=cp))
extremos_componentes = pd.concat(extremos_componentes, ignore_index=True)
display(extremos_componentes[["componente", "extremo", "Ciudad", "puntuación"]])
""")

md(r"""
## 7. Índice de atractivo y ranking

PCA describe la estructura, pero no decide qué le importa al comité. Como el caso no entrega preferencias cuantificadas, el escenario principal asigna **peso igual (1/9) a los nueve criterios ya orientados**. Es la regla neutral más transparente: no confunde varianza estadística con importancia empresarial.

Para usar PCA como insumo, se reconstruye cada ciudad con las cinco CP retenidas y se calcula el promedio de sus nueve criterios reconstruidos. Algebraicamente, si $T_5$ son las puntuaciones, $V_5$ las cargas y $w$ el vector de pesos, el índice es $T_5V_5w$. Así PCA filtra el 17.4% de variación de menor prioridad descriptiva, mientras $w$ hace explícita la preferencia de negocio. El resultado se estandariza y se ordena de mayor a menor.
""")

code(r"""
pesos_iguales = np.repeat(1 / len(variables), len(variables))
coeficientes_cp = pca.components_[:n_componentes] @ pesos_iguales
indice_crudo = scores_total[:, :n_componentes] @ coeficientes_cp
indice_z = (indice_crudo - indice_crudo.mean()) / indice_crudo.std(ddof=0)

tabla_pesos = pd.DataFrame({"criterio": variables, "peso_empresarial": pesos_iguales})
tabla_coef_cp = pd.DataFrame({
    "componente": [f"CP{i}" for i in range(1, n_componentes + 1)],
    "coeficiente_en_índice": coeficientes_cp,
})
display(tabla_pesos)
display(tabla_coef_cp)
""")

code(r"""
ranking = datos.copy()
ranking["Indice_atractivo_z"] = indice_z
ranking["Ranking"] = ranking["Indice_atractivo_z"].rank(method="min", ascending=False).astype(int)
ranking = ranking.sort_values(["Ranking", "Ciudad"]).reset_index(drop=True)

columnas_ranking = ["Ranking", "Ciudad", "Indice_atractivo_z"] + variables
print("Diez primeras ciudades")
display(ranking[columnas_ranking].head(10))
print("Diez últimas ciudades")
display(ranking[columnas_ranking].tail(10))
""")

md(r"""
El liderazgo de grandes áreas metropolitanas debe contrastarse con las cargas: el índice recompensa su concentración excepcional de salud, cultura y transporte, pero también registra penalizaciones de vivienda y crimen. El ranking es un filtro multicriterio, no una afirmación de que la primera ciudad domina en cada aspecto.
""")

md(r"""
## 8. Sensibilidad y robustez

Se prueban cuatro decisiones frente al principal:

1. **Promedio directo:** no descartar el 17.4% de variación restante.
2. **PCA-Kaiser:** retener solo 3 CP.
3. **Prioridad talento/familias:** mayor peso a Salud, Educación y seguridad; peso intermedio a Alojamiento, Transporte y Economía.
4. **Ajuste sin atípicos:** entrenar el escalador y PCA sin las 81 ciudades marcadas por IQR y volver a puntuar las 329.

Las correlaciones de Spearman evalúan el orden completo; la coincidencia *top 10* evalúa la zona decisiva.
""")

code(r"""
def estandarizar_vector(x):
    x = np.asarray(x)
    return (x - x.mean()) / x.std(ddof=0)

escenarios = pd.DataFrame({"Ciudad": datos["Ciudad"]})
escenarios["Principal_PCA80"] = estandarizar_vector(indice_crudo)
escenarios["Promedio_directo"] = estandarizar_vector(X_std.to_numpy() @ pesos_iguales)

coef_kaiser = pca.components_[:n_kaiser] @ pesos_iguales
escenarios["PCA_Kaiser"] = estandarizar_vector(scores_total[:, :n_kaiser] @ coef_kaiser)

pesos_talento = pd.Series(1.0, index=variables)
pesos_talento[["Cuidado de la salud y el medio ambiente", "Educación", "Crimen"]] = 1.5
pesos_talento[["Alojamiento", "Transporte", "Economía"]] = 1.2
pesos_talento = (pesos_talento / pesos_talento.sum()).to_numpy()
coef_talento = pca.components_[:n_componentes] @ pesos_talento
escenarios["Prioridad_talento"] = estandarizar_vector(scores_total[:, :n_componentes] @ coef_talento)

X_entrenamiento = X_orientado.loc[~es_atipica]
escalador_sin = StandardScaler().fit(X_entrenamiento)
Z_entrenamiento = escalador_sin.transform(X_entrenamiento)
pca_sin = PCA(svd_solver="full").fit(Z_entrenamiento)
n_80_sin = int(np.argmax(pca_sin.explained_variance_ratio_.cumsum() >= 0.80) + 1)
scores_sin = pca_sin.transform(escalador_sin.transform(X_orientado))
coef_sin = pca_sin.components_[:n_80_sin] @ pesos_iguales
escenarios["Ajuste_sin_atípicos"] = estandarizar_vector(scores_sin[:, :n_80_sin] @ coef_sin)

nombres_escenarios = [c for c in escenarios.columns if c != "Ciudad"]
for nombre in nombres_escenarios:
    escenarios[f"Rank_{nombre}"] = escenarios[nombre].rank(method="min", ascending=False)

columnas_rank = [f"Rank_{c}" for c in nombres_escenarios]
corr_rank = escenarios[columnas_rank].corr(method="spearman")
display(corr_rank)

plt.figure(figsize=(9, 7))
sns.heatmap(corr_rank, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1)
plt.title("Figura 5. Correlación de Spearman entre rankings")
plt.tight_layout()
plt.show()
""")

code(r"""
top_principal = set(escenarios.nlargest(10, "Principal_PCA80")["Ciudad"])
coincidencias = []
for nombre in nombres_escenarios[1:]:
    top_alterno = set(escenarios.nlargest(10, nombre)["Ciudad"])
    coincidencias.append({
        "escenario": nombre,
        "coincidencias_top10": len(top_principal & top_alterno),
        "porcentaje": len(top_principal & top_alterno) / 10,
    })
coincidencias = pd.DataFrame(coincidencias)
display(coincidencias)

estabilidad = escenarios[["Ciudad"] + columnas_rank].copy()
estabilidad["Ranking_promedio"] = estabilidad[columnas_rank].mean(axis=1)
estabilidad["Desv_ranking"] = estabilidad[columnas_rank].std(axis=1)
estabilidad["Mejor_ranking"] = estabilidad[columnas_rank].min(axis=1)
estabilidad["Peor_ranking"] = estabilidad[columnas_rank].max(axis=1)
display(estabilidad.sort_values(["Ranking_promedio", "Desv_ranking"]).head(15))
""")

md(r"""
Las correlaciones del orden completo son altas (aprox. 0.86 a 0.98 frente al escenario principal). Las primeras cinco por ranking promedio se mantienen entre los seis primeros puestos en todos los escenarios. Por tanto, la recomendación no depende de una sola regla de retención, ponderación o tratamiento de atípicos.
""")

md(r"""
## 9. Cinco finalistas

La selección usa, en orden: menor ranking promedio en los cinco escenarios, menor dispersión y mejor ranking principal. Esta regla privilegia robustez y es previa a observar los nombres; no se sustituyen ciudades por juicio subjetivo posterior.
""")

code(r"""
finalistas = (
    estabilidad
    .merge(ranking[["Ciudad", "Ranking", "Indice_atractivo_z"]], on="Ciudad", how="left")
    .sort_values(["Ranking_promedio", "Desv_ranking", "Ranking", "Ciudad"])
    .head(5)
    .reset_index(drop=True)
)
finalistas.insert(0, "Orden_recomendación", range(1, 6))
display(finalistas)

ciudades_finalistas = finalistas["Ciudad"].tolist()
assert ciudades_finalistas == ["New-York,NY", "Chicago,IL", "Washington,DC-MD-VA", "Boston,MA", "San-Francisco,CA"]
""")

code(r"""
perfiles = X_std.copy()
perfiles.insert(0, "Ciudad", datos["Ciudad"])
perfil_finalistas = perfiles.set_index("Ciudad").loc[ciudades_finalistas]

plt.figure(figsize=(14, 5.5))
sns.heatmap(perfil_finalistas, annot=True, fmt=".1f", cmap="vlag", center=0)
plt.title("Figura 6. Fortalezas y debilidades relativas de las cinco finalistas")
plt.xlabel("Criterio orientado: valores positivos son mejores")
plt.ylabel("")
plt.tight_layout()
plt.show()

resumen_finalistas = []
for ciudad in ciudades_finalistas:
    perfil = perfil_finalistas.loc[ciudad].sort_values(ascending=False)
    info = finalistas.loc[finalistas["Ciudad"] == ciudad].iloc[0]
    resumen_finalistas.append({
        "Ciudad": ciudad,
        "Ranking_principal": int(info["Ranking"]),
        "Ranking_promedio": info["Ranking_promedio"],
        "Peor_ranking": int(info["Peor_ranking"]),
        "Fortalezas_relativas": "; ".join(f"{v} ({perfil[v]:.2f} z)" for v in perfil.head(3).index),
        "Debilidades_relativas": "; ".join(f"{v} ({perfil[v]:.2f} z)" for v in perfil.tail(2).index),
    })
resumen_finalistas = pd.DataFrame(resumen_finalistas)
display(resumen_finalistas)
""")

md(r"""
### Lectura ejecutiva y trade-offs

- **New York:** fortaleza extraordinaria en Artes, Salud y Transporte; sus penalizaciones extremas en Crimen y Alojamiento exigen validación prioritaria. Su primer lugar no equivale a ausencia de riesgo.
- **Chicago:** combina Artes, Salud, Transporte y Educación; cede en Alojamiento y Economía.
- **Washington DC-MD-VA:** sobresale en Artes, Salud, Transporte y Educación; sus principales costos relativos son Alojamiento y Crimen.
- **Boston:** perfil fuerte y relativamente balanceado en Salud, Artes, Educación, Transporte y Recreación; Alojamiento y Crimen quedan bajo el promedio.
- **San Francisco:** lidera en Recreación, Clima y Transporte, con Salud, Artes y Educación también favorables; enfrenta el trade-off más fuerte de Alojamiento y un resultado débil en Crimen.

En conjunto, la lista ofrece capacidad urbana y atracción de talento, pero concentra ciudades costosas. La fase siguiente debe comprobar si sus ventajas justifican costos actuales y si existen alternativas con mejor asequibilidad.
""")

md(r"""
## 10. Limitaciones y decisión solicitada

### Limitaciones

- Los datos son de **1985**; no permiten recomendar una sede actual sin actualización.
- Los nueve puntajes agregados no informan definiciones, error de medición ni incertidumbre.
- No se incluyen salarios, disponibilidad de talento, alquiler de oficinas, impuestos, incentivos, conectividad digital, capacidad de expansión ni aceptación de traslado.
- El peso igual es neutral y auditable, pero no reemplaza preferencias explícitas del comité.
- PCA es lineal, sensible a la matriz de correlación y resume variación; no estima causalidad ni retorno financiero.
- Varias áreas corresponden a conglomerados metropolitanos, no necesariamente a límites administrativos comparables.

### Información para *due diligence*

Actualizar seguridad, vivienda y oficinas; disponibilidad/costo de talento; conectividad aérea, vial y digital; impuestos e incentivos; riesgos climáticos; capacidad de expansión; servicios para familias; y disposición real de empleados a trasladarse. Incluir una ciudad alternativa de alta asequibilidad como contraste comercial.

### Decisión solicitada

Autorizar estudios actuales, visitas y modelación financiera para las cinco finalistas, sin aprobar aún una sede definitiva. El comité debe además validar o modificar los pesos del índice antes de la decisión final.
""")

md(r"""
## 11. Exportación para trazabilidad

Se guardan las tablas que pueden alimentar una presentación. Cualquier cifra presentada debe provenir de estos archivos o de una salida visible del cuaderno.
""")

code(r"""
ranking.to_csv(RUTA_RESULTADOS / "ranking_completo_329_ciudades.csv", index=False)
resumen_finalistas.to_csv(RUTA_RESULTADOS / "cinco_ciudades_finalistas.csv", index=False)
tabla_varianza.to_csv(RUTA_RESULTADOS / "varianza_explicada_pca.csv", index=False)
cargas_retenidas.to_csv(RUTA_RESULTADOS / "cargas_componentes_retenidos.csv")
escenarios.to_csv(RUTA_RESULTADOS / "sensibilidad_rankings.csv", index=False)
perfil_finalistas.to_csv(RUTA_RESULTADOS / "perfiles_finalistas_z.csv")
correlaciones_fuertes.sort_values("abs_correlación", ascending=False).to_csv(
    RUTA_RESULTADOS / "correlaciones_criterios.csv", index=False
)

archivos = sorted(p.name for p in RUTA_RESULTADOS.glob("*.csv"))
print("Archivos generados:")
for archivo in archivos:
    print(" -", archivo)
assert len(archivos) == 7
print("\nValidación final: cuaderno ejecutado de principio a fin sin errores.")
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

dest = OUT / "Taller_01_Donde_abrir_nueva_sede_G12.ipynb"
nbf.write(nb, dest)
print(dest)
