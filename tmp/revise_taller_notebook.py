from pathlib import Path
import nbformat as nbf


PATH = Path(r"C:\Users\harol\OneDrive\Documentos\GitHub\MIID-4303-Aprendizaje-No-Supervisado-G12\semanas\semana-01\entrega-taller-01\Taller_01_Donde_abrir_nueva_sede_G12.ipynb")
nb = nbf.read(PATH, as_version=4)


def find(prefix):
    for i, cell in enumerate(nb.cells):
        if cell.source.strip().startswith(prefix):
            return i
    raise ValueError(prefix)


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


# La portada presenta el problema, pero no revela resultados antes de producirlos.
nb.cells[find("# Caso-taller:")].source = r"""
# Caso-taller: ¿Dónde abrir una nueva sede?

**Grupo 12 - MIID 4303, Aprendizaje no supervisado**

Una empresa quiere reducir 329 ciudades candidatas a cinco opciones para una fase de *due diligence*. Este cuaderno construye el filtro con los nueve criterios disponibles y muestra, en este orden, los datos, los patrones encontrados, el ranking y la recomendación.

La pregunta que guía el trabajo es:

> **¿Qué cinco ciudades muestran el perfil más atractivo y estable, y qué ventajas y riesgos tiene cada una?**

Los datos corresponden a 1985. Por eso los hallazgos no se presentan como una respuesta actual, sino como una forma reproducible de escoger dónde investigar primero.

### Recorrido

1. Conocer y revisar los datos.
2. Hacer comparables los nueve criterios.
3. Resumir los patrones con PCA.
4. Construir y visualizar el ranking.
5. Comprobar si cambia con otras decisiones.
6. Elegir cinco finalistas y explicar sus fortalezas y riesgos.
""".strip()

# Lenguaje de hallazgos más directo y menos académico.
replacements = {
    "Las varianzas originales difieren": r"""
### Hallazgo: las escalas no son comparables

Algunas columnas varían miles de veces más que otras. Si se usaran tal como vienen, las de números grandes dominarían el resultado. Estandarizar evita que la unidad de medida decida por nosotros y da a los nueve criterios el mismo punto de partida.
""",
    "La relación dominante es": r"""
### Hallazgo: los servicios urbanos aparecen juntos

Salud/medio ambiente y Artes tienen la relación más fuerte. Salud también se mueve con Educación y Transporte. En términos sencillos, las ciudades con más infraestructura suelen reunir varios servicios a la vez. Esto ayuda a explicar por qué algunos centros metropolitanos sobresalen, pero no demuestra que un criterio cause el otro.
""",
    "Se conservan los atípicos": r"""
### Hallazgo: las ciudades extremas son parte de la historia

Los casos extremos no parecen errores de digitación; son ciudades con perfiles poco comunes. Se mantienen porque pueden ser candidatas reales. Más adelante se repetirá el ajuste sin ellas para comprobar si cambian la lista final.
""",
    "### Decisión: retener cinco componentes": r"""
### Hallazgo: cinco componentes conservan una visión suficientemente amplia

Tres componentes resumen cerca del 64% de las diferencias, pero dejan por fuera señales útiles de vivienda, seguridad, recreación y transporte. Con cinco se conserva aproximadamente el 83%. Es un equilibrio razonable entre simplificar la información y no borrar contrastes importantes para una sede.
""",
    "### Nombres prudentes": r"""
### ¿Qué historias cuentan los componentes?

- **CP1 - oferta urbana frente a costo y seguridad:** las ciudades con mucha salud, cultura, transporte y recreación tienden a pagar un precio en vivienda o seguridad.
- **CP2 - educación y seguridad frente a actividad económica/recreativa:** muestra otro tipo de equilibrio entre vida familiar y dinamismo urbano.
- **CP3 - clima frente a economía:** separa con claridad esos dos perfiles.
- **CP4 - seguridad y economía frente a vivienda y transporte:** recuerda que una ciudad no suele ganar en todo al mismo tiempo.
- **CP5 - recreación, seguridad y conexión:** recoge una dimensión de calidad de vida y movilidad.

Estos nombres son ayudas para leer los datos de 1985, no etiquetas definitivas sobre las ciudades.
""",
    "El liderazgo de grandes áreas": r"""
### Hallazgo: el ranking favorece ciudades completas, no ciudades perfectas

Las primeras posiciones corresponden a ciudades con una oferta urbana muy amplia. Sin embargo, sus buenos resultados en salud, cultura y transporte conviven con debilidades claras en vivienda y seguridad. El ranking muestra un balance; no afirma que la primera ciudad sea la mejor en cada criterio.
""",
    "Las correlaciones del orden completo": r"""
### Hallazgo: la parte alta del ranking es estable

Aunque cambian la cantidad de componentes, los pesos y el tratamiento de casos extremos, los rankings se parecen bastante. Las cinco ciudades seleccionadas nunca caen más allá del sexto puesto. Esto da más confianza en la lista corta que en el orden exacto entre sus integrantes.
""",
    "### Lectura ejecutiva y trade-offs": r"""
### ¿Qué gana y qué arriesga la empresa con cada finalista?

- **New York:** oferta excepcional de artes, salud y transporte. El costo de vivienda y la seguridad son alertas serias; estar de primera no significa ser una opción sin riesgo.
- **Chicago:** combina cultura, salud, transporte y educación. Sus puntos débiles están en vivienda y economía.
- **Washington DC-MD-VA:** sobresale en cultura, salud, transporte y educación, pero pierde atractivo por vivienda y crimen.
- **Boston:** tiene el perfil más equilibrado entre salud, cultura, educación, transporte y recreación; vivienda y crimen siguen bajo el promedio.
- **San Francisco:** destaca en recreación, clima y transporte. Su principal riesgo es la vivienda, acompañado por un resultado débil en crimen.

La lista favorece capacidad urbana y atracción de talento, pero concentra lugares costosos. La siguiente fase debe comprobar si esas ventajas todavía compensan sus costos actuales.
""",
}
for prefix, text in replacements.items():
    nb.cells[find(prefix)].source = text.strip()

# Renumerar las figuras existentes para abrir espacio a las nuevas.
title_changes = {
    "Figura 2. Matriz": "Figura 3. Matriz",
    "Figura 3. Sedimentación": "Figura 4. Sedimentación",
    "Figura 4. Cargas": "Figura 5. Cargas",
    "Figura 5. Correlación": "Figura 8. Correlación",
    "Figura 6. Fortalezas": "Figura 10. Fortalezas",
}
for cell in nb.cells:
    if cell.cell_type == "code":
        for old, new in title_changes.items():
            cell.source = cell.source.replace(old, new)

# Figura 2: atípicos por criterio, insertada después de su identificación.
i = find("def resumen_atipicos_iqr") + 1
nb.cells.insert(i, code(r"""
z_largo = (
    Z.assign(Ciudad=datos["Ciudad"])
    .melt(id_vars="Ciudad", var_name="Criterio", value_name="Puntaje_z")
)
plt.figure(figsize=(12, 7))
sns.boxplot(data=z_largo, x="Puntaje_z", y="Criterio", color="#8DA0CB", showfliers=True)
plt.axvline(0, color="black", linewidth=1)
plt.title("Figura 2. Dispersión y valores extremos por criterio")
plt.xlabel("Desviaciones estándar respecto al promedio")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""))

# Figuras 6 y 7: ranking y mapa PCA, después de construir el ranking.
i = find('ranking = datos.copy()') + 1
nb.cells.insert(i, code(r"""
top15_grafico = ranking.head(15).sort_values("Indice_atractivo_z")
plt.figure(figsize=(10, 7))
ax = sns.barplot(data=top15_grafico, x="Indice_atractivo_z", y="Ciudad", color="#4C72B0")
ax.axvline(0, color="black", linewidth=0.8)
ax.bar_label(ax.containers[0], fmt="%.2f", padding=3, fontsize=8)
plt.title("Figura 6. Las 15 ciudades con mayor índice de atractivo")
plt.xlabel("Índice de atractivo (puntaje z)")
plt.ylabel("")
plt.tight_layout()
plt.show()

mapa_pca = scores[["Ciudad", "CP1", "CP2"]].merge(
    ranking[["Ciudad", "Ranking"]], on="Ciudad", how="left"
)
plt.figure(figsize=(11, 7))
sns.scatterplot(data=mapa_pca, x="CP1", y="CP2", color="#B8B8B8", alpha=0.65, s=38)
destacadas = mapa_pca.nsmallest(10, "Ranking")
sns.scatterplot(
    data=destacadas, x="CP1", y="CP2", hue="Ranking", palette="viridis_r",
    s=100, edgecolor="black", legend=False
)
for _, fila in destacadas.head(7).iterrows():
    plt.annotate(fila["Ciudad"], (fila["CP1"], fila["CP2"]), xytext=(5, 5),
                 textcoords="offset points", fontsize=8)
plt.axhline(0, color="gray", linewidth=0.7)
plt.axvline(0, color="gray", linewidth=0.7)
plt.title("Figura 7. Mapa de ciudades en las dos dimensiones principales")
plt.xlabel("CP1: oferta urbana frente a costo/seguridad")
plt.ylabel("CP2: educación/seguridad frente a economía/recreación")
plt.tight_layout()
plt.show()
"""))

# Figura 9: muestra cómo se mueven las finalistas entre escenarios.
i = find("finalistas = (") + 1
nb.cells.insert(i, code(r"""
orden_escenarios = [f"Rank_{c}" for c in nombres_escenarios]
etiquetas_escenarios = {
    "Rank_Principal_PCA80": "Principal",
    "Rank_Promedio_directo": "Promedio directo",
    "Rank_PCA_Kaiser": "Kaiser",
    "Rank_Prioridad_talento": "Talento/familias",
    "Rank_Ajuste_sin_atípicos": "Sin atípicos",
}
estabilidad_finalistas = (
    estabilidad[estabilidad["Ciudad"].isin(ciudades_finalistas)]
    .melt(id_vars="Ciudad", value_vars=orden_escenarios,
          var_name="Escenario", value_name="Posición")
)
estabilidad_finalistas["Escenario"] = estabilidad_finalistas["Escenario"].map(etiquetas_escenarios)
estabilidad_finalistas["Escenario"] = pd.Categorical(
    estabilidad_finalistas["Escenario"], categories=list(etiquetas_escenarios.values()), ordered=True
)
plt.figure(figsize=(11, 6))
sns.lineplot(data=estabilidad_finalistas, x="Escenario", y="Posición", hue="Ciudad",
             marker="o", linewidth=2.2)
plt.gca().invert_yaxis()
plt.yticks(range(1, 8))
plt.title("Figura 9. Posición de las cinco finalistas en cada escenario")
plt.xlabel("")
plt.ylabel("Posición (1 es mejor)")
plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
"""))

# Figura 11: perfiles individuales más fáciles de explicar que una tabla extensa.
i = find("perfiles = X_std.copy()") + 1
nb.cells.insert(i, code(r"""
fig, axes = plt.subplots(len(ciudades_finalistas), 1, figsize=(12, 17), sharex=True)
for ax, ciudad in zip(axes, ciudades_finalistas):
    perfil = perfil_finalistas.loc[ciudad].sort_values()
    colores = ["#C44E52" if valor < 0 else "#55A868" for valor in perfil]
    ax.barh(perfil.index, perfil.values, color=colores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(ciudad, loc="left", fontweight="bold")
    ax.set_ylabel("")
axes[-1].set_xlabel("Desviaciones estándar; verde = fortaleza, rojo = debilidad")
fig.suptitle("Figura 11. Balance de fortalezas y debilidades por finalista", fontsize=15, y=1.002)
plt.tight_layout()
plt.show()
"""))

# Limpiar salidas para que la nueva ejecución pruebe todas las celdas desde cero.
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

nbf.write(nb, PATH)
print(PATH)
