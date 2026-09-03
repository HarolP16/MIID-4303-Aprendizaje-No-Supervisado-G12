# Taller 5 - ¿Recomendar lo que ya suena o ayudar a descubrir?

Entrega del Grupo 12 para MIID 4303. Incluye el cuaderno técnico ejecutado, la presentación ejecutiva, el código reutilizable, resultados tabulares y figuras.

## Decisión

Pilotear **Apriori** como motor principal de `Descubre`, con una línea base de popularidad solo como respaldo cuando no existan reglas suficientes. El piloto debe incorporar límites de exposición, diversidad y una prueba A/B contra popularidad. Esta es una decisión sobre el motor y sus controles, no sobre los artistas que deben mostrarse hoy: Last.fm 2011 es una base histórica y los perfiles están truncados a 50 artistas.

## Reproducibilidad

### Google Colab

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarolP16/MIID-4303-Aprendizaje-No-Supervisado-G12/blob/main/semanas/Semana-05/entrega-taller-05/Taller_05_Recomendar_descubrir_G12.ipynb)

Abra el enlace y seleccione **Entorno de ejecución → Ejecutar todas**. El notebook instala solo las dependencias faltantes y descarga automáticamente desde este repositorio el código y los datos. No requiere montar Drive, cargar archivos ni modificar rutas.

### Ejecución local

Desde esta carpeta:

```powershell
python -m pip install -r requirements.txt
python analisis_recomendadores.py
```

El análisis usa `random_state=123`, listas de diez recomendaciones y un holdout aleatorio estratificado del 20% de las interacciones observadas por usuario. La ausencia se interpreta como desconocido, no como desagrado. Los conteos se transforman con `log1p` y se normalizan por usuario para coseno/SVD.

## Archivos principales

- `Taller_05_Recomendar_descubrir_G12.ipynb`: análisis narrado y ejecutado.
- `Presentacion_Taller_05_Descubre_G12.pptx`: recomendación para el cliente.
- `analisis_recomendadores.py`: implementación completa de los cinco motores.
- `resultados/`: cifras que sustentan el cuaderno y la presentación.
- `figuras/`: visualizaciones exportadas por el análisis.
