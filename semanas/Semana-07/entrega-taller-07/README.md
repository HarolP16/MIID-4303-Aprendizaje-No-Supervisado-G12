# Entrega semana 7 - Mapas de calor y patrullaje

Esta carpeta contiene los dos productos solicitados en la guía:

- `Taller_07_Mapas_Calor_Patrullaje_G12.ipynb`: cuaderno técnico ejecutado, con todas las preguntas respondidas.
- `Presentacion_Ejecutiva_Taller_07_G12.pptx`: presentación de 12 diapositivas orientada a la decisión del cliente.

## Recomendación

Probar un esquema híbrido en Bogotá: KDE bivariada con `cv_ml` para localizar concentraciones y conteos/tasas territoriales para controlar estabilidad, auditar y comunicar. Robos y homicidios se analizan por separado. La implementación propuesta es un piloto de 8 a 12 semanas con microterritorios comparables y evaluación antes de escalar.

## Reproducción

1. Abrir una terminal en esta carpeta.
2. Instalar las dependencias de `requirements.txt`.
3. Ejecutar todas las celdas del cuaderno en orden.

El cuaderno lee únicamente rutas relativas a `data/`, usa `random_state=123` y guarda tablas en `resultados/` y figuras en `figuras/`.

## Archivos de datos

- `data/Chicago_delitos_verano_2019.csv`
- `data/Areas_comunitarias_Chicago.zip`
- `data/osm_actividad_comercial_chicago.json`

La descarga de OpenStreetMap corresponde a una instantánea consultada en 2026. Se usa como proxy exploratorio de actividad comercial y no como medición causal contemporánea a los delitos de 2019.

## Resultado central

- 17.747 delitos: 17.603 robos y 144 homicidios.
- Los cinco territorios con más eventos concentran 31,7% del total.
- Robos y homicidios presentan baja correlación lineal por área: `r = 0,17`.
- La actividad comercial OSM se asocia con robos (`r = 0,84`), pero no con homicidios (`r = -0,12`).
- KDE con `cv_ml` ocupa el primer lugar del ranking operativo, con controles de estabilidad obligatorios.

## Alcance

El análisis identifica asociaciones y candidatos para focalización. No demuestra que un cambio de patrullaje cause una reducción del delito. Esa pregunta requiere el piloto y una evaluación causal.
