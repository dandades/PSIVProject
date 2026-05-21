# Football Player Team Classification

Este proyecto analiza un video de futbol para detectar jugadores validos,
mantener un identificador estable por jugador y clasificar cada jugador en
Equipo A o Equipo B segun los colores de la camiseta.

## Pipeline Actual

1. `main.py` lee el video y coordina la ejecucion completa.
2. `extract_players.py` detecta jugadores con YOLO, aplica ByteTrack y estabiliza IDs.
3. `labeling.py` entrena el clasificador de equipos sobre los tracks detectados.
4. `team_assigner/` extrae caracteristicas visuales de camiseta y aplica K-Means.
5. `trackers/` corrige identidades, filtra porteros/arbitros y dibuja las cajas finales.
6. `utils/` contiene utilidades sencillas para video y cajas.

## Estructura

- `main.py`: punto de entrada para inferencia de video.
- `extract_players.py`: deteccion, tracking, filtrado de jugadores y cache de stubs.
- `labeling.py`: asignacion de equipos a cada `track_id`.
- `manual_track_accuracy.py`: herramienta de evaluacion manual de tracks/equipos.
- `trackers/tracker.py`: fachada principal del tracker.
- `trackers/track_appearance.py`: apariencia visual usada para reidentificacion.
- `trackers/identity.py`: estabilizacion y fusion de IDs.
- `trackers/invalid_roles.py`: descarte de porteros y arbitros.
- `trackers/drawing.py`: cajas e IDs dibujados en el video.
- `team_assigner/player_features.py`: recorte, mascara verde y vector de caracteristicas.
- `team_assigner/team_model.py`: limpieza de muestras, K-Means y validacion de outliers.

## Clasificacion Por Equipos

Para cada jugador se toma la zona superior de la caja, se elimina el cesped con
una mascara verde y se extraen caracteristicas de color/textura. Despues se
agregan muestras limpias por `track_id` y se entrena K-Means con dos clusters.
El equipo queda fijado al `track_id`, no cambia frame a frame.

## Ejecucion

```bash
python main.py --input input_videos/110.mp4 --output output_videos/110_result.avi
```

Para recalcular tracks sin usar cache:

```bash
python main.py --input input_videos/110.mp4 --output output_videos/110_result.avi --no-stubs
```

Para evaluar manualmente:

```bash
python manual_track_accuracy.py --video input_videos/110.mp4 --output manual_eval/110
```
