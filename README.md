# Football Player Detection, Tracking & Team Classification in Video

Este proyecto analiza un video de futbol para detectar jugadores validos,
mantener un identificador estable por jugador y clasificar cada jugador en
Equipo A o Equipo B segun los colores de la camiseta.

## Pipeline

1. `main.py` lee el video y coordina la ejecucion completa.
2. `extract_players.py` detecta jugadores con YOLO, aplica ByteTrack y usa `stubs/` como cache de tracks.
3. `trackers/` estabiliza IDs, fusiona tracks fragmentados y filtra porteros/arbitros.
4. `labeling.py` asigna un equipo estable a cada `track_id`.
5. `team_assigner/` extrae caracteristicas visuales de camiseta y aplica K-Means para separar Equipo A y Equipo B.
6. `trackers/drawing.py` dibuja las inferencias finales en el video de resultado: cajas, IDs y equipo asignado.
7. `manual_track_accuracy.py` genera evidencias para evaluar manualmente tracks y equipos.

## Librerias

- `OpenCV`: lectura, escritura y manipulacion de frames de video.
- `Ultralytics YOLO`: deteccion de jugadores, porteros y arbitros.
- `Supervision`: integracion con ByteTrack para el seguimiento.
- `NumPy`: calculos numericos sobre cajas, posiciones y features visuales.
- `Scikit-learn`: K-Means y normalizacion para clasificar equipos.
- `PyTorch`: backend usado por el modelo YOLO.

## Ejecucion

Procesar el video por defecto:

```bash
python main.py
```

Indicar entrada y salida:

```bash
python main.py --input input_videos/110.mp4 --output output_videos/110_result.avi
```

Recalcular tracks sin usar cache:

```bash
python main.py --input input_videos/110.mp4 --output output_videos/110_result.avi --no-stubs
```

Usar otro modelo o cambiar el minimo de frames por track:

```bash
python main.py --model models/best.pt --min-track-frames 8
```

## Evaluacion Manual

Genera una evidencia por cada `track_id` y guarda resultados en CSV/JSON:

```bash
python manual_track_accuracy.py --video input_videos/110.mp4
```

Solo generar evidencias, sin revision interactiva:

```bash
python manual_track_accuracy.py --video input_videos/110.mp4 --collect-only
```

Opciones durante la revision:

| Tecla | Significado |
| --- | --- |
| `s` | Jugador valido y equipo correcto. |
| `n` | Jugador valido, equipo incorrecto. |
| `t` | No es jugador valido. |
| `k` | Saltar ID. |
| `q` | Salir y guardar. |
