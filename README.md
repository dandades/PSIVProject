# Football Player Detection, Tracking & Team Classification in Video

Proyecto para analizar videos de futbol: detecta jugadores, mantiene un
`track_id` estable y clasifica cada jugador en Equipo A o Equipo B segun la
apariencia de la camiseta.

Usa YOLO para deteccion, ByteTrack para seguimiento y K-Means sobre
caracteristicas de color/textura para separar los equipos.

## Estructura

- `main.py`: punto de entrada para procesar un video.
- `extract_players.py`: deteccion, tracking y cache de tracks.
- `labeling.py`: asignacion de equipo por `track_id`.
- `manual_track_accuracy.py`: evaluacion manual de tracks y equipos.
- `trackers/`: logica de tracking, reidentificacion, filtrado y dibujo.
- `team_assigner/`: extraccion de features y clasificacion por equipos.
- `training/`: notebooks y dataset para entrenar el detector YOLO.
- `models/best.pt`: modelo YOLO esperado por defecto.

## Librerias Principales

- `OpenCV`: lectura, escritura y manipulacion de frames de video.
- `Ultralytics YOLO`: deteccion de jugadores, porteros y arbitros.
- `Supervision`: integracion con ByteTrack para el seguimiento.
- `NumPy`: calculos numericos sobre cajas, posiciones y features visuales.
- `Scikit-learn`: K-Means y normalizacion para clasificar equipos.
- `PyTorch`: backend usado por el modelo YOLO.

## Instalacion

```bash
python -m venv .venv
```

En Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

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
