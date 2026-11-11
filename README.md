# LSTM/bolsa

Página web que entrena en vivo una red **LSTM en PyTorch** sobre precios diarios reales (Yahoo Finance) y predice la evolución de una acción, índice o criptomoneda. Deep learning aplicado a series temporales reales.

## Qué hace

- Descarga el histórico del ticker elegido (AAPL, MSFT, SAN.MC, ^IBEX, BTC-USD…).
- Entrena un LSTM de 2 capas que predice el **log-retorno del día siguiente** a partir de una ventana de días anteriores (retorno, rango intradía y cambio de volumen).
- Evalúa en un periodo de test que el modelo nunca vio y lo compara con un **baseline ingenuo** ("mañana = hoy").
- Pronostica los próximos N días con una banda de incertidumbre p10–p90 calculada con **Monte Carlo Dropout**.
- Muestra gráficos del precio real frente al predicho, el pronóstico y la curva de aprendizaje.

## Decisiones de diseño

- **Retornos en lugar de precios:** predecir el precio directamente produce gráficos bonitos pero engañosos, porque el modelo aprende a copiar el precio de ayer.
- **Split temporal** (nunca aleatorio) y normalización calculada solo con el train, para evitar fuga de información del futuro.
- Pérdida Huber, AdamW, `ReduceLROnPlateau`, clipping de gradiente y selección del mejor modelo por validación.
- Usa la GPU de Apple (MPS) o CUDA si están disponibles.

## Uso

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Abre <http://localhost:5050>.

También puedes entrenar desde la terminal:

```bash
python model.py AAPL
```

## Estructura

| Archivo | Contenido |
|---|---|
| `model.py` | Datos, modelo LSTM, entrenamiento, evaluación y pronóstico |
| `app.py` | Servidor Flask con la API `POST /api/predict` |
| `templates/index.html` | Interfaz web con Chart.js |

## Aviso

Proyecto educativo. Con datos reales, el LSTM rara vez supera con claridad al baseline ingenuo: los mercados son en gran parte impredecibles. Nada de esto es asesoría financiera.
