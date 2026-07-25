# Generación de Música ABC con LSTM y Transformer

Proyecto final de la materia **Deep Learning & NLP** de la Universidad San Francisco de Quito.

Este proyecto compara dos arquitecturas neuronales autoregresivas para la generación de música simbólica en notación ABC:

- Una red recurrente **LSTM** como línea base.
- Un **Transformer decoder-only** con self-attention causal implementado en PyTorch.

Ambos modelos son entrenados a nivel de carácter sobre una colección de melodías para piano en notación ABC. La comparación considera pérdida, perplejidad, tiempo de entrenamiento, número de parámetros y validez sintáctica de las composiciones generadas.

## Objetivos

El proyecto busca:

1. Explorar y limpiar un conjunto de melodías en notación ABC.
2. Construir un vocabulario a nivel de carácter.
3. Implementar un `Dataset` de PyTorch mediante ventanas deslizantes.
4. Entrenar una LSTM y un Transformer causal bajo condiciones comparables.
5. Generar nuevas secuencias musicales mediante muestreo por temperatura y `top-k`.
6. Validar las composiciones con `music21`.
7. Convertir las generaciones de ABC a MIDI y WAV.
8. Comparar cuantitativa y cualitativamente ambas arquitecturas.

## Conjunto de datos

Se utiliza el conjunto de datos **Piano Musics ABC Notation**, publicado en Kaggle por Uday2902:

[Piano Musics ABC Notation — Kaggle](https://www.kaggle.com/datasets/uday2902/piano-musics-abc-notation)

El archivo principal es:

```text
piano-musics-abc-notation.txt
```

### Estadísticas

| Característica | Valor |
|---|---:|
| Melodías originales | 4.443 |
| Fragmentos HTML eliminados | 6 |
| Melodías después de la limpieza | 4.437 |
| Caracteres después de la limpieza | 1.049.157 |
| Tamaño del vocabulario | 83 símbolos |
| Longitud mínima | 48 caracteres |
| Longitud máxima | 2.081 caracteres |
| Longitud media | 256 caracteres |
| Longitud mediana | 214 caracteres |

El conjunto de datos original no contiene cabeceras ABC estandarizadas como `X:`, `T:`, `M:`, `L:` o `K:`. Por este motivo, no se añaden cabeceras durante el entrenamiento.

Para validar y convertir las composiciones generadas a audio se utiliza la siguiente cabecera sintética:

```text
X:1
T:Generated
M:4/4
L:1/8
K:C
```

## Preprocesamiento

El notebook realiza los siguientes pasos:

1. Lectura del archivo de texto.
2. Separación de melodías mediante líneas en blanco.
3. Identificación de fragmentos HTML.
4. Eliminación de seis segmentos contaminados.
5. Creación de un vocabulario de caracteres.
6. Construcción de los mapeos `char2idx` e `idx2char`.
7. Codificación de los caracteres como índices enteros.
8. División secuencial en 90 % para entrenamiento y 10 % para validación.
9. Creación de ventanas deslizantes para predecir el siguiente carácter.

### Configuración de las secuencias

```python
CONTEXT_WINDOW = 100
BATCH_SIZE = 128
STRIDE = 1
```

Cada ejemplo de entrenamiento contiene:

- **Entrada:** 100 caracteres consecutivos.
- **Objetivo:** los mismos caracteres desplazados una posición hacia adelante.

## Modelos

### LSTM baseline

La arquitectura recurrente contiene:

- Embedding de 128 dimensiones.
- Dos capas LSTM.
- 256 unidades ocultas.
- Dropout de 0,2.
- Capa lineal de salida sobre los 83 símbolos del vocabulario.

**Parámetros entrenables:** 953.555.

Configuración de entrenamiento:

```text
Épocas: 40
Learning rate: 3e-3
Warmup: no
Optimizador: AdamW
Weight decay: 0,01
Gradient clipping: 5,0
```

### Transformer causal

El Transformer decoder-only fue implementado sin utilizar `nn.MultiheadAttention` ni `nn.Transformer`.

La arquitectura contiene:

- Embeddings de caracteres de 128 dimensiones.
- Embeddings posicionales aprendidos.
- Cuatro bloques Transformer.
- Cuatro cabezas de atención.
- Dimensión por cabeza de 32.
- Red feed-forward de 512 dimensiones.
- Activación GELU.
- Pre-layer normalization.
- Conexiones residuales.
- Dropout de 0,1.
- Máscara causal triangular.

**Parámetros entrenables:** 827.475.

Configuración de entrenamiento:

```text
Épocas: 50
Learning rate: 3e-4
Warmup: 3 épocas
Optimizador: AdamW
Weight decay: 0,01
Gradient clipping: 5,0
```

## Función de entrenamiento

La función común de entrenamiento incluye:

- Entropía cruzada como función de pérdida.
- Optimizador AdamW.
- Scheduler coseno del learning rate.
- Warmup opcional.
- Gradient clipping.
- Evaluación sobre el conjunto de validación.
- Cálculo de perplejidad.
- Registro del tiempo por época.
- Checkpoint de la última época.
- Checkpoint del modelo con menor pérdida de validación.
- Capacidad para reanudar el entrenamiento.

Los checkpoints se almacenan en:

```text
checkpoints/
├── LSTM_last.pt
├── LSTM_best.pt
├── Transformer_last.pt
└── Transformer_best.pt
```

## Resultados

Los siguientes valores corresponden a la ejecución realizada sobre una GPU NVIDIA H200:

| Métrica | LSTM | Transformer |
|---|---:|---:|
| Parámetros | 953.555 | 827.475 |
| Épocas | 40 | 50 |
| Train loss | 0,4074 | 0,6346 |
| Validation loss | **0,3354** | 0,5508 |
| Perplejidad | **1,40** | 1,73 |
| Tiempo por época | 226,5 s | **55,4 s** |
| Tiempo total aproximado | 151,0 min | **46,2 min** |
| Validez ABC | 99 % | 100 % |

La LSTM obtuvo menor pérdida y perplejidad de validación. El Transformer presentó un tiempo por época aproximadamente cuatro veces menor en el entorno evaluado.

La comparación de velocidad debe interpretarse con precaución, porque cuDNN fue deshabilitado para la LSTM debido a un conflicto de versiones en el servidor.

## Generación de música

La generación se realiza de forma autoregresiva. En cada iteración, el modelo predice la distribución del siguiente carácter y selecciona un símbolo mediante muestreo estocástico.

La función admite:

- Texto inicial o `seed`.
- Longitud de generación.
- Temperatura.
- Muestreo `top-k`.
- Ventana máxima de contexto.

Configuraciones utilizadas para las muestras cualitativas:

```text
LSTM:
temperature = 0.8
top_k = 10

Transformer:
temperature = 1.0
top_k = 20
```

Una temperatura menor produce secuencias más conservadoras. Una temperatura mayor incrementa la diversidad, pero también puede aumentar el riesgo de errores o repeticiones inestables.

## Validación y conversión a audio

Las muestras generadas se validan con `music21` después de añadir la cabecera sintética.

La conversión sigue este flujo:

```text
ABC → MIDI → WAV
```

Herramientas utilizadas:

- `music21`: validación y análisis de la notación ABC.
- `abc2midi`: conversión de ABC a MIDI.
- `FluidSynth`: síntesis de MIDI a WAV.
- `FluidR3_GM.sf2`: banco de sonidos utilizado para la reproducción.

La validación sintáctica indica que la composición puede ser interpretada por el parser. No garantiza por sí sola que la melodía sea musicalmente agradable, novedosa o coherente.

## Análisis cualitativo

El notebook calcula para cada muestra:

- Número total de caracteres.
- Cantidad de notas generadas.
- Número de notas distintas.
- Símbolos de acordes.
- Ligaduras y ornamentaciones.
- Patrones de ritmo cortado `<` y `>`.

En las muestras analizadas:

| Característica | LSTM | Transformer |
|---|---:|---:|
| Caracteres | 301 | 301 |
| Notas generadas | 144 | 181 |
| Notas distintas | 13 | 13 |
| Ornamentos y ligaduras | 13 | 8 |
| Patrones de ritmo cortado | 7 | 3 |

Estos valores corresponden a muestras individuales y deben interpretarse como un análisis ilustrativo, no como una evaluación estadística completa.

## Requisitos

### Python

Se recomienda Python 3.10 o superior.

### Dependencias de Python

```text
torch
numpy
pandas
matplotlib
music21
tqdm
jupyter
ipython
kagglehub
```

Instalación:

```bash
pip install torch numpy pandas matplotlib music21 tqdm jupyter ipython kagglehub
```

### Dependencias del sistema

En Ubuntu o Google Colab:

```bash
sudo apt-get update
sudo apt-get install -y abcmidi fluidsynth fluid-soundfont-gm
```

## Descarga del conjunto de datos

El conjunto de datos puede descargarse manualmente desde Kaggle o mediante `kagglehub`:

```python
import kagglehub

path = kagglehub.dataset_download(
    "uday2902/piano-musics-abc-notation"
)

print("Dataset descargado en:", path)
```

Después de descargarlo, el archivo debe quedar en:

```text
data/piano-musics-abc-notation.txt
```

## Ejecución

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

En sistemas Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y abcmidi fluidsynth fluid-soundfont-gm
```

### 3. Añadir el conjunto de datos

```text
data/
└── piano-musics-abc-notation.txt
```

### 4. Configurar el directorio del proyecto

El notebook contiene una ruta utilizada en el servidor original:

```python
UNI_PATH = "/home/mjlopezt/DeppLearning/ProyectoFinal"
```

Debe reemplazarse por la ubicación correspondiente al entorno donde se ejecute el proyecto. Para utilizar el directorio actual:

```python
BASE_DIR = "."
```

### 5. Configurar la GPU

En el servidor DGX se utilizó:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

Este valor debe modificarse según la GPU asignada. En Colab o en un equipo con una sola GPU, esta línea puede eliminarse.

### 6. Ejecutar el notebook

```bash
jupyter notebook "Proyecto_Deep_Learning(1).ipynb"
```

También puede abrirse y ejecutarse desde Google Colab.

Las celdas deben ejecutarse en orden, ya que las arquitecturas, vocabulario, loaders y modelos entrenados se reutilizan en las secciones posteriores.

## Nota sobre cuDNN

Durante la ejecución en el servidor compartido se detectó un conflicto entre:

```text
cuDNN del sistema: 9.2.0
nvidia-cudnn-cu12 instalado por pip: 9.1.0.70
```

El conflicto provocaba el error:

```text
CUDNN_STATUS_NOT_INITIALIZED
```

Como solución temporal se utilizó:

```python
torch.backends.cudnn.enabled = False
```

Este cambio afecta principalmente el rendimiento de la LSTM. No debe aplicarse automáticamente en otros entornos si cuDNN funciona correctamente.

## Estructura recomendada del repositorio

```text
.
├── Proyecto_Deep_Learning(1).ipynb
├── README.md
├── requirements.txt
├── paper/
│   └── informe_final.pdf
├── data/
│   └── piano-musics-abc-notation.txt
├── checkpoints/
│   ├── LSTM_best.pt
│   ├── LSTM_last.pt
│   ├── Transformer_best.pt
│   └── Transformer_last.pt
├── figures/
│   ├── lstm_training_curves.png
│   ├── transformer_training_curves.png
│   └── comparacion_lstm_transformer.png
├── generations/
│   ├── lstm_generado.abc
│   └── transformer_generado.abc
├── audio/
│   ├── lstm_generado.abc
│   ├── lstm_generado.mid
│   ├── lstm_generado.wav
│   ├── transformer_generado.abc
│   ├── transformer_generado.mid
│   └── transformer_generado.wav
└── comparacion_modelos.csv
```

Los checkpoints y archivos WAV pueden ser demasiado grandes para un repositorio de GitHub convencional. En ese caso, pueden excluirse mediante `.gitignore` o almacenarse con Git LFS.

## Limitaciones

- El contexto está limitado a 100 caracteres.
- El conjunto de datos no contiene tonalidad, compás ni otros metadatos originales.
- La cabecera sintética permite analizar las muestras, pero no recupera los metadatos musicales reales.
- La evaluación con `music21` mide parseabilidad, no calidad musical.
- El análisis auditivo no incluye una evaluación humana formal.
- Los resultados corresponden a una ejecución por arquitectura.
- La desactivación de cuDNN afecta la comparación de velocidad de la LSTM.
- La división se realiza secuencialmente sobre los caracteres codificados y no por melodías completas.

## Trabajo futuro

- Dividir entrenamiento y validación por melodías completas.
- Comparar múltiples semillas de entrenamiento.
- Probar ventanas de contexto mayores.
- Utilizar tokenización por eventos musicales o BPE.
- Incorporar embeddings posicionales relativos.
- Entrenar con metadatos ABC reales.
- Evaluar diversidad, repetición y coherencia tonal.
- Realizar una evaluación con músicos u oyentes.
- Repetir el benchmark con cuDNN habilitado para la LSTM.

## Referencias principales

- Hochreiter, S. y Schmidhuber, J. “Long Short-Term Memory”, 1997.
- Vaswani, A. et al. “Attention Is All You Need”, 2017.
- Huang, C.-Z. A. et al. “Music Transformer: Generating Music with Long-Term Structure”, 2019.
- Sturm, B. L. et al. “Music Transcription Modelling and Composition Using Deep Learning”, 2016.
- Cuthbert, M. S. y Ariza, C. “music21: A Toolkit for Computer-Aided Musicology and Symbolic Music Data”, 2010.
- Uday2902. “Piano Musics ABC Notation”. Kaggle.

## Autora

**María José López T.**  
Universidad San Francisco de Quito  
Deep Learning & NLP
