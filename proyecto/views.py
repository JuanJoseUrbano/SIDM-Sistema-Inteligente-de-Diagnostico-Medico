# ══════════════════════════════════════════════════════════════════════════════
# views.py — SIDM: Sistema Inteligente de Diagnóstico Médico
# Modelo: BETO (dccuchile/bert-base-spanish-wwm-cased)
#
# Arquitectura:
#   Texto crudo
#     → Limpieza mínima (URLs, emails, espacios)
#     → Tokenizador WordPiece de BETO (maneja acentos, puntuación, morfología)
#     → TFBertForSequenceClassification (fine-tuning completo)
#     → Predicción de enfermedad
# ══════════════════════════════════════════════════════════════════════════════

# ── Django ────────────────────────────────────────────────────────────────────
from django.shortcuts import render

# ── Librerías base ────────────────────────────────────────────────────────────
import os
import re
import json
import pickle

import numpy as np
import pandas as pd

# ── TensorFlow ────────────────────────────────────────────────────────────────
import tensorflow as tf

# ── HuggingFace Transformers ──────────────────────────────────────────────────
from transformers import BertTokenizerFast, TFBertForSequenceClassification

# ── ML / métricas ─────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)


# ══════════════════════════════════════════════════════════════════════════════
# Constantes de configuración
# ══════════════════════════════════════════════════════════════════════════════

# Checkpoint oficial de BETO cased en HuggingFace Hub.
BETO_CHECKPOINT = "dccuchile/bert-base-spanish-wwm-cased"

# Longitud máxima de secuencia tras tokenización WordPiece.
MAX_SEQ_LEN = 128

# ── Rutas absolutas ancladas a este mismo archivo ─────────────────────────────
#
#   Estructura esperada del proyecto:
#
#   SIDM-main/
#   ├── manage.py
#   ├── models/                  ← MODEL_DIR  (se crea automáticamente)
#   │   ├── modelo_beto/
#   │   ├── tokenizer_beto/
#   │   ├── label_encoder.pkl
#   │   └── metrics.json
#   └── sidm/                    ← carpeta de la app Django (aquí vive views.py)
#       ├── views.py
#       └── dataset/
#           ├── ENFERMEDADES_SINTOMAS.csv
#           └── ENFERMEDADES_SINTOMAS_redux.csv
#
# Si tu estructura difiere, ajusta solo APP_DIR y MODEL_DIR.

# Directorio donde está este views.py  →  SIDM-main/sidm/
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorio raíz del proyecto  →  SIDM-main/
PROJECT_DIR = os.path.dirname(APP_DIR)

# Dataset dentro de la app
DATASET_DIR = os.path.join(APP_DIR, "dataset")
CSV_PATH       = os.path.join(DATASET_DIR, "ENFERMEDADES_SINTOMAS.csv")
CSV_REDUX_PATH = os.path.join(DATASET_DIR, "ENFERMEDADES_SINTOMAS_redux.csv")

# Modelos en la raíz del proyecto (fuera de la app para no mezclar con el código)
MODEL_DIR          = os.path.join(PROJECT_DIR, "models", "models")
MODEL_PATH         = os.path.join(MODEL_DIR, "modelo_beto")
TOKENIZER_PATH     = os.path.join(MODEL_DIR, "tokenizer_beto")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_PATH       = os.path.join(MODEL_DIR, "metrics.json")

# Crear carpeta models al arrancar (no falla si ya existe)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Estado global (singletons en memoria) ─────────────────────────────────────
_tokenizer_beto: BertTokenizerFast | None = None
_label_encoder:  LabelEncoder      | None = None


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — Limpieza de texto
# ══════════════════════════════════════════════════════════════════════════════

def limpiar_texto(texto: str) -> str:
    """
    Limpieza mínima y no destructiva, diseñada para BETO cased.

    Se conserva la mayor parte del texto original porque BETO:
      - Maneja acentos y ñ nativamente.
      - Usa WordPiece para descomponer palabras en subpalabras.
      - Aprende por sí solo qué tokens son relevantes mediante la atención.

    Solo se eliminan ruidos que BETO no puede aprovechar: URLs, correos
    y espacios redundantes.
    """
    if pd.isna(texto):
        return ""

    texto = str(texto)
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)   # URLs
    texto = re.sub(r"\S+@\S+\.\S+", "", texto)            # emails
    texto = re.sub(r"\s+", " ", texto).strip()             # espacios extra

    return texto


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — Tokenización BETO
# ══════════════════════════════════════════════════════════════════════════════

def get_tokenizer(from_disk: bool = False) -> BertTokenizerFast:
    """
    Devuelve el tokenizador de BETO (singleton).

    Parámetros
    ----------
    from_disk : si True y existe TOKENIZER_PATH, lo carga desde disco
                (evita descargar de HuggingFace en cada reinicio del servidor).
    """
    global _tokenizer_beto

    if _tokenizer_beto is None:
        if from_disk and os.path.isdir(TOKENIZER_PATH):
            _tokenizer_beto = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
        else:
            _tokenizer_beto = BertTokenizerFast.from_pretrained(BETO_CHECKPOINT)

    return _tokenizer_beto


def encode_texts(
    texts: "pd.Series | list",
    tokenizer: BertTokenizerFast,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Convierte textos limpios en tensores de entrada para BETO.

    Retorna
    -------
    input_ids      : (n, MAX_SEQ_LEN)
    attention_mask : (n, MAX_SEQ_LEN)
    """
    encoding = tokenizer(
        list(texts),
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return encoding["input_ids"], encoding["attention_mask"]


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — Preprocesamiento completo
# ══════════════════════════════════════════════════════════════════════════════

def preprocessing(df: pd.DataFrame):
    """
    DataFrame → conjuntos de entrenamiento y prueba listos para BETO.

    Pasos: limpieza → codificación de etiquetas → split 70/30 → tokenización.

    Retorna
    -------
    x_train_ids, x_train_masks,
    x_test_ids,  x_test_masks,
    y_train, y_test
    """
    global _label_encoder

    df = df.copy()
    df["SINTOMAS"] = df["SINTOMAS"].apply(limpiar_texto)

    X = df["SINTOMAS"]
    Y = df["ENFERMEDAD"]

    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    _label_encoder = label_encoder

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        X,
        Y_encoded,
        test_size=0.30,
        shuffle=True,
        stratify=Y_encoded,
        random_state=1,
    )

    # Usar tokenizador local si ya existe (más rápido en reruns)
    tokenizer = get_tokenizer(from_disk=os.path.isdir(TOKENIZER_PATH))

    x_train_ids, x_train_masks = encode_texts(x_train_raw, tokenizer)
    x_test_ids,  x_test_masks  = encode_texts(x_test_raw,  tokenizer)

    return (
        x_train_ids, x_train_masks,
        x_test_ids,  x_test_masks,
        y_train, y_test,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4 — Construcción del modelo
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int) -> TFBertForSequenceClassification:
    """
    Carga BETO preentrenado y agrega un cabezal de clasificación.

    El fine-tuning actualiza TODOS los pesos (BETO + clasificador).

    Parámetros
    ----------
    num_classes : número de enfermedades únicas en el dataset
    """
    model = TFBertForSequenceClassification.from_pretrained(
        BETO_CHECKPOINT,
        num_labels=num_classes,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 5 — Predicción con top-K y confianza
# ══════════════════════════════════════════════════════════════════════════════

def predecir(
    model: TFBertForSequenceClassification,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    top_k: int = 3,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Genera predicciones y probabilidades a partir de los logits del modelo.

    Retorna
    -------
    y_pred : (n,)   — índice de clase con mayor probabilidad
    y_prob : (n, C) — distribución de probabilidad sobre todas las clases
    """
    input_ids      = tf.cast(input_ids,      tf.int32)
    attention_mask = tf.cast(attention_mask, tf.int32)
    token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)

    outputs = model(
        {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        training=False,
    )

    y_prob = tf.nn.softmax(outputs.logits, axis=-1).numpy()
    y_pred = np.argmax(y_prob, axis=1)

    return y_pred, y_prob


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 5.5 — Guardar y cargar modelo
# ══════════════════════════════════════════════════════════════════════════════

def guardar_modelo(
    model: TFBertForSequenceClassification,
    tokenizer: BertTokenizerFast,
    label_encoder: LabelEncoder,
) -> None:
    """
    Persiste el modelo, tokenizador y label_encoder en MODEL_DIR.
    """
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"✓ Modelo guardado en:       {MODEL_PATH}")
    print(f"✓ Tokenizador guardado en:  {TOKENIZER_PATH}")
    print(f"✓ Label encoder guardado en:{LABEL_ENCODER_PATH}")


def _modelo_guardado_existe() -> bool:
    """
    Comprueba que los tres artefactos necesarios estén presentes en disco:
      - carpeta modelo_beto/ con al menos config.json dentro
      - label_encoder.pkl
    El tokenizador es opcional en disco (se puede recargar de HuggingFace).
    """
    config_json = os.path.join(MODEL_PATH, "config.json")
    existe = os.path.isdir(MODEL_PATH) and os.path.isfile(config_json) \
             and os.path.isfile(LABEL_ENCODER_PATH)

    # Log siempre visible en la consola del servidor para depuración
    print("── Verificación de modelo guardado ──────────────────────────")
    print(f"   MODEL_DIR          : {MODEL_DIR}")
    print(f"   MODEL_PATH         : {MODEL_PATH}  → isdir={os.path.isdir(MODEL_PATH)}")
    print(f"   config.json        : {config_json}  → exists={os.path.isfile(config_json)}")
    print(f"   LABEL_ENCODER_PATH : {LABEL_ENCODER_PATH}  → exists={os.path.isfile(LABEL_ENCODER_PATH)}")
    print(f"   METRICS_PATH       : {METRICS_PATH}  → exists={os.path.isfile(METRICS_PATH)}")
    print(f"   → modelo_existe    : {existe}")
    print("─────────────────────────────────────────────────────────────")

    return existe


def cargar_modelo() -> "tuple[TFBertForSequenceClassification, BertTokenizerFast, LabelEncoder] | None":
    """
    Carga el modelo, tokenizador y label_encoder desde MODEL_DIR.
    Retorna la tupla o None si algún archivo no existe.
    """
    global _label_encoder, _tokenizer_beto

    if not _modelo_guardado_existe():
        return None

    try:
        model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)

        # Preferir tokenizador local; si no existe, descargar de HuggingFace
        tok_source = TOKENIZER_PATH if os.path.isdir(TOKENIZER_PATH) else BETO_CHECKPOINT
        tokenizer  = BertTokenizerFast.from_pretrained(tok_source)

        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        # Sincronizar singletons globales con lo cargado desde disco
        _tokenizer_beto = tokenizer
        _label_encoder  = label_encoder

        return model, tokenizer, label_encoder

    except Exception as e:
        print(f"✗ Error al cargar modelo: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 6 — Vistas Django
# ══════════════════════════════════════════════════════════════════════════════

def main(request):
    """Vista principal — menú para elegir entre entrenar o diagnosticar."""
    modelo_existe  = _modelo_guardado_existe()
    metrics_existe = os.path.isfile(METRICS_PATH)

    context = {
        "modelo_existe":  modelo_existe,
        "metrics_existe": metrics_existe,
    }
    return render(request, "index.html", context=context)


def entrenar(request):
    """
    Vista para entrenar el modelo.

    Flujo:
      1. Carga y unión de los dos CSV del dataset.
      2. Preprocesamiento y tokenización con BETO.
      3. Construcción y fine-tuning del modelo.
      4. Evaluación con múltiples métricas.
      5. Guardado del modelo y métricas.
      6. Renderizado de resultados.
    """
    global _label_encoder

    # ── 1. Carga del dataset ──────────────────────────────────────────────
    # Validar que los archivos existan antes de intentar leerlos
    for path in (CSV_PATH, CSV_REDUX_PATH):
        if not os.path.exists(path):
            context = {"error": f"Dataset no encontrado: {path}"}
            return render(request, "index.html", context=context)

    csv     = pd.read_csv(CSV_PATH)
    csv_aux = pd.read_csv(CSV_REDUX_PATH)
    data    = pd.concat([csv, csv_aux], ignore_index=True)
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # ── 2. Preprocesamiento ───────────────────────────────────────────────
    (
        x_train_ids, x_train_masks,
        x_test_ids,  x_test_masks,
        y_train, y_test,
    ) = preprocessing(data)

    num_classes = len(np.unique(y_train))

    # ── 3. Construcción del modelo ────────────────────────────────────────
    model = build_model(num_classes)

    # ── 4. Entrenamiento (fine-tuning) ────────────────────────────────────
    history = model.fit(
        {"input_ids": x_train_ids, "attention_mask": x_train_masks},
        y_train,
        validation_split=0.2,
        batch_size=16,
        epochs=5,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    # ── 5. Evaluación ─────────────────────────────────────────────────────
    y_pred, y_prob = predecir(model, x_test_ids, x_test_masks)

    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    top3 = top_k_accuracy_score(y_test, y_prob, k=3, labels=np.arange(num_classes))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_test, y_pred)

    # ── 6. Nombres de clases ──────────────────────────────────────────────
    class_names = list(_label_encoder.classes_)

    # ── 7. Matriz de confusión etiquetada ─────────────────────────────────
    matriz_confusion_etiquetada = [
        {
            "real": class_names[i],
            "valores": [
                {
                    "pred":        class_names[j],
                    "valor":       int(cm[i, j]),
                    "es_diagonal": i == j,
                }
                for j in range(len(class_names))
            ],
        }
        for i in range(len(class_names))
    ]

    # ── 8. Primeras 50 predicciones detalladas ────────────────────────────
    prediccion_50 = []
    for i in range(min(50, len(y_pred))):
        real_idx = int(y_test[i])
        pred_idx = int(y_pred[i])

        top3_idx = np.argsort(y_prob[i])[::-1][:3]
        top3_detalle = [
            {
                "enfermedad":   class_names[idx],
                "probabilidad": float(y_prob[i][idx]),
            }
            for idx in top3_idx
        ]

        prediccion_50.append(
            {
                "id":        i + 1,
                "real":      class_names[real_idx],
                "predicho":  class_names[pred_idx],
                "confianza": float(np.max(y_prob[i])),
                "acierto":   "Sí" if real_idx == pred_idx else "No",
                "top3":      top3_detalle,
            }
        )

    # ── 9. Métricas consolidadas ──────────────────────────────────────────
    metricas = {
        "accuracy":             round(float(acc),                                       4),
        "balanced_accuracy":    round(float(bacc),                                      4),
        "precision_macro":      round(float(precision_macro),                           4),
        "recall_macro":         round(float(recall_macro),                              4),
        "f1_macro":             round(float(f1_macro),                                  4),
        "precision_weighted":   round(float(precision_w),                               4),
        "recall_weighted":      round(float(recall_w),                                  4),
        "f1_weighted":          round(float(f1_weighted),                               4),
        "top3_accuracy":        round(float(top3),                                      4),
        "loss_train_final":     round(float(history.history["loss"][-1]),               4),
        "accuracy_train_final": round(float(history.history["accuracy"][-1]),           4),
        "val_loss_final":       round(float(history.history["val_loss"][-1]),           4),
        "val_accuracy_final":   round(float(history.history["val_accuracy"][-1]),       4),
    }

    # ── 10. Guardar modelo y métricas ─────────────────────────────────────
    guardar_modelo(model, get_tokenizer(), _label_encoder)

    metrics_bundle = {
        "metricas":                    metricas,
        "matriz_confusion":            cm.tolist(),
        "matriz_confusion_etiquetada": matriz_confusion_etiquetada,
        "reporte_clasificacion":       report,
        "prediccion":                  prediccion_50,
        "clases":                      class_names,
        "timestamp":                   pd.Timestamp.now().isoformat(),
    }

    try:
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_bundle, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"✗ Error al guardar métricas: {e}")

    context = {
        "metricas":                    metricas,
        "matriz_confusion":            cm.tolist(),
        "matriz_confusion_etiquetada": matriz_confusion_etiquetada,
        "reporte_clasificacion":       report,
        "prediccion":                  prediccion_50,
        "clases":                      class_names,
        "tipo":                        "entrenar",
    }

    return render(request, "index.html", context=context)


def diagnosticar(request):
    """
    Vista para hacer diagnósticos usando el modelo entrenado.

    GET  → muestra el formulario de síntomas.
    POST → procesa los síntomas y muestra el diagnóstico top-3.
    """
    resultado = cargar_modelo()

    if resultado is None:
        context = {
            "error": "No hay modelo entrenado. Primero debe entrenar el modelo.",
            "tipo":  "diagnosticar",
        }
        return render(request, "resultado.html", context=context)

    model, tokenizer, label_encoder = resultado

    if request.method == "GET":
        return render(request, "resultado.html", {"tipo": "formulario"})

    # ── POST: procesar síntomas ───────────────────────────────────────────
    sintomas_usuario = request.POST.get("sintomas", "").strip()

    if not sintomas_usuario:
        context = {
            "error": "Por favor ingresa síntomas para diagnosticar.",
            "tipo":  "formulario",
        }
        return render(request, "resultado.html", context=context)

    sintomas_limpios = limpiar_texto(sintomas_usuario)
    input_ids, attention_mask = encode_texts([sintomas_limpios], tokenizer)

    y_pred, y_prob = predecir(model, input_ids, attention_mask)

    pred_idx = int(y_pred[0])
    probs    = y_prob[0]

    # Top-3 completo (incluye la predicción principal)
    top3_idx = np.argsort(probs)[::-1][:3]

    # Enfermedad principal
    enfermedad_principal    = label_encoder.classes_[pred_idx]
    probabilidad_principal  = float(probs[pred_idx]) * 100

    # Las otras dos alternativas del top-3 (excluyendo la principal)
    top_3_alternativas = [
        {
            "nombre":       label_encoder.classes_[idx],
            "probabilidad": float(probs[idx]) * 100,
        }
        for idx in top3_idx
        if idx != pred_idx          # no repetir la predicción principal
    ]

    context = {
        "tipo":                  "diagnostico",
        "sintomas_usuario":      sintomas_usuario,
        "enfermedad_principal":  enfermedad_principal,
        "probabilidad_principal": probabilidad_principal,
        "top_3":                 top_3_alternativas,   # lista de 0–2 alternativas
    }

    return render(request, "resultado.html", context=context)


def mostrar_metricas(request):
    """Vista para mostrar métricas guardadas sin reentrenar."""
    if not os.path.exists(METRICS_PATH):
        context = {"error": "No hay métricas guardadas. Entrena el modelo primero."}
        return render(request, "metricas.html", context=context)

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_bundle = json.load(f)
    except Exception as e:
        context = {"error": f"Error al leer métricas: {e}"}
        return render(request, "metricas.html", context=context)

    return render(request, "metricas.html", context=metrics_bundle)