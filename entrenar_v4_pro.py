import pickle
import numpy as np
import tensorflow as tf
import os

print("--- 🧠 ENTRENAMIENTO NIVEL EXPERTO (V4) ---")

# 1. CARGAR DATOS
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict['data'], dict['labels']

path = 'cifar-10-batches-py/data_batch_1'
if not os.path.exists(path):
    print(f"❌ ERROR: No encuentro '{path}'.")
    exit()

print("Cargando y procesando dataset...")
X_raw, Y_raw = load_cifar_batch(path)
X_train = X_raw.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
X_train = X_train / 255.0  # Normalizar
Y_train = np.array(Y_raw)

# 2. DEFINIR MODELO PROFESIONAL (Con Aumento de Datos + Regularización)
model = tf.keras.models.Sequential([
    # CAPA DE AUMENTO DE DATOS (Solo activa durante entrenamiento)
    # Esto simula tener más fotos girándolas y haciendo zoom
    tf.keras.layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),

    # BLOQUE 1: Extracción de características finas
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(), # Normaliza para aprender más rápido
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2), # Apaga neuronas al azar para evitar memorización

    # BLOQUE 2: Formas medias
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    # BLOQUE 3: Conceptos complejos
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.4),

    # CEREBRO FINAL (Clasificación)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax') # 10 Clases
])

# Usamos una tasa de aprendizaje (learning rate) pequeña para ser precisos
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nIniciando entrenamiento ROBUSTO... (Paciencia, vale la pena)")
# Entrenamos por 20 épocas para que aproveche el aumento de datos
history = model.fit(X_train, Y_train, epochs=20, batch_size=64)

model.save('modelo_v4_expert.keras')
print("\n✅ ¡LISTO! Cerebro 'modelo_v4_expert.keras' guardado.")