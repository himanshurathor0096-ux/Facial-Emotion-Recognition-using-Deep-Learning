"""Train a small CNN on an appropriately licensed image dataset.

Expected layout:
data/fer_train/angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/
Optional validation folder: data/fer_validation/ with the same class folders.
The ZIP intentionally does not bundle facial image data or a trained model.
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, models

TRAIN_DIR = "data/fer_train"
VAL_DIR = "data/fer_validation"
OUT = "models/emotion_model.keras"
if not os.path.isdir(TRAIN_DIR):
    raise SystemExit("Training folder missing. Read data/README.md and README.md first.")

common = dict(labels="inferred", label_mode="int", color_mode="grayscale",
              image_size=(48, 48), batch_size=64)
if os.path.isdir(VAL_DIR) and any(os.scandir(VAL_DIR)):
    train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, shuffle=True, seed=42, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(VAL_DIR, shuffle=False, **common)
else:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, shuffle=True, seed=42, validation_split=0.2, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, shuffle=False, seed=42, validation_split=0.2, subset="validation", **common)

class_names = train_ds.class_names
print("Classes:", class_names)
if class_names != ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
    print("WARNING: app.py CLASS_NAMES must be updated to match this exact order:", class_names)

def scale(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

train_ds = train_ds.map(scale).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(scale).prefetch(tf.data.AUTOTUNE)

model = models.Sequential([
    layers.Input((48, 48, 1)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(128, activation="relu"),
    layers.Dropout(0.4), layers.Dense(len(class_names), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
os.makedirs("models", exist_ok=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(OUT, monitor="val_loss", save_best_only=True)
]
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Measured validation loss={loss:.4f}; validation accuracy={acc:.4f}")
with open("models/class_names.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(class_names) + "\n")
