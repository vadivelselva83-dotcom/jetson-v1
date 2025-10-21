# file: train_retinanet.py
import os, json, math, random
import tensorflow as tf
import tensorflow_addons as tfa
import keras_cv
from tensorflow import keras

# Paths
IMG_ROOT = "data/images"
TRAIN_JSON = "data/annotations/instances_train.json"
VAL_JSON   = "data/annotations/instances_val.json"
OUTPUT_DIR = "outputs_retinanet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1 class: "headlight" (class_id=1)
NUM_CLASSES = 1
IMG_SIZE = 640  # Downscale for speed on Jetson
BATCH = 8
EPOCHS = 30
BASE_LR = 5e-4

# --- Datasets (COCO) ---
train_ds = keras_cv.datasets.COCODataset(
    bounding_box_format="xywh",  # KerasCV internal; we'll convert to "xyxy" later
    split="train",
    annotation_path=TRAIN_JSON,
    img_path=IMG_ROOT,
)
val_ds = keras_cv.datasets.COCODataset(
    bounding_box_format="xywh",
    split="val",
    annotation_path=VAL_JSON,
    img_path=IMG_ROOT,
)

# Convert to "xyxy" and keep only class_id==1 (headlight)
def filter_and_convert(sample):
    img = sample["images"]
    boxes = sample["bounding_boxes"]["boxes"]  # [N,4] xywh
    classes = sample["bounding_boxes"]["classes"]  # [N]
    # keep only 1 (headlight). If your dataset has class 0 or different, adjust here.
    mask = tf.where(tf.equal(classes, 1))[:,0]
    boxes = tf.gather(boxes, mask, axis=0)
    classes = tf.gather(classes, mask, axis=0) - 1  # shift to 0..(NUM_CLASSES-1)

    # xywh -> xyxy
    x, y, w, h = tf.unstack(boxes, axis=-1)
    boxes_xyxy = tf.stack([x, y, x+w, y+h], axis=-1)

    return {
        "images": img,
        "bounding_boxes": {
            "boxes": boxes_xyxy,
            "classes": classes
        }
    }

train_ds = train_ds.map(filter_and_convert, num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = val_ds.map(filter_and_convert,   num_parallel_calls=tf.data.AUTOTUNE)

# Augmentations tuned for night scenes
augment = keras.Sequential([
    keras.layers.Resizing(IMG_SIZE, IMG_SIZE, crop_to_aspect_ratio=True),
    keras_cv.layers.RandomFlip(mode="horizontal"),
    # gamma/brightness/contrast simulate glare and dark frames
    keras.layers.Lambda(lambda x: tf.image.adjust_gamma(x, gamma=tf.random.uniform([], 0.6, 1.4))),
    keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
    keras.layers.Lambda(lambda x: tf.image.random_contrast(x, 0.6, 1.4)),
])

def preprocess(sample):
    img = tf.image.convert_image_dtype(sample["images"], tf.float32)
    img = augment(img)
    # KerasCV expects dict with image + bbox in "xyxy"
    return {"images": img, "bounding_boxes": sample["bounding_boxes"]}

train_ds = train_ds.shuffle(1024).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(
    BATCH, padding_values={"images": 0.0, "bounding_boxes": {"boxes": 0.0, "classes": -1}}
).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(
    lambda s: {"images": tf.image.convert_image_dtype(
        keras.layers.Resizing(IMG_SIZE, IMG_SIZE, crop_to_aspect_ratio=True)(s["images"]), tf.float32),
        "bounding_boxes": s["bounding_boxes"]
    }, num_parallel_calls=tf.data.AUTOTUNE
).padded_batch(
    BATCH, padding_values={"images": 0.0, "bounding_boxes": {"boxes": 0.0, "classes": -1}}
).prefetch(tf.data.AUTOTUNE)

# Model: RetinaNet (resnet50 backbone)
model = keras_cv.models.RetinaNet(
    classes=NUM_CLASSES,
    bounding_box_format="xyxy",
    backbone="resnet50",
    include_rescaling=False,
)

# Optimizer, losses, metrics
steps_per_epoch = 100 if len(list(iter(train_ds))) == 0 else None
optimizer = tfa.optimizers.AdamW(learning_rate=BASE_LR, weight_decay=1e-4)
model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=optimizer,
    metrics=[keras_cv.metrics.BoxCOCOMetrics(
        bounding_box_format="xyxy",
        evaluate_freq=1,
        name="coco_metrics"
    )],
)

ckpt = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTPUT_DIR, "ckpt"),
    save_weights_only=True, save_best_only=True, monitor="val_coco_metrics_AP", mode="max"
)
tb = keras.callbacks.TensorBoard(log_dir=os.path.join(OUTPUT_DIR, "logs"))

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[ckpt, tb])

# Save SavedModel
export_path = os.path.join(OUTPUT_DIR, "savedmodel")
model.save(export_path)
print("Saved:", export_path)
