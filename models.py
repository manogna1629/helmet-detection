"""
models.py
---------
Model builders for the helmet / no_helmet binary classifier:
- build_custom_cnn: a small CNN trained from scratch
- build_transfer_model: MobileNetV2 / ResNet50 / EfficientNetB0 / DenseNet121
  with an ImageNet-pretrained, frozen (initially) backbone and a new
  classification head
- compile_model: shared compile step (binary cross-entropy + Adam)

_PREPROCESS maps each supported backbone name to its required
Keras preprocessing function, so the notebook and gradcam.py can look it up
by name instead of hard-coding it.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

_BACKBONES = {
    "mobilenetv2": (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "resnet50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
    "densenet121": (tf.keras.applications.DenseNet121, tf.keras.applications.densenet.preprocess_input),
}

_PREPROCESS = {name: fn for name, (_, fn) in _BACKBONES.items()}


def build_custom_cnn(input_shape=(160, 160, 3), dropout=0.4):
    """
    Small CNN baseline, trained from scratch. Expects normalized [0,1] input.
    Binary output (sigmoid) for helmet (0) vs no_helmet (1).
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs, name="custom_cnn")


def build_transfer_model(backbone_name, input_shape=(160, 160, 3), weights="imagenet"):
    """
    Build a transfer-learning model: ImageNet-pretrained backbone (frozen)
    + new classification head. Expects RAW [0,255] input — preprocessing is
    applied inside the model via the backbone-specific preprocess_input fn.

    Returns (model, base_model) — base_model is returned separately so the
    notebook can unfreeze/fine-tune its top layers later.
    """
    if backbone_name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone_name}'. Options: {list(_BACKBONES)}")

    backbone_cls, preprocess_fn = _BACKBONES[backbone_name]

    base_model = backbone_cls(input_shape=input_shape, include_top=False, weights=weights)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(preprocess_fn, name="preprocess")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name=backbone_name)
    return model, base_model


def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
