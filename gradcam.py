"""
gradcam.py
----------
Grad-CAM heatmap generation, split into two paths since the custom CNN and
the transfer-learning models are structured differently:

- gradcam_custom_cnn: works directly on the custom_cnn functional model,
  targeting its last Conv2D layer.
- gradcam_transfer_model: targets the last conv layer INSIDE the wrapped
  backbone submodel (e.g. MobileNetV2/ResNet50), since for transfer models
  the conv layers live one level down inside `base_model`, not on the
  outer model directly.

overlay_heatmap blends a heatmap onto the original (uint8, RGB) image for
display.
"""

import numpy as np
import tensorflow as tf
import cv2


def _last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def gradcam_custom_cnn(model, img_batch):
    """
    img_batch: (1, H, W, 3) already preprocessed exactly as the model expects
    (normalized to [0,1] for the custom CNN).

    Returns a (H, W) heatmap normalized to [0, 1].
    """
    last_conv = _last_conv_layer_name(model)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def gradcam_transfer_model(model, base_model, preprocess_fn, img_batch):
    """
    img_batch: (1, H, W, 3) RAW [0,255] pixels (the model applies its own
    preprocess_fn internally via the Lambda layer, so we don't pre-apply it
    here — we only need it to locate/verify preprocessing if inspecting the
    backbone directly).

    Builds a grad model from the outer model's input straight through to the
    backbone's last conv layer AND the final prediction, so gradients flow
    through the same preprocessing Lambda the model was trained with.
    """
    last_conv = _last_conv_layer_name(base_model)

    # Rebuild a functional path: outer model input -> preprocess -> base_model
    # -> named conv layer output, reusing the already-trained layers.
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [base_model.get_layer(last_conv).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(display_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    display_img: (H, W, 3) uint8 RGB image
    heatmap: (h, w) float heatmap in [0, 1], smaller spatial size than display_img

    Returns a uint8 RGB overlay image the same size as display_img.
    """
    h, w = display_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (display_img.astype(np.float32) * (1 - alpha) +
               heatmap_color.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)
