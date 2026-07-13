"""
data.py
-------
Loads the cropped helmet/no_helmet classification dataset into
train/validation/test tf.data pipelines, and builds a mild,
domain-appropriate augmentation layer.
"""

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(cropped_dir, img_size=(160, 160), batch_size=32, seed=42,
                   val_split=0.15, test_split=0.15):
    """
    Load images from cropped_dir/<class_name>/*.jpg using
    image_dataset_from_directory, and split into train/val/test.

    Class labels are inferred alphabetically from subfolder names
    (so "helmet" -> 0, "no_helmet" -> 1) and returned as class_names
    in that index order.

    Returns: train_ds, val_ds, test_ds, class_names
    """
    # First split off a combined val+test portion, then split that in half.
    holdout_frac = val_split + test_split

    train_ds = tf.keras.utils.image_dataset_from_directory(
        cropped_dir, validation_split=holdout_frac, subset="training",
        seed=seed, image_size=img_size, batch_size=batch_size, label_mode="int",
    )
    holdout_ds = tf.keras.utils.image_dataset_from_directory(
        cropped_dir, validation_split=holdout_frac, subset="validation",
        seed=seed, image_size=img_size, batch_size=batch_size, label_mode="int",
    )

    class_names = train_ds.class_names

    # holdout_ds batches -> split roughly in half into val_ds / test_ds
    holdout_batches = tf.data.experimental.cardinality(holdout_ds).numpy()
    val_batches = max(1, holdout_batches // 2)

    val_ds = holdout_ds.take(val_batches)
    test_ds = holdout_ds.skip(val_batches)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_augmentation():
    """
    Mild, domain-appropriate augmentation:
    - horizontal flip only (NOT vertical — an upside-down helmet photo isn't realistic)
    - small rotation to mimic camera angle variation
    - modest brightness/contrast jitter to mimic different site lighting
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")


def prepare(ds, augment_layer=None, normalize=False, shuffle=False, shuffle_buffer=1000):
    """
    Apply (optionally) shuffling, augmentation, and normalization to a
    tf.data.Dataset of (image_batch, label_batch) pairs.

    normalize=True  -> scale pixels to [0, 1] (for the custom CNN, which has
                        no built-in preprocessing layer)
    normalize=False -> leave pixels in [0, 255] (for transfer-learning models,
                        which apply their own backbone-specific preprocessing
                        inside models.py)
    """
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    if augment_layer is not None:
        ds = ds.map(lambda x, y: (augment_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)

    if normalize:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)
