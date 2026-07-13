"""
voc_crop.py
-----------
Converts a Pascal-VOC object-detection dataset (images + XML annotation files,
one XML per image, one <object> per labeled box) into a classification-ready
dataset by cropping each labeled box into its own image file.

Used for the Construction Safety Helmet Detection project to turn the
Kaggle "hard-hat-detection" object-detection dataset into a helmet /
no_helmet binary classification dataset.
"""

import os
import xml.etree.ElementTree as ET
from collections import Counter

from PIL import Image


def _iter_annotations(annotations_dir):
    """Yield (xml_path, filename_without_ext) for every .xml file in annotations_dir."""
    for fname in sorted(os.listdir(annotations_dir)):
        if fname.lower().endswith(".xml"):
            yield os.path.join(annotations_dir, fname), os.path.splitext(fname)[0]


def discover_classes(annotations_dir):
    """
    Scan every XML annotation file and count how many bounding boxes exist
    for each raw class name. Returns a collections.Counter.

    This is meant to be run BEFORE deciding on a CLASS_MAP, since raw class
    names/casing can vary between dataset versions.
    """
    counts = Counter()
    for xml_path, _ in _iter_annotations(annotations_dir):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is not None and name_el.text:
                counts[name_el.text.strip()] += 1
    return counts


def _find_image_file(images_dir, base_name):
    """Try common image extensions for a given annotation base filename."""
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        candidate = os.path.join(images_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def crop_dataset(annotations_dir, images_dir, output_dir, class_map,
                  min_box_size=20, padding_frac=0.10):
    """
    Read every XML annotation, map each object's raw class name through
    class_map (raw_name -> output_label or None to skip), and save a padded
    crop of each surviving box into output_dir/<output_label>/.

    Parameters
    ----------
    annotations_dir : str        directory of Pascal VOC .xml files
    images_dir : str             directory of the corresponding images
    output_dir : str             root directory to write cropped images into
    class_map : dict[str, str|None]
    min_box_size : int           skip boxes smaller than this (pixels, either side)
    padding_frac : float         fraction of box size added as padding on each side

    Returns
    -------
    dict summary: counts of crops written per output label, skipped boxes,
    and images that had no matching image file.
    """
    written = Counter()
    skipped_small = 0
    skipped_unmapped = 0
    missing_images = 0

    for xml_path, base_name in _iter_annotations(annotations_dir):
        img_path = _find_image_file(images_dir, base_name)
        if img_path is None:
            missing_images += 1
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            img_w, img_h = im.size

            box_idx = 0
            for obj in root.findall("object"):
                name_el = obj.find("name")
                raw_name = name_el.text.strip() if name_el is not None and name_el.text else None
                label = class_map.get(raw_name)
                if label is None:
                    skipped_unmapped += 1
                    continue

                bnd = obj.find("bndbox")
                if bnd is None:
                    continue
                xmin = int(float(bnd.find("xmin").text))
                ymin = int(float(bnd.find("ymin").text))
                xmax = int(float(bnd.find("xmax").text))
                ymax = int(float(bnd.find("ymax").text))

                box_w = xmax - xmin
                box_h = ymax - ymin
                if box_w < min_box_size or box_h < min_box_size:
                    skipped_small += 1
                    continue

                pad_x = int(box_w * padding_frac)
                pad_y = int(box_h * padding_frac)
                crop_box = (
                    max(0, xmin - pad_x),
                    max(0, ymin - pad_y),
                    min(img_w, xmax + pad_x),
                    min(img_h, ymax + pad_y),
                )

                crop = im.crop(crop_box)

                label_dir = os.path.join(output_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                out_name = f"{base_name}_{box_idx}.jpg"
                crop.save(os.path.join(label_dir, out_name), quality=95)

                written[label] += 1
                box_idx += 1

    return {
        "written_per_class": dict(written),
        "total_written": sum(written.values()),
        "skipped_small_boxes": skipped_small,
        "skipped_unmapped_boxes": skipped_unmapped,
        "images_missing_on_disk": missing_images,
    }
