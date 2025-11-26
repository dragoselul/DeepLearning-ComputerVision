import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image


def get_average_image_size(image_files, images_dir):
    widths = []
    heights = []

    if not image_files:
        print(f"No images found in {images_dir}")
        return 0, 0
    print(f"Processing images: {len(image_files)}")
    for img_file in tqdm(image_files):
        img_path = os.path.join(images_dir, img_file)
        try:
            with Image.open(img_path) as img:
                # PIL.Image.size returns (width, height)
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception as e:
            print(f"Skipping {img_file}: {e}")

    # Calculate averages
    avg_w = np.mean(widths)
    avg_h = np.mean(heights)

    print(f"\nTotal Images: {len(widths)}")
    print(f"Average Size: {avg_w:.2f} x {avg_h:.2f} (W x H)")

    return avg_w, avg_h


def resize_and_pad(image, boxes, target_size=(512, 512)):
    """
    Resizes image/boxes to target_size with black padding (letterboxing).
    Aggressively casts to int to avoid 'numpy.float64' errors.
    """
    target_w, target_h = map(int, target_size)  # Ensure target is int
    h, w = image.shape[:2]

    # 1. Calculate Scale
    scale = min(target_w / w, target_h / h)

    # 2. Resize Image (Force dimensions to int)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    # 3. Create Canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Calculate Padding (Force to int)
    pad_x = int((target_w - new_w) / 2)
    pad_y = int((target_h - new_h) / 2)

    # Paste with explicit int indices
    canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = resized

    # 4. Resize Boxes
    if len(boxes) > 0:
        boxes = np.array(boxes, dtype=np.float32)
        # Scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y

        # Optional: Clip boxes to image bounds to avoid going off-canvas
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, target_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, target_h)

    return canvas, boxes


def save_xml(sample_id, boxes, shape, out_dir):
    """Writes Pascal VOC XML for the resized image."""
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = f"{sample_id}.png"

    size = ET.SubElement(root, "size")
    for k, v in zip(['height', 'width', 'depth'], shape):
        ET.SubElement(size, k).text = str(v)

    for box in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "pothole"
        bndbox = ET.SubElement(obj, "bndbox")
        coords = ['xmin', 'ymin', 'xmax', 'ymax']
        for k, v in zip(coords, box):
            ET.SubElement(bndbox, k).text = str(int(v))

    ET.ElementTree(root).write(os.path.join(out_dir, f"{sample_id}.xml"))


if __name__ == "__main__":
    BASE_DIR = os.path.join(os.path.dirname(__file__), "../../", "potholes")
    OUT_DIR = os.path.join(os.path.dirname(__file__), "../../", "potholes_processed")
    IMAGE_DIR = os.path.join(BASE_DIR, "images")
    image_files = [f for f in os.listdir(IMAGE_DIR)]
    target_size = get_average_image_size(image_files, IMAGE_DIR)


    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'annotations'), exist_ok=True)

    # Get valid files
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory not found: {IMAGE_DIR}")
        exit(1)

    print(f"Found {len(image_files)} images. Resizing to {target_size}...")

    # --- Main Loop with TQDM ---
    for fname in tqdm(image_files, desc="Processing"):
        name = os.path.splitext(fname)[0]
        img_path = os.path.join(IMAGE_DIR, fname)
        xml_path = os.path.join(BASE_DIR, "annotations", name + ".xml")

        try:
            # 1. Read Image
            img = cv2.imread(img_path)
            if img is None: continue

            # 2. Read Annotations (if exist)
            boxes = []
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                for obj in tree.findall("object/bndbox"):
                    boxes.append([
                        float(obj.find(c).text) for c in ['xmin', 'ymin', 'xmax', 'ymax']
                    ])

            # 3. Resize & Pad
            new_img, new_boxes = resize_and_pad(img, boxes, target_size)

            # 4. Save Results
            cv2.imwrite(os.path.join(OUT_DIR, "images", f"{name}.png"), new_img)
            if len(new_boxes) > 0:
                save_xml(name, new_boxes, new_img.shape, os.path.join(OUT_DIR, "annotations"))

        except Exception as e:
            print(f"Failed on {fname}: {e}")

    print(f"Done! Data saved to: {OUT_DIR}")
