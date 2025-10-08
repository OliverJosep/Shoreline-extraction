import numpy as np

def crop(img, position1, position2):
    return img[position1[0]:position2[0], position1[1]:position2[1]]

def merge_masks(img1, img2, position1, position2):
    merged = img1.copy()
    merged[position1[0]:position2[0], position1[1]:position2[1]] = img2
    return merged

def merge_image_with_mask(image, mask, alpha=0.5):
    COLOR_CLASS_0 = [165, 90, 0]
    COLOR_CLASS_1 = [0, 0, 200]

    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    colored_mask[mask == 0] = COLOR_CLASS_0
    colored_mask[mask == 1] = COLOR_CLASS_1

    blended = (alpha * image + (1 - alpha) * colored_mask).astype(np.uint8)

    return blended

def apply_masks(image, predicted_mask, shoreline_pixel_predicted_mask):
    # Copy original image
    overlay = image.copy()

    alpha = 1

    red = np.array([255, 0, 0], dtype=np.uint8)       # Predicted shoreline

    # Create masks for shoreline pixels
    only_predicted = predicted_mask == shoreline_pixel_predicted_mask

    overlay[only_predicted] = (
        alpha * red + (1 - alpha) * overlay[only_predicted]
    ).astype(np.uint8)

    return overlay