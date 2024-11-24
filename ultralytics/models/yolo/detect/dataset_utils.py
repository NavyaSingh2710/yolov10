import cv2
import numpy as np
import random
import math


def random_perspective(img, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0):
    """
    Apply random perspective transformation to the image and its polygon-like bounding boxes.
    """
    height, width = img.shape[:2]

    # Center matrix
    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2

    # Perspective matrix
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # Rotation and scale
    R = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    scale_factor = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale_factor)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined transformation matrix
    M = T @ S @ R @ P @ C
    if perspective:
        img = cv2.warpPerspective(img, M, (width, height), borderValue=(114, 114, 114))
    else:
        img = cv2.warpAffine(img, M[:2], (width, height), borderValue=(114, 114, 114))

    # Transform polygon-like label coordinates
    if len(targets):
        n = len(targets)
        xy = np.ones((n, 3))
        xy[:, :2] = targets
        xy = xy @ M.T
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(-1, 2)
        else:
            xy = xy[:, :2].reshape(-1, 2)
        targets = xy

    return img, targets


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    Randomly adjust the hue, saturation, and value of the image.
    """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    dtype = img.dtype
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[..., 0] = (img_hsv[..., 0] * r[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * r[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * r[2], 0, 255)
    img_hsv = img_hsv.astype(dtype)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def cutout(image, targets, fraction=0.5):
    """
    Applies cutout augmentation by randomly covering parts of the image.
    """
    h, w = image.shape[:2]
    scales = [0.5, 0.25, 0.125, 0.0625]

    for scale in scales:
        mask_h = random.randint(1, int(h * scale))
        mask_w = random.randint(1, int(w * scale))

        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        image[ymin:ymax, xmin:xmax] = [random.randint(0, 255) for _ in range(3)]

    return image, targets


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize image with unchanged aspect ratio using padding.
    """
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = np.mod(dw, 32) // 2, np.mod(dh, 32) // 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh, dh
    left, right = dw, dw
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


def load_mosaic(images, targets, img_size):
    """
    Combines four images into a mosaic.
    """
    s = img_size
    yc, xc = [random.randint(s // 2, s * 3 // 2) for _ in range(2)]
    mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    mosaic_targets = []

    for i, (img, target) in enumerate(zip(images, targets)):
        h, w = img.shape[:2]
        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
        mosaic_img[y1a:y2a, x1a:x2a] = img[:y2a - y1a, :x2a - x1a]
        mosaic_targets.append(target)

    mosaic_targets = np.concatenate(mosaic_targets, 0)
    return mosaic_img, mosaic_targets


def mixup(img1, targets1, img2, targets2, alpha=0.5):
    """
    Mix two images and their labels with a weight.
    """
    mix_img = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    mix_targets = np.concatenate((targets1, targets2), axis=0)
    return mix_img, mix_targets
