import sys
import os
import json
import cv2
import mediapipe as mp
import numpy as np

REGION_NAMES = [
    'forehead', 'left_temple', 'right_temple', 'nose',
    'left_cheek', 'right_cheek', 'upper_lip_area', 'chin',
    'left_jawline', 'right_jawline'
]

REGION_COLORS = {
    'forehead': (255, 0, 0), 'left_temple': (0, 128, 255), 'right_temple': (0, 128, 255),
    'nose': (0, 255, 0), 'left_cheek': (255, 0, 255), 'right_cheek': (255, 0, 255),
    'upper_lip_area': (0, 255, 255), 'chin': (255, 255, 0),
    'left_jawline': (128, 0, 255), 'right_jawline': (128, 0, 255)
}


def _canonical_indices(connections):
    s = set()
    for a, b in connections:
        s.add(a)
        s.add(b)
    return s


def get_eye_and_oval_indices():
    """MediaPipe'in resmi goz/kas/iris ve yuz-oval kontur nokta gruplarini doner."""
    fm = mp.solutions.face_mesh
    eye_related = set()
    for name in ['FACEMESH_LEFT_EYE', 'FACEMESH_RIGHT_EYE',
                 'FACEMESH_LEFT_EYEBROW', 'FACEMESH_RIGHT_EYEBROW',
                 'FACEMESH_LEFT_IRIS', 'FACEMESH_RIGHT_IRIS']:
        eye_related |= _canonical_indices(getattr(fm, name))
    oval = _canonical_indices(fm.FACEMESH_FACE_OVAL)
    return eye_related, oval


def detect_landmarks(image_bgr):
    """Fotografta 478 MediaPipe Face Mesh landmark'ini tespit eder.
    Return: {landmark_index: (x_px, y_px)} veya None (yuz bulunamazsa)."""
    h, w = image_bgr.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    return {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}


def classify_regions(pts, eye_related, oval_pts):
    """478 landmark'i 10 cilt bolgesine geometrik olarak atar.
    Anchor noktalar (kas, goz kosesi, burun, agiz, cene) kullanilarak
    y/x bantlarina gore siniflandirma yapilir; yuz-oval kontur noktalari
    ayrica temple/jawline/chin siniri icin eklenir."""

    def P(i):
        return np.array(pts[i], dtype=float)

    y_eyebrow = (P(105)[1] + P(334)[1]) / 2
    y_nose_bot = P(2)[1]
    y_mouth_bot = P(17)[1]
    x_left_eye, x_right_eye = P(33)[0], P(263)[0]
    x_nose_l, x_nose_r = P(98)[0], P(327)[0]
    x_mouth_l, x_mouth_r = P(61)[0], P(291)[0]
    mouth_w = x_mouth_r - x_mouth_l
    x_chin_l = x_mouth_l + mouth_w * 0.22
    x_chin_r = x_mouth_r - mouth_w * 0.22

    regions = {r: set() for r in REGION_NAMES}

    def assign(i, x, y, is_oval=False):
        if y < y_eyebrow:
            if x < x_left_eye: regions['left_temple'].add(i)
            elif x > x_right_eye: regions['right_temple'].add(i)
            else: regions['forehead'].add(i)
        elif y < y_nose_bot:
            if x < x_left_eye: regions['left_temple'].add(i)
            elif x > x_right_eye: regions['right_temple'].add(i)
            elif is_oval: pass
            elif x_nose_l <= x <= x_nose_r: regions['nose'].add(i)
            elif x < x_nose_l: regions['left_cheek'].add(i)
            else: regions['right_cheek'].add(i)
        elif y < y_mouth_bot:
            if is_oval:
                if x < x_mouth_l: regions['left_cheek'].add(i)
                elif x > x_mouth_r: regions['right_cheek'].add(i)
            elif x_mouth_l <= x <= x_mouth_r: regions['upper_lip_area'].add(i)
            elif x < x_mouth_l: regions['left_cheek'].add(i)
            else: regions['right_cheek'].add(i)
        else:
            if x_chin_l <= x <= x_chin_r: regions['chin'].add(i)
            elif x < x_chin_l: regions['left_jawline'].add(i)
            else: regions['right_jawline'].add(i)

    for i, (x, y) in pts.items():
        if i in eye_related:
            continue
        assign(i, x, y, is_oval=False)

    for i in oval_pts:
        if i in eye_related:
            continue
        x, y = pts[i]
        assign(i, x, y, is_oval=True)

    return {k: sorted(v) for k, v in regions.items()}


def build_masks_and_crops(image_bgr, pts, regions, out_dir, suffix=''):
    """Her bolge icin convex-hull poligonu, maskesi ve kirpilmis goruntuyu
    out_dir icine kaydeder. Ayrica tum poligonlarin cizili oldugu bir
    genel-bakis goruntusu uretir."""
    os.makedirs(out_dir, exist_ok=True)
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    skipped = []

    for name, idxs in regions.items():
        if len(idxs) < 3:
            skipped.append(name)
            continue
        pts_arr = np.array([pts[i] for i in idxs], dtype=np.int32)
        hull = cv2.convexHull(pts_arr)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        x, y, bw, bh = cv2.boundingRect(hull)
        pad = 6
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)

        region_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
        crop = region_masked[y0:y1, x0:x1]
        fname = f'{suffix}_crop_{name}.jpg' if suffix else f'crop_{name}.jpg'
        cv2.imwrite(os.path.join(out_dir, fname), crop)

        cv2.polylines(overlay, [hull], isClosed=True, color=REGION_COLORS[name], thickness=3)

    overview_name = f'{suffix}_overview.jpg' if suffix else 'overview.jpg'
    cv2.imwrite(os.path.join(out_dir, overview_name), overlay)
    return skipped


def run_pipeline(image_path, out_dir, suffix=''):
    """Uctan uca: fotograf -> landmark -> bolge -> maske/crop."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Goruntu okunamadi: {image_path}")

    pts = detect_landmarks(image_bgr)
    if pts is None:
        print("UYARI: Fotografta yuz tespit edilemedi.")
        return None

    eye_related, oval_pts = get_eye_and_oval_indices()
    regions = classify_regions(pts, eye_related, oval_pts)
    skipped = build_masks_and_crops(image_bgr, pts, regions, out_dir, suffix)

    print(f"{'Bolge':<18}{'Nokta Sayisi'}")
    for r, idxs in regions.items():
        flag = '  <-- YETERSIZ NOKTA' if r in skipped else ''
        print(f"{r:<18}{len(idxs)}{flag}")

    with open(os.path.join(out_dir, f'{suffix}_regions.json' if suffix else 'regions.json'), 'w') as f:
        json.dump(regions, f, indent=2)

    return regions


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Kullanim: python facial_region_segmentation.py <foto_yolu> <cikti_klasoru> [suffix]")
        sys.exit(1)
    img_path = sys.argv[1]
    out_dir = sys.argv[2]
    suffix = sys.argv[3] if len(sys.argv) > 3 else ''
    run_pipeline(img_path, out_dir, suffix)
