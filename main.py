import cv2
import numpy as np
import os
from glob import glob

# ==============================
# CONFIGURAÇÕES
# ==============================
MIN_AREA = 70
MAX_AREA = 15000
RESIZE = (640, 480)

# ==============================
# FUNÇÃO: FILTRAR FOLHAS
# ==============================
def get_leaf_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 100, 50])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

# ==============================
# FUNÇÃO: EXTRAIR FOLHAS
# ==============================
def extract_leaves(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaves = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            pts = cnt.reshape(-1, 2)
            dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
            tip = pts[np.argmax(dists)]

            leaves.append({
                "center": np.array([cx, cy], dtype=np.float32),
                "tip": tip.astype(np.float32)
            })

    return leaves

# ==============================
# PEDIR CAMINHO
# ==============================
folder = input("Digite o caminho da pasta com as imagens: ").strip()

image_paths = sorted(glob(os.path.join(folder, "*.jpg")))

if len(image_paths) < 2:
    print("Poucas imagens.")
    exit()

# ==============================
# PRIMEIRA IMAGEM
# ==============================
first_img = cv2.imread(image_paths[0])

if first_img is None:
    print("Erro ao carregar primeira imagem.")
    exit()

first_img = cv2.resize(first_img, RESIZE)
prev_img = first_img.copy()

mask = get_leaf_mask(prev_img)
leaves = extract_leaves(mask)

points = []
for leaf in leaves:
    points.append(leaf["center"])
    points.append(leaf["tip"])

if len(points) == 0:
    print("Nenhuma folha detectada.")
    exit()

points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

tracks = [[p[0]] for p in points]

# ==============================
# LOOP PRINCIPAL
# ==============================
frame_idx = 0

for path in image_paths[1:]:

    img = cv2.imread(path)

    if img is None:
        print(f"Erro ao carregar: {path}")
        continue

    img = cv2.resize(img, RESIZE)
    display = img.copy()

    # máscara
    mask = get_leaf_mask(img)

    # ==============================
    # REDETECÇÃO SE NECESSÁRIO
    # ==============================
    if len(points) == 0:
        print("Re-detectando folhas...")
        leaves = extract_leaves(mask)

        points = []
        for leaf in leaves:
            points.append(leaf["center"])
            points.append(leaf["tip"])

        if len(points) == 0:
            prev_img = img.copy()
            continue

        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        tracks = [[p[0]] for p in points]
        prev_img = img.copy()
        continue

    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, points, None,
        winSize=(15,15),
        maxLevel=2
    )

    # ==============================
    # FALHA
    # ==============================
    if new_points is None or status is None:
        print("Perda de rastreamento.")
        points = []
        prev_img = img.copy()
        continue

    new_tracks = []
    new_points_filtered = []

    for i in range(len(status)):
        if status[i] == 1:
            pt = new_points[i][0]

            tracks[i].append(pt)
            new_tracks.append(tracks[i])
            new_points_filtered.append(pt)

            # desenhar ponto atual
            cv2.circle(display, tuple(pt.astype(int)), 4, (0, 0, 255), -1)

    # desenhar trilhas
    for track in new_tracks:
        for i in range(1, len(track)):
            p1 = tuple(track[i-1].astype(int))
            p2 = tuple(track[i].astype(int))
            cv2.line(display, p1, p2, (255, 0, 0), 1)

    if len(new_points_filtered) == 0:
        print("Todos pontos perdidos.")
        points = []
        prev_img = img.copy()
        continue

    tracks = new_tracks
    points = np.array(new_points_filtered, dtype=np.float32).reshape(-1, 1, 2)

    # ==============================
    # OVERLAY DE STATUS
    # ==============================
    cv2.putText(display, f"Frame: {frame_idx}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.putText(display, f"Pontos: {len(points)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # ==============================
    # EXIBIÇÃO
    # ==============================
    cv2.imshow("Analise", display)
    cv2.imshow("Mascara (Folhas)", mask)

    key = cv2.waitKey(50) & 0xFF
    if key == 27:  # ESC
        break

    prev_img = img.copy()
    frame_idx += 1

# ==============================
# RESULTADO FINAL
# ==============================
output = first_img.copy()

for track in tracks:
    for i in range(1, len(track)):
        p1 = tuple(track[i-1].astype(int))
        p2 = tuple(track[i].astype(int))
        cv2.line(output, p1, p2, (0, 0, 255), 2)

output_path = os.path.join(folder, "resultado_crescimento.jpg")
cv2.imwrite(output_path, output)

print("Resultado salvo em:", output_path)

cv2.imshow("Resultado Final", output)
cv2.waitKey(0)
cv2.destroyAllWindows()