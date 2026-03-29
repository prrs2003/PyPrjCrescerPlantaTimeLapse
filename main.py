import cv2
import numpy as np
import os
from glob import glob

# ==============================
# CONFIGURAÇÕES
# ==============================
MIN_AREA = 100        # mínimo para considerar folha
MAX_AREA = 10000
DIST_THRESHOLD = 20   # para associar pontos entre frames

# ==============================
# PEDIR CAMINHO
# ==============================
folder = input("Digite o caminho da pasta com as imagens: ").strip()

image_paths = sorted(glob(os.path.join(folder, "*.jpg")))

if len(image_paths) < 2:
    print("Poucas imagens.")
    exit()

# ==============================
# FUNÇÃO: FILTRAR FOLHAS
# ==============================
def get_leaf_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Verde vivo (folha)
    lower_green = np.array([35, 80, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Limpeza
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

            # ponto mais distante (ponta)
            pts = cnt.reshape(-1, 2)
            dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
            tip = pts[np.argmax(dists)]

            leaves.append({
                "center": np.array([cx, cy], dtype=np.float32),
                "tip": tip.astype(np.float32)
            })

    return leaves

# ==============================
# INICIALIZAÇÃO
# ==============================
first_img = cv2.imread(image_paths[0])
prev_img = first_img.copy()

mask = get_leaf_mask(prev_img)
leaves = extract_leaves(mask)

# pontos iniciais (centros + pontas)
points = []
for leaf in leaves:
    points.append(leaf["center"])
    points.append(leaf["tip"])

points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

# armazenar trilhas
tracks = [[p[0]] for p in points]

# ==============================
# RASTREAMENTO
# ==============================
for path in image_paths[1:]:
    img = cv2.imread(path)

    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, points, None,
        winSize=(15,15),
        maxLevel=2
    )

    good_new = new_points[status == 1]
    good_old = points[status == 1]

    new_tracks = []
    idx = 0

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        tracks[i].append(new)
        new_tracks.append(tracks[i])

    tracks = new_tracks
    points = good_new.reshape(-1, 1, 2)

    prev_img = img.copy()

# ==============================
# DESENHAR RESULTADO
# ==============================
output = first_img.copy()

for track in tracks:
    for i in range(1, len(track)):
        p1 = tuple(track[i-1].astype(int))
        p2 = tuple(track[i].astype(int))

        cv2.line(output, p1, p2, (0, 0, 255), 2)

# ==============================
# SALVAR RESULTADO
# ==============================
output_path = os.path.join(folder, "resultado_crescimento.jpg")
cv2.imwrite(output_path, output)

print("Imagem gerada em:", output_path)

# opcional visualizar
cv2.imshow("Resultado", output)
cv2.waitKey(0)
cv2.destroyAllWindows()