import os
import cv2
import numpy as np
from tqdm import tqdm


class PlantIndividualTracker:
    def __init__(self):
        # Filtros de Clorofila (ExG + HSV)
        self.lower_green = np.array([35, 60, 45])
        self.upper_green = np.array([95, 255, 255])

    def get_mask(self, img):
        """Isola apenas o tecido vegetal vivo."""
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        # Índice Excess Green
        b, g, r = cv2.split(blur.astype(np.float32))
        exg = 2 * g - r - b
        exg = np.clip(exg, 0, 255).astype(np.uint8)
        _, m_exg = cv2.threshold(exg, 40, 255, cv2.THRESH_BINARY)
        # Filtro de Cor
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        m_hsv = cv2.inRange(hsv, self.lower_green, self.upper_green)
        # Fusão e Limpeza
        mask = cv2.bitwise_and(m_exg, m_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return mask

    def get_temporal_color(self, idx, total):
        """Gradiente: Azul (Passado) -> Verde -> Vermelho (Presente)."""
        hue = int((idx / total) * 160)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def process(self, folder_path, alpha=0.6):
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))],
                       key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
        if len(files) < 2: return

        # 1. Identificar Indivíduos na ÚLTIMA foto (onde estão maiores e mais visíveis)
        last_img = cv2.imread(os.path.join(folder_path, files[-1]))
        h, w = last_img.shape[:2]
        final_mask = self.get_mask(last_img)

        # Encontra componentes conectados (Indivíduos)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)

        # Filtrar indivíduos por tamanho (min 500 pixels) e guardar posições
        plants = []
        for i in range(1, num_labels):  # Pula o fundo (index 0)
            if stats[i, cv2.CC_STAT_AREA] > 500:
                plants.append({
                    'id': len(plants) + 1,
                    'center': centroids[i],
                    'contours_history': []
                })

        print(f"Identificadas {len(plants)} plantas distintas.")

        # 2. Processar cronologicamente os contornos de cada indivíduo
        overlay = np.zeros_like(last_img)

        for frame_idx, filename in enumerate(tqdm(files, desc="Analisando Cronologia")):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is None: continue

            mask = self.get_mask(img)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            color = self.get_temporal_color(frame_idx, len(files))

            for c in cnts:
                if cv2.contourArea(c) < 150: continue

                # Encontrar a qual planta este contorno pertence (por proximidade)
                M = cv2.moments(c)
                if M['m00'] == 0: continue
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

                # Acha a planta ID mais próxima deste contorno
                best_dist = float('inf')
                best_id = -1
                for p in plants:
                    dist = np.sqrt((cx - p['center'][0]) ** 2 + (cy - p['center'][1]) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = p['id']

                # Se estiver perto o suficiente de uma planta conhecida, desenha
                if best_dist < 200:  # Raio de busca de 200px
                    cv2.drawContours(overlay, [c], -1, color, 2, cv2.LINE_AA)

        # 3. Mesclar e Adicionar Etiquetas (Labels)
        result = cv2.addWeighted(last_img, 1.0, overlay, alpha, 0)

        for p in plants:
            pos = (int(p['center'][0]) - 10, int(p['center'][1]))
            # Sombra da etiqueta para leitura
            cv2.putText(result, f"#{p['id']}", pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            # Etiqueta branca
            cv2.putText(result, f"#{p['id']}", pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Salvar final
        out_path = os.path.join(folder_path, "MAPA_INDIVIDUALIZADO.jpg")
        cv2.imwrite(out_path, result)
        print(f"\nFinalizado! Analisadas {len(plants)} plantas.")
        print(f"Resultado salvo em: {out_path}")


if __name__ == "__main__":
    path = input("Digite o caminho da pasta: ")
    PlantIndividualTracker().process(path)