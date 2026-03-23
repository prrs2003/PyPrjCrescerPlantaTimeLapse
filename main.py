### Versão 0.1.1

import os
import cv2
import numpy as np
from tqdm import tqdm


class BioTrackerPure:
    def __init__(self):
        # Range HSV muito mais restrito para o "Verde Clorofila"
        # Saturação mínima subiu para 80 (ignora tons pastéis/beges)
        # Valor (brilho) mínimo subiu para 50 (ignora sombras escuras)
        self.lower_green = np.array([35, 80, 50])
        self.upper_green = np.array([85, 255, 255])

    def get_vegetation_mask(self, img):
        """Aplica múltiplos filtros para garantir que APENAS vegetação seja detectada."""
        # 1. Suavização para remover ruído eletrônico do sensor
        work_img = cv2.GaussianBlur(img, (5, 5), 0)

        # 2. Separação de canais (Float para cálculos precisos)
        b, g, r = cv2.split(work_img.astype(np.float32))

        # 3. Cálculo do ExG (Excess Green Index)
        # Fórmula: 2*G - R - B. Folhas dão valores altos, madeira/bambu dão valores baixos.
        exg = 2 * g - r - b
        exg_mask = np.where(exg > 50, 255, 0).astype(np.uint8)

        # 4. Filtro de Razão (O verde DEVE ser dominante)
        # Se G não for pelo menos 1.2x maior que R, provavelmente é amarelado (bambu)
        ratio_mask = np.where((g > r * 1.2) & (g > b * 1.1), 255, 0).astype(np.uint8)

        # 5. Filtro HSV Estrito
        hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # 6. COMBINAÇÃO DAS TRAVAS (Só é planta se passar em TODOS os testes)
        # ExG + Razão + HSV
        combined = cv2.bitwise_and(exg_mask, ratio_mask)
        combined = cv2.bitwise_and(combined, hsv_mask)

        # 7. Limpeza Morfológica (Fecha buracos nas folhas e remove pontinhos)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)  # Remove sujeira
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Une partes da folha

        # 8. Filtro de Área Mínima (Remove manchas pequenas que não são plantas)
        final_mask = np.zeros_like(mask)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 400:  # Aumentado para ignorar qualquer resíduo
                cv2.drawContours(final_mask, [c], -1, 255, -1)

        return final_mask

    def get_temporal_color(self, idx, total):
        """Azul (antigo) -> Vermelho (novo)."""
        hue = int((idx / total) * 160)
        hsv = np.uint8([[[hue, 255, 255]]])
        return tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]))

    def process(self, folder_path, alpha=0.65):
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))],
                       key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
        if len(files) < 2: return

        # Imagem base (última do timelapse)
        last_img = cv2.imread(os.path.join(folder_path, files[-1]))
        h, w = last_img.shape[:2]

        # Identificar Indivíduos (Plantas separadas)
        final_mask = self.get_vegetation_mask(last_img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, 8)

        plants = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 500:
                plants.append({'id': len(plants) + 1, 'center': centroids[i]})

        print(f"Detectadas {len(plants)} plantas (Vegetação Pura).")

        overlay = np.zeros_like(last_img)
        for frame_idx, filename in enumerate(tqdm(files, desc="Filtrando Clorofila")):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is None: continue

            mask = self.get_vegetation_mask(img)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = self.get_temporal_color(frame_idx, len(files))

            for c in cnts:
                # Associa ao indivíduo mais próximo para desenhar a borda
                M = cv2.moments(c);
                cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
                cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0

                for p in plants:
                    if np.sqrt((cx - p['center'][0]) ** 2 + (cy - p['center'][1]) ** 2) < 250:
                        cv2.drawContours(overlay, [c], -1, color, 2, cv2.LINE_AA)
                        break

        # Renderização Final
        result = cv2.addWeighted(last_img, 1.0, overlay, alpha, 0)
        for p in plants:
            pos = (int(p['center'][0]), int(p['center'][1]))
            cv2.putText(result, f"PLANTA #{p['id']}", pos, cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(result, f"PLANTA #{p['id']}", pos, cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1,
                        cv2.LINE_AA)

        out_path = os.path.join(folder_path, "RESULTADO_VEGETACAO_ESTRITO.jpg")
        cv2.imwrite(out_path, result)
        print(f"\nMapa de vegetação gerado com sucesso!")


if __name__ == "__main__":
    path = input("Caminho da pasta: ")
    BioTrackerPure().process(path)