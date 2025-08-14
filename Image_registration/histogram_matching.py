import cv2
import numpy as np
from skimage.exposure import match_histograms
import os
import matplotlib.pyplot as plt

def histogram_match_visible_to_thermal(visible_rgb_path, thermal_gray_path, output_path="salidas/visible_matched.png", mostrar=True):
    # Leer imagen visible en RGB y térmica en escala de grises
    visible_bgr = cv2.imread(visible_rgb_path)
    thermal_gray = cv2.imread(thermal_gray_path, cv2.IMREAD_GRAYSCALE)

    if visible_bgr is None or thermal_gray is None:
        raise FileNotFoundError("No se pudo cargar una de las imágenes.")

    # Convertir visible de BGR a YUV (separamos luminancia)
    visible_yuv = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(visible_yuv)

    # Aplicar histogram matching: canal Y → thermal
    Y_matched = match_histograms(Y, thermal_gray).astype(np.uint8)

    # Reconstruir imagen YUV con Y ajustado
    matched_yuv = cv2.merge([Y_matched, U, V])
    matched_bgr = cv2.cvtColor(matched_yuv, cv2.COLOR_YUV2BGR)

    # Guardar imagen RGB final
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, matched_bgr)
    print(f"[INFO] Imagen visible ajustada guardada en: {output_path}")

    # Mostrar imágenes si se desea
    if mostrar:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Visible original (RGB)")

        plt.subplot(1, 3, 2)
        plt.imshow(thermal_gray, cmap="gray")
        plt.title("Térmica (referencia)")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Visible ajustada (RGB)")
        plt.tight_layout()
        plt.show()

    return matched_bgr

# === USO ===
histogram_match_visible_to_thermal(
    visible_rgb_path="captures/visible/right/RIGHT_visible_20250710_110814.png",
    thermal_gray_path="captures/inverse/thermal_20250710_110814.png",
    output_path="histogram_matching/visible_matched.png"
)

