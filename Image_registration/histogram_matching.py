import cv2
import numpy as np
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
import os

def histogram_matching_termica_a_visible(
    ruta_termica,
    ruta_visible,
    ruta_salida="salidas/termica_matched.png",
    mostrar=True
):
    # Leer imágenes en escala de grises
    termica = cv2.imread(ruta_termica, cv2.IMREAD_GRAYSCALE)
    visible = cv2.imread(ruta_visible, cv2.IMREAD_GRAYSCALE)

    if termica is None or visible is None:
        raise FileNotFoundError("No se pudo cargar una de las imágenes.")

    # Aplicar histogram matching: Térmica → Visible
    termica_ajustada = match_histograms(termica, visible).astype(np.uint8)

    # Guardar resultado
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    cv2.imwrite(ruta_salida, termica_ajustada)
    print(f"[INFO] Imagen térmica ajustada guardada en: {ruta_salida}")

    # Mostrar comparación si se desea
    if mostrar:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(termica, cmap="gray")
        plt.title("Térmica original")

        plt.subplot(1, 3, 2)
        plt.imshow(visible, cmap="gray")
        plt.title("Visible (referencia)")

        plt.subplot(1, 3, 3)
        plt.imshow(termica_ajustada, cmap="gray")
        plt.title("Térmica ajustada")
        plt.tight_layout()
        plt.show()

    return termica_ajustada

# === EJEMPLO DE USO ===
ruta_termica = "captures/inverse/thermal_20250710_110814.png"
ruta_visible = "captures/visible/right/RIGHT_visible_20250710_110814.png"
ruta_salida = "histogram_matching/visible_matched.png"

histogram_matching_termica_a_visible(ruta_termica, ruta_visible, ruta_salida)
