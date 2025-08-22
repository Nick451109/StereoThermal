import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import cv2
import tifffile as tf
import numpy as np
from Image_registration.util import (
    histogram_match_visible_to_thermal,
    evaluate_normalized_mutual_information,
)
from util import *
from Image_registration import registration
from Fusion.fusion import *
from Disparity.disparity import compute_disparity


def main(
    imageLeft,
    imageRight,
    imageThermal,
    output_name,
    extractor,
    transformation_method="homography",
    threshold=200,

    # === [VISUALIZACION] MATCH DE KEYPOINTS ===
    visualize=False,                 # True para dibujar correspondencias
    vis_score_threshold=0.50,        # umbral de score para dibujado
    save_visuals=True, 
):
    # === [RUTA] RESULTADOS DEL REGISTRO (Datasets/TarDAL_RGBT/registration_results/<extractor>/) === 
    base_dir = os.path.join("Datasets/TarDAL_RGBT/registration_results", extractor)
    os.makedirs(base_dir, exist_ok=True)
    viz_dir = os.path.join(base_dir, "viz")
    if save_visuals:
        os.makedirs(viz_dir, exist_ok=True)

    # === [METRICA] NMI PRE (visible vs térmica, sin warp) ===
    vis_pre = cv2.imread(imageRight)  # BGR
    thr_pre = cv2.imread(imageThermal, cv2.IMREAD_GRAYSCALE)
    nmi_pre = evaluate_normalized_mutual_information(vis_pre, thr_pre)
    print(f"[METRIC] ({transformation_method}) NMI pre:  {nmi_pre:.4f}")

    # === [REGISTRO] TERMICA ALINEADA A VISIBLE ===
    image_warped, matches, scores, error = registration.procesar_imagenes(
        ruta_imagen0=imageRight,
        ruta_imagen1=imageThermal,
        extractor_tipo=extractor,
        threshold=threshold,
        transformation_method=transformation_method,
    )

    if image_warped is None:
        print(f"[ERROR] Falló el registro con {transformation_method}.")
        return

    # === [METRICA] NMI POST (visible warpeda vs térmica alineada por tamaño) ===
    thr_post = cv2.resize(thr_pre, (image_warped.shape[1], image_warped.shape[0]))
    nmi_post = evaluate_normalized_mutual_information(image_warped, thr_post)
    print(f"[METRIC] ({transformation_method}) NMI post: {nmi_post:.4f}")

    # === [VISUALIZACIÓN] CORRESPONDENCIAS (opcional) ===
    if visualize:
        png_name = f"{transformation_method}_{output_name}_viz.png"
        ruta_png = os.path.join(viz_dir, png_name)
        try:
            # === [REGISTRO] VISUALIZACIÓN MEJORADA ===
            registration.mostrar_correspondencias_mejorada(
                ruta_imagen0=imageRight,                # visible
                ruta_imagen1=imageThermal,              # térmica
                extractor_tipo=extractor,
                max_num_keypoints=8192,
                umbral_score=vis_score_threshold,
                guardar_figura=save_visuals,
                ruta_guardado=ruta_png,
                filtro_imagen0=None,    # ajusta si usas filtros
                filtro_imagen1=None,
                mostrar_correspondencias=True,
                mostrar_metricas=True,
            )
            if save_visuals:
                print(f"[VIZ] Guardada visualización: {ruta_png}")
        except Exception as e:
            print(f"[WARN] No se pudo generar la visualización: {e}")


    # === [CONCATENAR] RGB + térmica → [H, W, 4] ===
    thermal_channel = np.expand_dims(thr_post, axis=-1)
    rgbt_image = np.concatenate([image_warped, thermal_channel], axis=-1)
    print(f"[INFO] Imagen RGBT: {rgbt_image.shape}")

    # === [RUTA] Guardar TIFF con 4 canales ===
    output_path = os.path.join(
        base_dir, f"{transformation_method}_{output_name}.tiff"
    )
    tf.imwrite(
        output_path,
        rgbt_image,
        photometric="minisblack",
        metadata={"description": f"RGB + Térmico con {transformation_method}"},
        compression=None,
    )
    print(f"[INFO] Guardada: {output_path}")


if __name__ == "__main__":

    # Imagen con preprocesado (histograma ajustado)
    matched_path = histogram_match_visible_to_thermal(
        visible_rgb_path="./Datasets/TarDAL_RGBT/rgb/00370.png",
        thermal_gray_path="./Datasets/TarDAL_RGBT/thermal/00370.png",
        mostrar=True,
    )

    # -------------------- Ejemplo incremento de matches 118 a 125 matches (incorrecto) -------------------
    # Registro sin preprocesar
    # image_thermal_path = "captures/inverse/thermal_20250710_110814.png"
    # image_left_path = "captures/visible/left/LEFT_visible_20250710_110814.png"
    # image_right_path = "captures/visible/right/RIGHT_visible_20250710_110814.png"

    # Registro con histogram matching
    # image_thermal_path = "captures/inverse/thermal_20250710_110814.png"
    # image_left_path = matched_path
    # image_right_path = matched_path

    # -------------------------------------------------------------------------------------------

    # -------------- Ejemplo incremento de matches 380 a 420 matches -------------------
    image_thermal_path = "./Datasets/TarDAL_RGBT/thermal/00370.png"
    image_left_path = "./Datasets/TarDAL_RGBT/rgb/00370.png"
    image_right_path = "./Datasets/TarDAL_RGBT/rgb/00370.png"

    # image_thermal_path = "./Datasets/TarDAL_RGBT/thermal/00370.png"
    # image_left_path = matched_path
    # image_right_path = matched_path
    # -------------------------------------------------------------------------------------------

    # -------------------- Ejemplo incremento de matches 16 a 71 matches -------------------
    # image_thermal_path = "./Datasets/TarDAL_RGBT/thermal/00388.png"
    # image_left_path = "./Datasets/TarDAL_RGBT/rgb/00388.png"
    # image_right_path = "./Datasets/TarDAL_RGBT/rgb/00388.png"

    # image_thermal_path = "./Datasets/TarDAL_RGBT/thermal/00388.png"
    # image_left_path = matched_path
    # image_right_path = matched_path
    # -------------------------------------------------------------------------------------------

    # Extraer nombre base de la imagen
    base_filename = os.path.splitext(os.path.basename(image_thermal_path))[0]

    # Umbral y extractor
    threshold = 5
    extractor = "aliked"

    # Transformaciones a aplicar
    transformations = [
        "homography",
        "affine",
        "rigid",
        "similarity",
        "translation",
        "translation2",
        "rigid_ransac",
        "translation_ransac",
    ]

    for transformation_name in transformations:
        main(
            imageLeft=image_left_path,
            imageRight=image_right_path,
            imageThermal=image_thermal_path,
            output_name=base_filename,
            extractor=extractor,
            transformation_method=transformation_name,
            threshold=threshold,
        )

    print("END")
