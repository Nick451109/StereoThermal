import os
import matplotlib.pyplot as plt
import torch
import cv2
import tifffile as tf
import numpy as np
from Image_registration.util import (
    evaluate_rmse_gray,
    histogram_match_visible_to_thermal,
    evaluate_normalized_mutual_information,
    evaluate_nrmse,
    evaluate_ncc_edges,
    evaluate_psnr_edges,
    evaluate_ssim_edges,
    evaluate_homography,
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
    usar_histogram_matching=True,
):
    print("----------------------------------------------------------------")
    # === [RUTA] RESULTADOS DEL REGISTRO (Datasets/TarDAL_RGBT/registration_results/<extractor>/) === 
    base_dir = os.path.join("Datasets/TarDAL_RGBT/registration_results", extractor)
    os.makedirs(base_dir, exist_ok=True)

    # === [RUTA] RESULTADOS VISUALIZADOR DE KEYPOINTS === 
    matches_dir = os.path.join(base_dir, "matches")
    os.makedirs(matches_dir, exist_ok=True)

    # === [METRICA] NMI PRE (visible vs térmica, sin warp) ===
    vis_pre = cv2.imread(imageRight)  # BGR
    thr_pre = cv2.imread(imageThermal, cv2.IMREAD_GRAYSCALE)
    nmi_pre = evaluate_normalized_mutual_information(vis_pre, thr_pre)
    print(f"[METRIC] ({transformation_method}) NMI pre:  {nmi_pre:.4f}")

    # === [METRICA] RMSE PRE ===
    if usar_histogram_matching:
        try:
            vis_pre_yuv = cv2.cvtColor(vis_pre, cv2.COLOR_BGR2YUV)
            Y_pre = vis_pre_yuv[:, :, 0]  # canal de luminancia
            rmse_pre = evaluate_rmse_gray(Y_pre, thr_pre)
            print(f"[METRIC] ({transformation_method}) RMSE pre:  {rmse_pre:.4f}")
        except Exception as e:
            print(f"[WARN] No se pudo calcular RMSE pre: {e}")


    # === [REGISTRO] TERMICA ALINEADA A VISIBLE ===
    image_warped, matches, scores, error = registration.procesar_imagenes(
        ruta_imagen0=imageRight,
        ruta_imagen1=imageThermal,
        extractor_tipo=extractor,
        threshold=threshold,
        transformation_method=transformation_method,
        visualize=True,
        visualize_save_path=os.path.join(
            matches_dir, f"{transformation_method}_{output_name}_matches.png"
        ),
    )

    if image_warped is None:
        print(f"[ERROR] Falló el registro con {transformation_method}.")
        return
    
    # === [METRICA] ERROR DE REPROYECCION ===
    try:
        if error is not None and isinstance(error, (float, int)):
            print(f"[METRIC] ({transformation_method}) Error reproyección: {error:.4f}")
    except Exception as e:
        print(f"[WARN] No se pudo calcular error de reproyección: {e}")

    # === [METRICA] NMI POST (visible warpeda vs térmica alineada por tamaño) ===
    thr_post = cv2.resize(thr_pre, (image_warped.shape[1], image_warped.shape[0]))
    nmi_post = evaluate_normalized_mutual_information(image_warped, thr_post)
    print(f"[METRIC] ({transformation_method}) NMI post: {nmi_post:.4f}")

    # === [METRICA] RMSE post sobre canal Y igualado vs térmica ===
    # Convertir image_warped (BGR) a YUV y extraer canal Y
    if usar_histogram_matching:
        image_warped_yuv = cv2.cvtColor(image_warped, cv2.COLOR_BGR2YUV)
        Y_warped = image_warped_yuv[:, :, 0]
        rmse_post = evaluate_rmse_gray(Y_warped, thr_post)
        print(f"[METRIC] ({transformation_method}) RMSE post: {rmse_post:.4f}")
        image_warped_gray = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY)
        print(f"[METRIC] ({transformation_method}) NRMSE: {evaluate_nrmse(image_warped_gray, thr_post):.4f}")


    # === [METRICA] ===
    print(f"[EXP.METRIC] ({transformation_method}) NCC Sobel: {evaluate_ncc_edges(image_warped, thr_post, method='sobel'):.4f}")
    print(f"[EXP.METRIC] ({transformation_method}) PSNR Sobel: {evaluate_psnr_edges(image_warped, thr_post, method='sobel'):.4f}")
    print(f"[EXP.METRIC] ({transformation_method}) SSIM Sobel: {evaluate_ssim_edges(image_warped, thr_post, method='sobel'):.4f}")

    print(f"[EXP.METRIC] ({transformation_method}) NCC Canny: {evaluate_ncc_edges(image_warped, thr_post, method='canny'):.4f}")
    print(f"[EXP.METRIC] ({transformation_method}) PSNR Canny: {evaluate_psnr_edges(image_warped, thr_post, method='canny'):.4f}")
    print(f"[EXP.METRIC] ({transformation_method}) SSIM Canny: {evaluate_ssim_edges(image_warped, thr_post, method='canny'):.4f}")

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
    print(f"[INFO] .TIFF Guardado: {output_path}")
    print("----------------------------------------------------------------")


if __name__ == "__main__":

    # Ruta base de la imagen térmica (constante)
    image_thermal_path = "./Datasets/parallax_Tardal/thermal_out/00082.png"

    # Cambiar esto a True solo cuando quieras aplicar histogram matching
    APLICAR_HISTOGRAM_MATCHING = True

    if APLICAR_HISTOGRAM_MATCHING:
        matched_path = histogram_match_visible_to_thermal(
            visible_rgb_path="./Datasets/parallax_Tardal/rgb_out/00082.png",
            thermal_gray_path=image_thermal_path,
            mostrar=True,
        )
        image_left_path = matched_path
        image_right_path = matched_path
    else:
        image_left_path = "./Datasets/parallax_Tardal/rgb_out/00082.png"
        image_right_path = "./Datasets/parallax_Tardal/rgb_out/00082.png"



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
    # image_thermal_path = "./Datasets/TarDAL_RGBT/thermal/00370.png"
    # image_left_path = "./Datasets/TarDAL_RGBT/rgb/00370.png"
    # image_right_path = "./Datasets/TarDAL_RGBT/rgb/00370.png"

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
    threshold = 0
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
            usar_histogram_matching=APLICAR_HISTOGRAM_MATCHING,
        )

    print("END")
