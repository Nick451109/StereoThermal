import os
import torch
import cv2
import tifffile as tf
import numpy as np
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
):
    # Carpeta de salida: results/<extractor>/
    output_dir = os.path.join("results", extractor)
    os.makedirs(output_dir, exist_ok=True)

    # Registro: la térmica se registra a la visible
    image_warped, matches, scores, error = registration.procesar_imagenes(
        ruta_imagen0=imageRight,
        ruta_imagen1=imageThermal,
        extractor_tipo=extractor,
        threshold=threshold,
        transformation_method=transformation_method,
    )

    # print(f"[INFO] Tipos: {len(matches)}, {scores}")

    if image_warped is None:
        print(f"[ERROR] Falló el registro con {transformation_method}.")
        return

    # Leer térmica y convertir a [H, W, 1]
    thermal = cv2.imread(imageThermal, cv2.IMREAD_GRAYSCALE)
    thermal_resized = cv2.resize(
        thermal, (image_warped.shape[1], image_warped.shape[0])
    )
    thermal_channel = np.expand_dims(thermal_resized, axis=-1)

    # Concatenar RGB + térmica → [H, W, 4]
    rgbt_image = np.concatenate([image_warped, thermal_channel], axis=-1)
    print(f"[INFO] Imagen RGBT: {rgbt_image.shape}")

    # Guardar TIFF con 4 canales
    output_path = os.path.join(
        output_dir, f"{transformation_method}_{output_name}.tiff"
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
    # Rutas de las imágenes
    # image_thermal_path = "captures/inverse/thermal_20250710_110814.png"
    # image_left_path = "captures/visible/left/LEFT_visible_20250710_110814.png"
    # image_right_path = "captures/visible/right/RIGHT_visible_20250710_110814.png"

    # image_thermal_path = "Cameras/captures/thermal/thermal_20241030_130522.png"
    # image_left_path = "Cameras/captures/visible/left_rect/LEFT_visible_20241030_130522.png"
    # image_right_path = "Cameras/captures/visible/right_rect/RIGHT_visible_20241030_130522.png"

    image_thermal_path = "./Datasets/TarDAL_RGBT/thermal/00274.png"
    image_left_path = "./Datasets/TarDAL_RGBT/rgb/00274.png"
    image_right_path = "./Datasets/TarDAL_RGBT/rgb/00274.png"

    # image_thermal_path = "captures/inverse/thermal_20250710_110814.png"
    # image_left_path = "captures/visible/left/LEFT_visible_20250710_110814.png"
    # image_right_path = "captures/visible/right/RIGHT_visible_20250710_110814.png"

    # image_thermal_path = "prueba/00388_ir.png"
    # image_left_path = "prueba/00388_rgb.png"
    # image_right_path = "prueba/00388_rgb.png"


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
