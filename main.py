from array import array
import torch
import cv2
from util import *
from Image_registration import registration
from Fusion.fusion import *
import tifffile as tf
from Disparity.disparity import compute_disparity
import numpy as np


def main(
    imageLeft,
    imageRight,
    imageThermal,
    threshold=200,
    transformation_method="homography",
):
    # Registro: la térmica se registra a la visible
    image_warped, matches, scores, error = registration.procesar_imagenes(
        ruta_imagen0=imageRight,
        ruta_imagen1=imageThermal,
        threshold=threshold,
        transformation_method=transformation_method,
    )
    
    if image_warped is None:
        print("[ERROR] Falló el registro.")
        return

    # Leer térmica y convertir a [H, W, 1]
    thermal = cv2.imread(imageThermal, cv2.IMREAD_GRAYSCALE)
    thermal_resized = cv2.resize(thermal, (image_warped.shape[1], image_warped.shape[0]))
    thermal_channel = np.expand_dims(thermal_resized, axis=-1)

    # Concatenar RGB + térmica → [H, W, 4]
    rgbt_image = np.concatenate([image_warped, thermal_channel], axis=-1)
    print(f"[INFO] Imagen RGBT: {rgbt_image.shape}")

    # Mostrar
    # cv2.imshow("Imagen Registrada (RGB)", image_warped)
    # cv2.imshow("Canal térmico", thermal_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Guardar TIFF con 4 canales
    tf.imwrite(
        "image_rgbt_registered.tiff",
        rgbt_image,
        photometric="minisblack",  # 'rgb' solo funciona para 3 canales
        metadata={"description": "Imagen registrada RGB + Térmico"},
        compression=None,
    )


if __name__ == "__main__":

    # Rutas de las imágenes
    image_thermal_path = "captures/inverse/thermal_20250710_110814.png"
    image_left_path = "captures/visible/left/LEFT_visible_20250710_110814.png"
    image_right_path = "captures/visible/right/RIGHT_visible_20250710_110814.png"

    # image_thermal_path = "Cameras/captures/thermal/thermal_20241030_130522.png"
    # image_left_path = "Cameras/captures/visible/left_rect/LEFT_visible_20241030_130522.png"
    # image_right_path = "Cameras/captures/visible/right_rect/RIGHT_visible_20241030_130522.png"

    # pocos matches
    # image_thermal_path = "./captures/thermal/thermal_20250612_152008.png"
    # image_left_path = "./captures/visible/left/LEFT_visible_20250612_152008.png"
    # image_right_path = "./captures/visible/right/RIGHT_visible_20250612_152008.png"

    # image_thermal_path = "./captures/thermal/thermal_20250625_102056.png"
    # image_left_path = "./captures/visible/left/LEFT_visible_20250625_102056.png"
    # image_right_path = "./captures/visible/right/RIGHT_visible_20250625_102056.png"

    transformation_name = "homography"  # "homography", "affine", "similarity"
    # Umbral para el registro
    threshold = 70

    # cambiar a threshold=200

    main(
        imageLeft=image_left_path,
        imageRight=image_right_path,
        imageThermal=image_thermal_path,
        threshold=threshold,
        transformation_method=transformation_name,
    )

    print("END")
