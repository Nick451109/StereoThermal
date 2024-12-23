import os
import cv2
import torch
import numpy as np
from Disparity.Selective_IGEV.bridge_selective import get_SELECTIVE_disparity_map

def compute_disparity(img_left: np.array, img_right: np.array):
    """
    Calcula el mapa de disparidad de un par de imágenes usando el método especificado en la configuración.
    
    :param img_left: Imagen del lado izquierdo como array de numpy.
    :param img_right: Imagen del lado derecho como array de numpy.
    :return: Mapa de disparidad como array de numpy.
    """
    # Acceso a los métodos de disparidad configurados
    
    if isinstance(img_left, str):
        img_left = cv2.imread(img_left, cv2.IMREAD_COLOR)
        if img_left is None:
            raise ValueError("No se pudo cargar la imagen izquierda desde la ruta proporcionada.")


    if isinstance(img_right, str):
        img_right = cv2.imread(img_right, cv2.IMREAD_COLOR)
        if img_right is None:
            raise ValueError("No se pudo cargar la imagen derecha desde la ruta proporcionada.")
        
    # Usar Selective para calcular disparidad
    disparity = get_SELECTIVE_disparity_map(
        restore_ckpt="Disparity/Selective_IGEV/pretrained_models/middlebury_train.pth",
        img_left_array=img_left,
        img_right_array=img_right,
        save_numpy=True,
        slow_fast_gru=True,
    )
    disparity = disparity[0]
    
    return disparity

