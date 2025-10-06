from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image, to_tensor
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance,  ImageFilter
import torch
import cv2
import torch.nn.functional as F
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
import os
import csv
import pandas as pd


def apply_filter_grayscale(imagen_tensor):
    """
    Convierte una imagen a escala de grises y replica los canales para mantener la compatibilidad de 3 canales.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.

    Returns:
        torch.Tensor: Tensor de la imagen filtrada en escala de grises con 3 canales.
    """
    imagen_pil = to_pil_image(imagen_tensor.cpu())
    imagen_grayscale = imagen_pil.convert("L")
    imagen_grayscale_tensor = torch.tensor(np.array(imagen_grayscale)).unsqueeze(0).repeat(3, 1, 1).float() / 255.0
    print(f"Grayscale Tensor - Min: {imagen_grayscale_tensor.min().item()}, "
          f"Max: {imagen_grayscale_tensor.max().item()}, "
          f"Mean: {imagen_grayscale_tensor.mean().item()}")
    return imagen_grayscale_tensor.to(imagen_tensor.device)

def apply_filter_hsv(imagen_tensor, factor_saturacion=1.5):
    """
    Aumenta la saturación de una imagen en el espacio de color HSV.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.
        factor_saturacion (float, opcional): Factor para aumentar la saturación. Por defecto es 1.5.

    Returns:
        torch.Tensor: Tensor de la imagen filtrada con saturación aumentada.
    """
    imagen_pil = to_pil_image(imagen_tensor.cpu())
    imagen_hsv = imagen_pil.convert("HSV")
    h, s, v = imagen_hsv.split()
    s = s.point(lambda i: min(int(i * factor_saturacion), 255))
    imagen_hsv_modificada = Image.merge("HSV", (h, s, v))
    imagen_rgb = imagen_hsv_modificada.convert("RGB")
    imagen_rgb_tensor = to_tensor(imagen_rgb).to(imagen_tensor.device)
    print(f"HSV Tensor - Min: {imagen_rgb_tensor.min().item()}, "
          f"Max: {imagen_rgb_tensor.max().item()}, "
          f"Mean: {imagen_rgb_tensor.mean().item()}")
    return imagen_rgb_tensor

def apply_filter_contraste(imagen_tensor, factor_contraste=1.5):
    """
    Ajusta el contraste de una imagen.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.
        factor_contraste (float, opcional): Factor por el cual se ajusta el contraste. >1 aumenta el contraste, <1 lo disminuye.

    Returns:
        torch.Tensor: Tensor de la imagen con contraste ajustado.
    """
    imagen_pil = to_pil_image(imagen_tensor.cpu())
    enhancer = ImageEnhance.Contrast(imagen_pil)
    imagen_contraste = enhancer.enhance(factor_contraste)
    imagen_contraste_tensor = to_tensor(imagen_contraste).to(imagen_tensor.device)
    print(f"Contrast Tensor - Min: {imagen_contraste_tensor.min().item()}, "
          f"Max: {imagen_contraste_tensor.max().item()}, "
          f"Mean: {imagen_contraste_tensor.mean().item()}")
    return imagen_contraste_tensor

def apply_filter_blur(imagen_tensor, radio=2):
    """
    Aplica un desenfoque gaussiano a una imagen.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.
        radio (float, opcional): Radio del desenfoque.

    Returns:
        torch.Tensor: Tensor de la imagen con desenfoque aplicado.
    """
    imagen_pil = to_pil_image(imagen_tensor.cpu())
    imagen_blur = imagen_pil.filter(ImageFilter.GaussianBlur(radius=radio))
    imagen_blur_tensor = to_tensor(imagen_blur).to(imagen_tensor.device)
    print(f"Blur Tensor - Min: {imagen_blur_tensor.min().item()}, "
          f"Max: {imagen_blur_tensor.max().item()}, "
          f"Mean: {imagen_blur_tensor.mean().item()}")
    return imagen_blur_tensor

def apply_filter_canny(imagen_tensor, threshold1=100, threshold2=200):
    """
    Aplica la detección de bordes Canny a una imagen.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.
        threshold1 (int, opcional): Primer umbral para la detección de bordes.
        threshold2 (int, opcional): Segundo umbral para la detección de bordes.

    Returns:
        torch.Tensor: Tensor de la imagen con bordes detectados.
    """
    imagen_np = imagen_tensor.cpu().permute(1, 2, 0).numpy() * 255.0
    imagen_np = imagen_np.astype(np.uint8)
    imagen_gray = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(imagen_gray, threshold1, threshold2)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_tensor = torch.tensor(edges_rgb).permute(2, 0, 1).float() / 255.0
    return edges_tensor.to(imagen_tensor.device)

def apply_filter(imagen_tensor, tipo_filtro):
    """
    Aplica un filtro especificado a una imagen utilizando funciones dedicadas.

    Args:
        imagen_tensor (torch.Tensor): Tensor de la imagen a filtrar.
        tipo_filtro (str, opcional): Tipo de filtro a aplicar. Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none". Por defecto es "none".

    Returns:
        torch.Tensor: Tensor de la imagen filtrada.
    """
    tipo_filtro = tipo_filtro.lower() if tipo_filtro else "none"
    if tipo_filtro == "none":
        #print("No se aplicará ningún filtro.")
        return imagen_tensor
    elif tipo_filtro == "grayscale":
        return apply_filter_grayscale(imagen_tensor)
    elif tipo_filtro == "hsv":
        return apply_filter_hsv(imagen_tensor)
    elif tipo_filtro == "contrast":
        return apply_filter_contraste(imagen_tensor)
    elif tipo_filtro == "blur":
        return apply_filter_blur(imagen_tensor)
    elif tipo_filtro == "canny":
        return apply_filter_canny(imagen_tensor)
    else:
        raise ValueError(f"Filtro tipo '{tipo_filtro}' no es válido. "
                         f"Selecciona entre 'grayscale', 'hsv', 'blur', 'contrast', 'canny', 'none'.")
    
def histogram_match_visible_to_thermal(visible_rgb_path, thermal_gray_path, output_path="histogram_matched_results/visible_matched.png", mostrar=True):
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

    # imagen RGB final
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, matched_bgr)
    print(f"[INFO] Imagen visible ajustada guardada en: {output_path}")

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

    return output_path

def calcular_distancia_media(pts0, pts1):
    """
    Calcula la distancia euclidiana media entre dos conjuntos de puntos.

    Args:
        pts0 (np.ndarray): Conjunto de puntos en la primera imagen.
        pts1 (np.ndarray): Conjunto de puntos en la segunda imagen.

    Returns:
        float: Distancia euclidiana media.
    """
    if len(pts0) == 0 or len(pts1) == 0:
        return float('inf')
    distancias = np.linalg.norm(pts0 - pts1, axis=1)
    return np.mean(distancias)

# ======================================
# [INICIO METRICAS] 
# ======================================

# === [METRICAS LOCALES] ===
def count_matches(matches):
    """
    [Métrica local]
    Número total de matches detectados por el matcher (LightGlue, etc.).
    """
    return len(matches) if matches is not None else 0

def count_inliers(mask):
    """
    [Métrica local]
    Número de inliers después de aplicar RANSAC al modelo geométrico.
    """
    if mask is None:
        return 0
    return int(np.sum(mask))

def count_outliers(num_matches, num_inliers):
    """
    [Métrica local]
    Número de outliers = matches totales - inliers.
    """
    return max(0, num_matches - num_inliers)

def compute_inlier_ratio(num_inliers, num_matches):
    """
    [Métrica local]
    Proporción de inliers respecto al total de matches.
    """
    if num_matches == 0:
        return 0.0
    return num_inliers / num_matches

def compute_outlier_ratio(num_outliers, num_matches):
    """
    [Métrica local]
    Proporción de outliers respecto al total de matches.
    """
    if num_matches == 0:
        return 0.0
    return num_outliers / num_matches

def compute_matching_score(num_inliers, pts0, pts1):
    """
    [Métrica local]
    Matching Score (HPatches-style): inliers / min(#keypoints en cada imagen).
    """
    min_kpts = min(len(pts0), len(pts1))
    if min_kpts == 0:
        return 0.0
    return num_inliers / min_kpts

# === [METRICAS GLOBALES] ===
def evaluate_homography(M, pts0, pts1):
    """
    Evalúa la homografía calculando el error de reproyección.

    Parámetros:
        M (np.ndarray): Matriz de transformación. Puede ser:
                        - Homografía completa de tamaño (3, 3)
                        - Transformación afín/similaridad de tamaño (2, 3)
        pts0 (np.ndarray): Puntos en la primera imagen (N, 2).
        pts1 (np.ndarray): Puntos correspondientes en la segunda imagen (N, 2).

    Retorna:
        float: Error medio de reproyección.
    """
    # Si no hay matriz o no hay puntos, el error es infinito
    if M is None or pts0.size == 0 or pts1.size == 0:
        return float('inf')

    # Asegurar que M sea una homografía 3x3, si es afín (2x3) la convertimos a (3x3)
    if M.shape == (2, 3):
        H = np.eye(3, dtype=M.dtype)
        H[:2, :] = M
    elif M.shape == (3, 3):
        H = M
    else:
        raise ValueError(f"La matriz M tiene dimensiones no soportadas: {M.shape}")

    # Asegurar que pts0 y pts1 sean Nx2
    if pts0.shape[1] != 2 or pts1.shape[1] != 2:
        raise ValueError("pts0 y pts1 deben ser de la forma Nx2.")

    # Convertir pts0 a coordenadas homogéneas Nx3
    pts0_homog = np.hstack([pts0, np.ones((pts0.shape[0], 1), dtype=pts0.dtype)])

    # Transformar pts0 con H, resultado Nx3
    pts0_transformed = (H @ pts0_homog.T).T

    # Normalizar por la tercera coordenada
    # Evitar división por cero
    w = pts0_transformed[:, 2]
    w[w == 0] = 1.0  # Si hubiera algún 0, lo evitamos, aunque no debería ocurrir con una homografía válida.
    pts0_transformed /= w[:, np.newaxis]

    # Tomar solo x,y
    pts0_transformed = pts0_transformed[:, :2]

    # Calcular error de reproyección
    errors = np.linalg.norm(pts0_transformed - pts1, axis=1)
    return np.mean(errors)

def evaluate_normalized_mutual_information(imgA, imgB, bins=64):
    """
    Calcula NMI entre dos imágenes 2D (uint8).
    NMI = (H(A) + H(B)) / H(A,B). Devuelve >= 1; cuanto mayor, mejor dependencia.
    """

    # Asegurar 2D uint8
    def to_gray_u8(x):
        if x.ndim == 3:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    A = to_gray_u8(imgA)
    B = to_gray_u8(imgB)

    # Ajustar tamaño si difieren
    if A.shape != B.shape:
        B = cv2.resize(B, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Histogramas conjuntos y marginales
    joint_hist, _, _ = np.histogram2d(A.ravel(), B.ravel(), bins=bins, range=[[0,255],[0,255]])
    joint_prob = joint_hist / (np.sum(joint_hist) + 1e-12)
    pA = np.sum(joint_prob, axis=1)  # marginal A
    pB = np.sum(joint_prob, axis=0)  # marginal B

    # Entropías
    def H(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p + 1e-12))
    HA = H(pA)
    HB = H(pB)
    HAB = H(joint_prob.ravel())

    # NMI
    nmi = (HA + HB) / (HAB + 1e-12)
    return float(nmi)

def evaluate_rmse_gray(img1, img2):
    """
    Calcula el RMSE entre dos imágenes en escala de grises.
    Se espera que ambas estén del mismo tamaño y tipo uint8 o float32.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    return np.sqrt(mse)

# === [METRICAS CROSS-SPECTRAL EN BORDES] ===
def evaluate_nrmse(img1, img2):
    """
    Calcula el NRMSE (Normalized RMSE) entre dos imágenes en escala de grises.
    Normaliza con el rango dinámico (255 para imágenes uint8).
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / 255.0
    return float(nrmse)

def evaluate_ncc_edges(img1, img2, method="sobel"):
    """
    Calcula el NCC (Normalized Cross Correlation) entre bordes de dos imágenes.
    method: "sobel" o "canny"
    """
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if method == "sobel":
        edges1 = cv2.Sobel(img1, cv2.CV_32F, 1, 1, ksize=3)
        edges2 = cv2.Sobel(img2, cv2.CV_32F, 1, 1, ksize=3)
    elif method == "canny":
        edges1 = cv2.Canny(img1, 100, 200)
        edges2 = cv2.Canny(img2, 100, 200)
    else:
        raise ValueError("Método no válido. Usa 'sobel' o 'canny'.")

    # Normalizar a float
    edges1 = edges1.astype(np.float32)
    edges2 = edges2.astype(np.float32)

    num = np.sum((edges1 - edges1.mean()) * (edges2 - edges2.mean()))
    den = np.sqrt(np.sum((edges1 - edges1.mean())**2) * np.sum((edges2 - edges2.mean())**2) + 1e-12)
    return float(num / den)

def evaluate_psnr_edges(img1, img2, method="sobel"):
    """
    Calcula el PSNR en bordes entre dos imágenes.
    method: "sobel" o "canny"
    """
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if method == "sobel":
        edges1 = cv2.Sobel(img1, cv2.CV_8U, 1, 1, ksize=3)
        edges2 = cv2.Sobel(img2, cv2.CV_8U, 1, 1, ksize=3)
    elif method == "canny":
        edges1 = cv2.Canny(img1, 100, 200)
        edges2 = cv2.Canny(img2, 100, 200)
    else:
        raise ValueError("Método no válido. Usa 'sobel' o 'canny'.")

    return cv2.PSNR(edges1, edges2)

def evaluate_ssim_edges(img1, img2, method="sobel"):
    """
    Calcula el SSIM en bordes entre dos imágenes.
    method: "sobel" o "canny"
    """
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if method == "sobel":
        edges1 = cv2.Sobel(img1, cv2.CV_8U, 1, 1, ksize=3)
        edges2 = cv2.Sobel(img2, cv2.CV_8U, 1, 1, ksize=3)
    elif method == "canny":
        edges1 = cv2.Canny(img1, 100, 200)
        edges2 = cv2.Canny(img2, 100, 200)
    else:
        raise ValueError("Método no válido. Usa 'sobel' o 'canny'.")

    ssim_val = ssim(edges1, edges2, data_range=edges2.max() - edges2.min())
    return float(ssim_val)

# == [PROCESAR METRICAS LOCALES - GLOBALES] ===
def compute_local_metrics(pts0, pts1, matches, mask, M):
    """
    Calcula todas las métricas locales y el error de reproyección.
    Maneja correctamente el caso en que no existe máscara (sin RANSAC).
    Retorna un diccionario con valores compatibles para exportar a Excel.
    """

    # --- Total de matches ---
    num_matches = count_matches(matches)

    # Si no hay matches, devolver todo como N/A
    if num_matches == 0:
        return {
            "Matches": 0,
            "Inliers": "N/A",
            "Outliers": "N/A",
            "Inlier_ratio": "N/A",
            "Outlier_ratio": "N/A",
            "Matching_score": "N/A",
            "Error_reprojection": "N/A",
            "RANSAC": "No"
        }

    # --- Si hay máscara (modelo robusto) ---
    if mask is not None:
        num_inliers = count_inliers(mask)
        num_outliers = count_outliers(num_matches, num_inliers)
        inlier_ratio = round(compute_inlier_ratio(num_inliers, num_matches), 4)
        outlier_ratio = round(compute_outlier_ratio(num_outliers, num_matches), 4)
        matching_score = round(compute_matching_score(num_inliers, pts0, pts1), 4)
        ransac_status = "Sí"
    else:
        # --- Sin RANSAC: no hay información real de inliers/outliers ---
        num_inliers = "N/A"
        num_outliers = "N/A"
        inlier_ratio = "Sin RANSAC"
        outlier_ratio = "Sin RANSAC"
        matching_score = "N/A"
        ransac_status = "No"

    # --- Error de reproyección ---
    if M is not None and pts0.size > 0 and pts1.size > 0:
        error = round(evaluate_homography(M, pts0, pts1), 4)
    else:
        error = "N/A"

    # --- Retornar diccionario completo ---
    return {
        "Matches": num_matches,
        "Inliers": num_inliers,
        "Outliers": num_outliers,
        "Inlier_ratio": inlier_ratio,
        "Outlier_ratio": outlier_ratio,
        "Matching_score": matching_score,
        "Error_reprojection": error,
        "RANSAC": ransac_status
    }

def compute_all_metrics(image_warped, thermal, usar_histogram_matching=True):
    """
    Calcula todas las métricas globales entre la imagen registrada (RGB) y la térmica.
    Combina NMI, RMSE, NRMSE, NCC, PSNR y SSIM en bordes (Sobel y Canny).

    Parámetros:
        image_warped (np.ndarray): Imagen registrada (BGR o RGB).
        thermal (np.ndarray): Imagen térmica (grayscale).
        usar_histogram_matching (bool): Si es True, calcula RMSE/NRMSE sobre canal Y.

    Retorna:
        dict: métricas globales con formato estandarizado para exportar o imprimir.
    """
    thr_post = cv2.resize(thermal, (image_warped.shape[1], image_warped.shape[0]))
    gray_warped = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY)

    metrics = {}
    metrics["NMI_post"] = evaluate_normalized_mutual_information(image_warped, thr_post)

    if usar_histogram_matching:
        image_warped_yuv = cv2.cvtColor(image_warped, cv2.COLOR_BGR2YUV)
        Y_warped = image_warped_yuv[:, :, 0]
        metrics["RMSE_post"] = evaluate_rmse_gray(Y_warped, thr_post)
        metrics["NRMSE"] = evaluate_nrmse(gray_warped, thr_post)
    else:
        metrics["RMSE_post"] = None
        metrics["NRMSE"] = None

    # --- Métricas de bordes (cross-spectral) ---
    metrics["NCC_Sobel"] = evaluate_ncc_edges(image_warped, thr_post, method="sobel")
    metrics["PSNR_Sobel"] = evaluate_psnr_edges(image_warped, thr_post, method="sobel")
    metrics["SSIM_Sobel"] = evaluate_ssim_edges(image_warped, thr_post, method="sobel")

    metrics["NCC_Canny"] = evaluate_ncc_edges(image_warped, thr_post, method="canny")
    metrics["PSNR_Canny"] = evaluate_psnr_edges(image_warped, thr_post, method="canny")
    metrics["SSIM_Canny"] = evaluate_ssim_edges(image_warped, thr_post, method="canny")

    return metrics

# ======================================
# [FIN METRICAS] 
# ======================================

def filter_and_analyze_matches(matches, scores, threshold=0.75):
    """
    Filters reliable matches based on a score threshold and calculates statistics.

    Parameters:
    - matches: List of match objects.
    - scores: List or tensor of scores corresponding to each match.
    - threshold: Score threshold to determine reliable matches (default is 0.75).

    Returns:
    - reliable_points: List of reliable matches above the threshold.
    - average_score: Average score of all matches.
    - util_percentage_points: Percentage of matches that are reliable.
    """
    # Filter reliable points with a score >= threshold
    reliable_points = [match for match, score in zip(matches, scores) if score.item() >= threshold]

    # Calculate the average score
    average_score = scores.mean().item()
    print(f"The average score is: {average_score}\n")

    print(f"The number of points above a score threshold of {threshold} is: {len(reliable_points)}\n")

    util_percentage_points = (len(reliable_points) * 100) / len(matches)
    print("Therefore, for this registration:")
    print(f"{util_percentage_points:.2f}% of the points are precise")

    return reliable_points, average_score, util_percentage_points

# === [TRANSFORMACIONES] ===
def apply_afin_transformation(feats0, feats1, matches01, imagen0, imagen1, threshold=50):
    """
    Aplica una transformación afín para registrar imagen0 con respecto a imagen1.

    Parámetros:
    - feats0: Características extraídas de imagen0.
    - feats1: Características extraídas de imagen1.
    - matches01: Diccionario con los emparejamientos entre feats0 y feats1.
    - imagen0: Tensor de PyTorch de la primera imagen (C, H, W).
    - imagen1: Tensor de PyTorch de la segunda imagen (C, H, W).
    - threshold: Número mínimo de correspondencias requeridas para proceder.

    Retorna:
    - imagen0_warped: Imagen resultante después de aplicar la transformación afín.
    """
    
    # Extraer los emparejamientos y las puntuaciones
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir los puntos clave a NumPy y asegurarse de que sean float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)
    
    # Verificar si hay suficientes correspondencias
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # Calcular la matriz de transformación afín
    M, inliers = cv2.estimateAffine2D(
        pts0,
        pts1,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=5.0
    )
    
    if M is None:
        raise ValueError("No se pudo calcular la transformación afín.")

    # Convertir tensores a arrays de NumPy y transponer para obtener (H, W, C)
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)
    
    # Asegurarse de que las imágenes sean de tipo uint8
    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)
    
    # Aplicar la transformación afín
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    error = evaluate_homography(M, pts0, pts1)
    return imagen0_warped, points0, scores, error

def apply_homography_transformation(feats0, feats1, matches01, imagen0, imagen1, threshold=50):
    """
    Aplica una transformación de homografía para registrar imagen0 con respecto a imagen1.

    Retorna:
    - imagen0_warped: Imagen resultante después de aplicar la transformación.
    - points0: keypoints en imagen0 usados.
    - scores: puntuaciones de los matches.
    - metrics_local: diccionario con métricas locales y globales.
    """

    matches, scores = matches01["matches"], matches01["scores"]
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]
    
    pts0 = points0.cpu().numpy()
    pts1 = points1.cpu().numpy()

    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # Calcular la matriz de homografía con RANSAC robusto
    M, mask = cv2.findHomography(pts0, pts1, cv2.USAC_MAGSAC, 5.0) 
    
    if M is None:
        raise ValueError("No se pudo calcular la homografía.")

    # Convertir tensor de PyTorch a imagen PIL para la transformación
    imagen0_pil = to_pil_image(imagen0.cpu())
    imagen1_pil = to_pil_image(imagen1.cpu())
    
    # Aplicar la transformación de perspectiva
    imagen0_warped = cv2.warpPerspective(
        np.array(imagen0_pil), 
        M, 
        (imagen1.shape[2], imagen1.shape[1])
    )

    # Llamamos a la función centralizada de métricas
    metrics_local = compute_local_metrics(pts0, pts1, matches, mask, M)

    return imagen0_warped, points0, scores, metrics_local

def apply_similarity_transformation(feats0, feats1, matches01, imagen0, imagen1, threshold=50):
    """
    Aplica una transformación de similaridad para registrar imagen0 con respecto a imagen1.

    Retorna:
    - imagen0_warped: Imagen registrada después de aplicar la transformación.
    - points0: keypoints en imagen0 usados.
    - scores: puntuaciones de los matches.
    - metrics_local: diccionario con métricas locales y globales.
    """

    # Extraer los emparejamientos y las puntuaciones
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir los puntos clave a NumPy float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # Verificar si hay suficientes correspondencias
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # Calcular la matriz de transformación de similaridad
    # estimateAffinePartial2D con fullAffine=False = similaridad
    M, mask = cv2.estimateAffinePartial2D(
        pts0,
        pts1,
        method=cv2.LMEDS,
        ransacReprojThreshold=5.0
    )

    if M is None:
        raise ValueError("No se pudo calcular la transformación de similaridad.")

    # Convertir tensores a arrays de NumPy y transponer (C,H,W) → (H,W,C)
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    # Asegurarse de que las imágenes sean uint8
    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    # Aplicar la transformación de similaridad
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    # Llamamos a la función centralizada de métricas
    metrics_local = compute_local_metrics(pts0, pts1, matches, mask, M)

    return imagen0_warped, points0, scores, metrics_local

def apply_rigid_transformation(feats0, feats1, matches01, imagen0, imagen1, threshold=50):
    """
    Aplica una transformación rígida (rotación + traslación) para registrar imagen0 con respecto a imagen1.

    Retorna:
    - imagen0_warped: Imagen registrada después de aplicar la transformación.
    - points0: keypoints en imagen0 usados.
    - scores: puntuaciones de los matches.
    - metrics_local: diccionario con métricas locales y globales.
    """

    # Extraer los emparejamientos y las puntuaciones
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir los puntos clave a NumPy float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # Verificar si hay suficientes correspondencias
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # Estimar la transformación rígida (rotación + traslación)
    centroid0 = np.mean(pts0, axis=0)
    centroid1 = np.mean(pts1, axis=0)
    pts0_centered = pts0 - centroid0
    pts1_centered = pts1 - centroid1

    H = np.dot(pts0_centered.T, pts1_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Asegurarse de que la matriz de rotación es válida
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Calcular traslación
    t = centroid1.T - np.dot(R, centroid0.T)

    # Construir la matriz de transformación 2x3 para cv2.warpAffine
    M = np.hstack((R, t.reshape(2, 1)))

    # Convertir tensores a imágenes NumPy
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    # Aplicar la transformación rígida
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    # --- Métricas locales y globales ---
    # ⚠️ cv2.estimateAffinePartial2D devuelve máscara, aquí no usamos RANSAC → no hay mask
    mask = None  
    metrics_local = compute_local_metrics(pts0, pts1, matches, mask, M)

    return imagen0_warped, points0, scores, metrics_local

def apply_rigid_transformation_ransac(
    feats0, feats1, matches01, imagen0, imagen1,
    threshold=50, ransac_iterations=1000, ransac_threshold=5.0
):
    """
    Aplica una transformación rígida (rotación + traslación) con RANSAC
    para registrar imagen0 con respecto a imagen1.

    Retorna:
    - imagen0_warped: Imagen registrada después de aplicar la transformación.
    - points0: keypoints en imagen0 usados.
    - scores: puntuaciones de los matches.
    - metrics_local: diccionario con métricas locales y globales.
    """

    # --- Extracción de emparejamientos y puntos ---
    matches = matches01["matches"]
    scores = matches01["scores"]
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # --- Verificación de número mínimo de matches ---
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # --- RANSAC manual para estimar transformación rígida ---
    max_inliers = 0
    best_M = None
    best_inliers = None

    for _ in range(ransac_iterations):
        # Seleccionar aleatoriamente 2 pares de puntos
        idx = np.random.choice(len(pts0), 2, replace=False)
        sample_pts0 = pts0[idx]
        sample_pts1 = pts1[idx]

        # Estimar la transformación rígida con los 2 pares seleccionados
        M_candidate = estimate_rigid_transform(sample_pts0, sample_pts1)

        # Aplicar la transformación a todos los puntos
        pts0_transformed = (np.dot(M_candidate[:, :2], pts0.T) + M_candidate[:, 2:3]).T

        # Calcular error de reproyección
        errors = np.linalg.norm(pts0_transformed - pts1, axis=1)

        # Determinar inliers
        inliers = errors < ransac_threshold
        num_inliers = np.sum(inliers)

        # Actualizar si mejora el modelo
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_M = M_candidate
            best_inliers = inliers

    if best_M is None:
        raise ValueError("No se pudo calcular la transformación rígida con RANSAC.")

    # Recalcular la transformación usando solo los inliers
    M = estimate_rigid_transform(pts0[best_inliers], pts1[best_inliers])

    # --- Crear máscara binaria tipo OpenCV ---
    mask = best_inliers.astype(np.uint8).reshape(-1, 1)

    # --- Aplicar la transformación a la imagen ---
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    # --- Calcular métricas locales y globales ---
    metrics_local = compute_local_metrics(pts0, pts1, matches, mask, M)

    return imagen0_warped, points0, scores, metrics_local

def estimate_rigid_transform(pts0, pts1):
    # Implementación similar a la anterior
    centroid0 = np.mean(pts0, axis=0)
    centroid1 = np.mean(pts1, axis=0)
    pts0_centered = pts0 - centroid0
    pts1_centered = pts1 - centroid1
    H = np.dot(pts0_centered.T, pts1_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid1.T - np.dot(R, centroid0.T)
    M = np.hstack((R, t.reshape(2, 1)))
    return M

def apply_translation_transformation(feats0, feats1, matches01, imagen0, imagen1, threshold=50):
    """
    Aplica una transformación de traslación para registrar imagen0 con respecto a imagen1.

    Retorna:
    - imagen0_warped: Imagen registrada después de aplicar la transformación.
    - points0: keypoints en imagen0 usados.
    - scores: puntuaciones de los matches.
    - metrics_local: diccionario con métricas locales y globales.
    """

    # --- Extracción de emparejamientos ---
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir a NumPy float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # --- Verificación de cantidad mínima de matches ---
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # --- Calcular vector de traslación (usando mediana para robustez) ---
    translations = pts1 - pts0
    t_x = np.median(translations[:, 0])
    t_y = np.median(translations[:, 1])

    # --- Construir matriz de transformación 2x3 ---
    M = np.array([[1, 0, t_x],
                  [0, 1, t_y]], dtype=np.float32)

    # --- Convertir tensores a NumPy (C,H,W) → (H,W,C) ---
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    # Asegurar tipo uint8
    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    # --- Aplicar la traslación ---
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    # --- Calcular métricas locales y globales ---
    # ⚠️ En traslación pura no hay RANSAC, por lo tanto no hay máscara de inliers
    mask = None  
    metrics_local = compute_local_metrics(pts0, pts1, matches, mask, M)

    return imagen0_warped, points0, scores, metrics_local

def apply_translation_transformation_ransac(
    feats0, feats1, matches01, imagen0, imagen1,
    threshold=1, ransacReprojThreshold=5.0
):
    """
    Aplica una transformación de traslación para registrar imagen0 con respecto a imagen1 utilizando RANSAC.

    Retorna:
    - imagen0_warped: Imagen registrada después de aplicar la transformación.
    - points0: keypoints en imagen0 usados.
    - scores: puntuaciones de los matches.
    - metrics_local: diccionario con métricas locales y globales.
    """

    # --- Extracción de emparejamientos ---
    matches = matches01["matches"]
    scores = matches01["scores"]

    # Obtener los puntos clave correspondientes
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # Convertir a NumPy float32
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # --- Verificación mínima de correspondencias ---
    if len(pts0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(pts0)} matches")
        return None

    # --- Ajuste robusto con RANSAC ---
    pts0_reshaped = pts0.reshape(-1, 1, 2)
    pts1_reshaped = pts1.reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffine2D(
        pts0_reshaped,
        pts1_reshaped,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransacReprojThreshold,
        refineIters=10
    )

    if M is None:
        raise ValueError("No se pudo calcular la transformación de traslación.")

    # --- Extraer solo el componente de traslación ---
    t_x, t_y = M[0, 2], M[1, 2]
    M_translation = np.array([[1, 0, t_x],
                              [0, 1, t_y]], dtype=np.float32)

    # --- Convertir tensores a imágenes NumPy ---
    imagen0_np = imagen0.cpu().numpy().transpose(1, 2, 0)
    imagen1_np = imagen1.cpu().numpy().transpose(1, 2, 0)

    # Asegurar formato uint8
    if imagen0_np.dtype != np.uint8:
        imagen0_np = (imagen0_np * 255).astype(np.uint8)
    if imagen1_np.dtype != np.uint8:
        imagen1_np = (imagen1_np * 255).astype(np.uint8)

    # --- Aplicar transformación ---
    imagen0_warped = cv2.warpAffine(
        imagen0_np,
        M_translation,
        (imagen1_np.shape[1], imagen1_np.shape[0])
    )

    # --- Calcular métricas locales y globales ---
    mask = inliers.astype(np.uint8) if inliers is not None else None
    metrics_local = compute_local_metrics(pts0, pts1, matches, mask, M_translation)

    return imagen0_warped, points0, scores, metrics_local

def apply_translation_torch(imagen, t_x, t_y):
    """
    Aplica una traslación pura a una imagen tensorial usando PyTorch.
    Se asume que `imagen` tiene forma (C, H, W).
    """
    # Extraer las dimensiones correctamente
    C, H, W = imagen.shape

    # Crear la matriz de transformación afín (2x3)
    M = torch.tensor(
        [[1, 0, t_x / (W / 2)],  # Normalizar la traslación a [-1, 1]
         [0, 1, t_y / (H / 2)]],
        dtype=torch.float32,
        device=imagen.device
    ).unsqueeze(0)  # Ahora M tiene forma (1, 2, 3)

    # Añadir dimensión de batch a la imagen
    imagen_4d = imagen.unsqueeze(0)  # (1, C, H, W)

    # Crear la cuadrícula para la transformación
    grid = F.affine_grid(M, imagen_4d.size(), align_corners=False)

    # Aplicar la transformación
    warped = F.grid_sample(imagen_4d, grid, align_corners=False)

    # Remover la dimensión de batch antes de retornar
    return warped.squeeze(0)

def apply_translation_torch(imagen, t_x, t_y):
    """
    Aplica una traslación pura a una imagen tensorial usando PyTorch.
    Se asume que `imagen` tiene forma (C, H, W).

    Parámetros:
    - imagen (torch.Tensor): Tensor de la imagen de forma (C,H,W).
    - t_x, t_y (float): Traslaciones en píxeles.

    Retorna:
    - warped (torch.Tensor): Imagen traducida.
    - H (np.ndarray): Matriz homográfica 3x3 en coordenadas de píxeles para usar con evaluate_homography.
    """
    C, H, W = imagen.shape

    # Crear la matriz de transformación afín (2x3) normalizada para grid_sample
    M_normalized = torch.tensor(
        [[1, 0, t_x / (W / 2)],  # Normalizar la traslación a [-1, 1]
         [0, 1, t_y / (H / 2)]],
        dtype=torch.float32,
        device=imagen.device
    )

    # Aplicar la transformación con PyTorch
    M_u = M_normalized.unsqueeze(0)  # (1, 2, 3)
    imagen_4d = imagen.unsqueeze(0)  # (1, C, H, W)
    grid = F.affine_grid(M_u, imagen_4d.size(), align_corners=False)
    warped = F.grid_sample(imagen_4d, grid, align_corners=False).squeeze(0)

    # Convertir la matriz normalizada a una homografía 3x3 en coordenadas de píxeles

    # Extraer traslaciones normalizadas
    t_x_norm = M_normalized[0, 2].item()
    t_y_norm = M_normalized[1, 2].item()

    # Convertir traslaciones normalizadas a píxeles
    pixel_t_x = t_x_norm * (W / 2)
    pixel_t_y = t_y_norm * (H / 2)

    # Construir la matriz homográfica 3x3 en píxeles
    H = np.array([
        [1, 0, pixel_t_x],
        [0, 1, pixel_t_y],
        [0, 0,       1.0]
    ], dtype=np.float32)

    return warped, H

def apply_translation_transformation2(feats0, feats1, matches01, imagen0, imagen1, threshold=50, umbral_puntuacion=0):
    """
    Aplica una transformación de traslación para registrar imagen0 con respecto a imagen1 usando PyTorch.

    Retorna:
    - imagen0_warped: Imagen registrada después de aplicar la transformación.
    - points0: keypoints en imagen0 usados.
    - scores: puntuaciones de los matches.
    - metrics_local: diccionario con métricas locales y globales.
    """

    # --- Extraer emparejamientos y puntuaciones ---
    matches = matches01["matches"]
    scores = matches01["scores"]

    # --- Obtener los puntos clave correspondientes ---
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]

    # --- Filtrar por puntuación ---
    valid_matches = scores > umbral_puntuacion
    points0 = points0[valid_matches]
    points1 = points1[valid_matches]
    matches = matches[valid_matches]
    scores = scores[valid_matches]

    # --- Verificación de número mínimo de matches válidos ---
    if len(points0) < threshold:
        print(f"La imagen no es lo suficientemente precisa, solo posee {len(points0)} matches válidos.")
        return None

    # --- Convertir a NumPy ---
    pts0 = points0.cpu().numpy().astype(np.float32)
    pts1 = points1.cpu().numpy().astype(np.float32)

    # --- Calcular vector de traslación robusto (mediana) ---
    translations = pts1 - pts0
    t_x = np.median(translations[:, 0])
    t_y = np.median(translations[:, 1])

    # --- Aplicar traslación con PyTorch ---
    imagen0_warped, M = apply_translation_torch(imagen0, t_x, t_y)

    # --- Métricas locales y globales ---
    mask = None  # ⚠️ No hay RANSAC, así que no existe máscara de inliers
    metrics_local = compute_local_metrics(pts0, pts1, matches, mask, M)

    return imagen0_warped, points0, scores, metrics_local


# === [FIN TRANSFORMACIONES] ===
def save_metrics_to_excel(xlsx_path, data, fieldnames, decimals=6):
    # Redondear y formatear los valores
    formatted_data = {}
    for k, v in data.items():
        if isinstance(v, (float, int)):
            formatted_data[k] = round(v, decimals)
        else:
            formatted_data[k] = v

    df = pd.DataFrame([[formatted_data.get(col, None) for col in fieldnames]], columns=fieldnames)

    # Si existe → append, si no → crear nuevo
    if os.path.exists(xlsx_path):
        book = pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="overlay")
        startrow = book.sheets['Sheet1'].max_row
        df.to_excel(book, sheet_name="Sheet1", index=False, header=False, startrow=startrow)
        book.close()
    else:
        # Aquí aplicamos estilo directamente
        styled = df.style.set_table_styles(
            [
                {"selector": "th", "props": "background-color: #4CAF50; color: white; text-align: center;"},
                {"selector": "td", "props": "text-align: center;"}
            ]
        ).format(precision=decimals)

        styled.to_excel(xlsx_path, index=False, sheet_name="Sheet1", engine="openpyxl")


