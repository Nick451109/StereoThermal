import PIL
import torch
from Image_registration.lightglue import (
    LightGlue,
    SuperPoint,
    DISK,
    SIFT,
    ALIKED,
    DoGHardNet,
    viz2d,
)
from Image_registration.lightglue.utils import load_image, rbd
from torchvision.transforms.functional import resize, to_pil_image, to_tensor
from torchvision.utils import save_image
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from Image_registration.util import *
import seaborn as sns
import glob


from skimage.metrics import structural_similarity as ssim

def _read_as_bgr(path):

    '''
    Garantiza que la imagen de entrada siempre esté en formato BGR de 3 canales
    incluso si el archivo original está en escala de grises o tiene canal alfa (BGRA).

    Se emplea ya que estos métodos (cv2.circle, cv2.line) de openCV esperan trabajar 
    sobre imágenes BGR de 3 canales.

    '''

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    # Si es 1 canal → BGR para que dibujar sea fácil
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # BGRA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# === [KEYPOINTS] Helpers para unir puntos a partir de matches ===
def _pts_from_matches(feats0, feats1, matches01):
    kpts0 = feats0["keypoints"].detach().cpu().numpy()
    kpts1 = feats1["keypoints"].detach().cpu().numpy()
    matches = matches01["matches"].detach().cpu().numpy()
    if len(matches) == 0:
        return np.empty((0,2)), np.empty((0,2)), matches
    pts0 = kpts0[matches[:, 0]]
    pts1 = kpts1[matches[:, 1]]
    return pts0, pts1, matches

def compute_inliers_homography(feats0, feats1, matches01, ransac_thresh=5.0):
    pts0, pts1, matches = _pts_from_matches(feats0, feats1, matches01)
    if len(matches) < 4:
        return np.array([], dtype=int)
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return np.array([], dtype=int)
    return np.where(mask.ravel() == 1)[0]

def compute_inliers_affine_ransac(feats0, feats1, matches01, ransac_thresh=3.0, max_iters=2000):
    pts0, pts1, matches = _pts_from_matches(feats0, feats1, matches01)
    if len(matches) < 3:
        return np.array([], dtype=int)
    M, inliers = cv2.estimateAffine2D(pts0, pts1, method=cv2.RANSAC,
                                      ransacReprojThreshold=ransac_thresh,
                                      maxIters=max_iters, refineIters=10)
    if M is None or inliers is None:
        return np.array([], dtype=int)
    return np.where(inliers.ravel() == 1)[0]

def compute_inliers_similarity_ransac(feats0, feats1, matches01, ransac_thresh=3.0, max_iters=2000):
    """
    Similarity (rot+trans+escala uniforme) ≈ estimateAffinePartial2D con RANSAC.
    """
    pts0, pts1, matches = _pts_from_matches(feats0, feats1, matches01)
    if len(matches) < 2:
        return np.array([], dtype=int)
    M, inliers = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_thresh,
                                             maxIters=max_iters, refineIters=10)
    if M is None or inliers is None:
        return np.array([], dtype=int)
    return np.where(inliers.ravel() == 1)[0]

def compute_inliers_rigid_ransac(feats0, feats1, matches01, ransac_thresh=3.0, max_iters=2000):
    """
    Rigid (rot+trans, sin escala). Aproximamos usando estimateAffinePartial2D y
    luego validamos con el mismo mask. Para visualización de inliers sirve igual.
    """
    return compute_inliers_similarity_ransac(feats0, feats1, matches01,
                                             ransac_thresh=ransac_thresh,
                                             max_iters=max_iters)

def compute_inliers_translation_ransac(feats0, feats1, matches01, ransac_thresh=2.0, max_iters=2000):
    """
    RANSAC para traslación pura:
    - Hipótesis (dx, dy) a partir de un match.
    - Residual = || (p0 + [dx,dy]) - p1 ||.
    """
    pts0, pts1, matches = _pts_from_matches(feats0, feats1, matches01)
    n = len(matches)
    if n < 1:
        return np.array([], dtype=int)
    best_inliers = []
    rng = np.random.default_rng(0)
    for _ in range(max_iters):
        i = rng.integers(0, n)
        dx, dy = (pts1[i] - pts0[i])
        pred = pts0 + np.array([dx, dy])
        residuals = np.linalg.norm(pred - pts1, axis=1)
        inliers = np.where(residuals <= ransac_thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            # criterio de parada temprano
            if len(best_inliers) > 0.9 * n:
                break
    return np.array(best_inliers, dtype=int)

# === [KEYPOINTS] FIN ===

def visualize_matches(
    ruta_imagen0: str,
    ruta_imagen1: str,
    feats0: dict,
    feats1: dict,
    matches01: dict,
    inliers_idx: np.ndarray | None = None,
    max_lines: int | None = 600,
    color=(0, 255, 0),
    thickness: int = 1,
    radius: int = 2,
    alpha: float = 0.6,
    save_path: str | None = None,
    show: bool = False,
):
    """
    
    Dibuja matches con INLIERS en VERDE y OUTLIERS en ROJO, y añade conteos.
    - feats0/feats1: dict con 'keypoints' (N,2) en píxeles.
    - matches01['matches']: (M,2) con índices (i0, i1).
    - inliers_idx: índices de matches considerados inliers (opcional).

    OJO
    Si existen muchos matches, se puede limitar a, por ejemplo, 600.
    
    """
    img0 = _read_as_bgr(ruta_imagen0)
    img1 = _read_as_bgr(ruta_imagen1)
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)
    W = w0 + w1

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0 + w1] = img1
    overlay = canvas.copy()

    kpts0 = feats0["keypoints"].detach().cpu().numpy()
    kpts1 = feats1["keypoints"].detach().cpu().numpy()
    matches = matches01["matches"].detach().cpu().numpy()

    if max_lines is not None and len(matches) > max_lines:
        idx = np.linspace(0, len(matches) - 1, max_lines).astype(int)
        matches = matches[idx]
        if inliers_idx is not None:
            # Reindexar inliers_idx a la submuestra
            orig_to_sub = {orig: i for i, orig in enumerate(idx)}
            inliers_idx = np.array([orig_to_sub[i] for i in inliers_idx if i in orig_to_sub], dtype=int)

    if inliers_idx is None:
        inliers_idx = np.array([], dtype=int)

    all_idx = np.arange(len(matches))
    outliers_idx = np.setdiff1d(all_idx, inliers_idx)

    red = (0, 0, 255)
    green = (0, 255, 0)

    # OUTLIERS primero
    for j in outliers_idx:
        i0, i1 = matches[j]
        x0, y0 = kpts0[i0]
        x1, y1 = kpts1[i1]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1 + w0)), int(round(y1)))
        cv2.circle(overlay, p0, radius, red, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, p1, radius, red, -1, lineType=cv2.LINE_AA)
        cv2.line(overlay, p0, p1, red, thickness, lineType=cv2.LINE_AA)

    # INLIERS
    for j in inliers_idx:
        i0, i1 = matches[j]
        x0, y0 = kpts0[i0]
        x1, y1 = kpts1[i1]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1 + w0)), int(round(y1)))
        cv2.circle(overlay, p0, radius, green, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, p1, radius, green, -1, lineType=cv2.LINE_AA)
        cv2.line(overlay, p0, p1, green, thickness, lineType=cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    # Conteos
    total = len(matches)
    num_in = len(inliers_idx)
    num_out = total - num_in
    def put(text, org, color):
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    put(f"Total matches: {total}", (10, 25), (0,0,0))
    put(f"Inliers: {num_in}",       (10, 50), (0,255,0))
    put(f"Outliers: {num_out}",     (10, 75), (0,0,255))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis)

    if show:
        plt.figure(figsize=(14, 6))
        plt.axis("off")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

    return vis

def procesar_imagenes(
    ruta_imagen0,
    ruta_imagen1,
    extractor_tipo="superpoint",
    max_num_keypoints=8192,
    dispositivo=None,
    filtro_imagen0=None,  # Nuevo parámetro para aplicar filtros a imagen0
    filtro_imagen1=None,  # Nuevo parámetro para aplicar filtros a imagen1
    threshold=0,
    transformation_method="homography",
    visualize=False,  # Nuevo parámetro para activar/desactivar la visualización de keypoints
    visualize_save_path=None,  # Ruta para guardar la visualización de keypoints (opcional)
):
    """
    Procesa dos imágenes para obtener una imagen resultante tras aplicar la transformación de perspectiva.

    Args:
        ruta_imagen0 (str): Ruta al primer archivo de imagen (To be registered).
        ruta_imagen1 (str): Ruta al segundo archivo de imagen (Fixed).
        extractor_tipo (str, opcional): Tipo de extractor a utilizar. Opciones: "superpoint", "disk", "sift", "aliked", "doghardnet". Por defecto es "superpoint".
        max_num_keypoints (int, opcional): Número máximo de puntos clave a extraer. Por defecto es 4096.
        dispositivo (torch.device, opcional): Dispositivo para ejecutar el modelo. Si es None, se selecciona automáticamente GPU si está disponible.
        filtro_imagen0 (str, opcional): Tipo de filtro a aplicar a imagen0. Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none". Por defecto es None.
        filtro_imagen1 (str, opcional): Tipo de filtro a aplicar a imagen1. Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none". Por defecto es None.

    Returns:
        PIL.Image: Imagen resultante tras la transformación de perspectiva.
        Actualmente se regresa una imagen tipo numpy array en BGR
    """
    print("[INFO] Procesando imagenes..")
    print("[INFO] Método de transformación:", transformation_method)
    # Configurar el dispositivo
    if dispositivo is None:
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE] Dispositivo utilizado:", dispositivo)

    # Inicializar extractor y matcher según el tipo especificado
    extractor_dict = {
        "superpoint": (
            SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(dispositivo),
            LightGlue(features="superpoint").eval().to(dispositivo),
        ),
        "disk": (
            DISK(max_num_keypoints=max_num_keypoints).eval().to(dispositivo),
            LightGlue(features="disk").eval().to(dispositivo),
        ),
        "sift": (
            SIFT(max_num_keypoints=max_num_keypoints).eval().to(dispositivo),
            LightGlue(features="sift").eval().to(dispositivo),
        ),
        "aliked": (
            ALIKED(max_num_keypoints=max_num_keypoints).eval().to(dispositivo),
            LightGlue(features="aliked").eval().to(dispositivo),
        ),
        "doghardnet": (
            DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(dispositivo),
            LightGlue(features="doghardnet").eval().to(dispositivo),
        ),
    }

    if extractor_tipo.lower() not in extractor_dict:
        raise ValueError(
            f"Extractor tipo '{extractor_tipo}' no es válido. Selecciona entre 'superpoint', 'disk', 'sift', 'aliked', 'doghardnet'."
        )

    extractor, matcher = extractor_dict[extractor_tipo.lower()]
    print(f"[INFO] Extractor: {extractor.__class__.__name__} - Matcher: {matcher.__class__.__name__}")

    try:
        # Cargar y preparar las imágenes
        imagen0 = load_image(ruta_imagen0).to(dispositivo)
        imagen1 = load_image(ruta_imagen1).to(dispositivo)

        # Aplicar los filtros a cada imagen si se especifica
        imagen0 = apply_filter(imagen0, filtro_imagen0)
        imagen1 = apply_filter(imagen1, filtro_imagen1)

        # Extraer características
        feats0 = extractor.extract(imagen0)
        feats1 = extractor.extract(imagen1)

        # Emparejar características
        matches01 = matcher({"image0": feats0, "image1": feats1})
        #print(f"[INFO] Matches: {len(matches01['matches'])}")
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        #print(f"[INFO] Matches después de rbd: {len(matches01['matches'])}")

        # # Obtener los puntos clave y las correspondencias del primer conjunto
        # kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        # m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
        # === [VISUALIZADOR] Visualizar matches si está activado ===
        if visualize:
            try:
                # Elegir el cálculo de inliers acorde al modelo:
                tm = transformation_method.lower()
                if tm == "homography":
                    inliers_idx = compute_inliers_homography(feats0, feats1, matches01, ransac_thresh=5.0)
                elif tm == "affine":
                    inliers_idx = compute_inliers_affine_ransac(feats0, feats1, matches01, ransac_thresh=3.0)
                elif tm == "rigid" or tm == "similarity":
                    inliers_idx = compute_inliers_similarity_ransac(feats0, feats1, matches01, ransac_thresh=3.0)
                elif tm == "translation" or tm == "translation2":
                    inliers_idx = compute_inliers_translation_ransac(feats0, feats1, matches01, ransac_thresh=2.0)
                elif tm == "rigid_ransac":
                    inliers_idx = compute_inliers_rigid_ransac(feats0, feats1, matches01, ransac_thresh=3.0)
                elif tm == "translation_ransac":
                    inliers_idx = compute_inliers_translation_ransac(feats0, feats1, matches01, ransac_thresh=2.0)
                else:
                    inliers_idx = None  # fallback

                visualize_matches(
                    ruta_imagen0=ruta_imagen0,
                    ruta_imagen1=ruta_imagen1,
                    feats0=feats0,
                    feats1=feats1,
                    matches01=matches01,
                    inliers_idx=inliers_idx,      # Separación por colores inliers/outliers
                    save_path=visualize_save_path,
                    show=False
                )
                if visualize_save_path:
                    print(f"[INFO] Imagen de inliers/outliers guardada en: {visualize_save_path}")
            except Exception as e:
                print(f"[WARN] No se pudo generar visualización de keypoints: {e}")

        if transformation_method == "homography":
            imagen0_warped, points0, scores, metrics_local = apply_homography_transformation(
                feats0, feats1, matches01, imagen0, imagen1, threshold=threshold
            )

        elif transformation_method == "affine":
            imagen0_warped, points0, scores, metrics_local = apply_afin_transformation(
                feats0, feats1, matches01, imagen0, imagen1, threshold=threshold
            )
        elif transformation_method == "rigid":
            imagen0_warped, points0, scores, metrics_local = apply_rigid_transformation(
                feats0, feats1, matches01, imagen0, imagen1, threshold=threshold
            )
        elif transformation_method == "similarity":
            imagen0_warped, points0, scores, metrics_local = apply_similarity_transformation(
                feats0, feats1, matches01, imagen0, imagen1, threshold=threshold
            )
        elif transformation_method == "translation":
            imagen0_warped, points0, scores, metrics_local = apply_translation_transformation(
                feats0, feats1, matches01, imagen0, imagen1, threshold=threshold
            )
        elif transformation_method == "translation2":
            imagen0_warped, points0, scores, metrics_local = apply_translation_transformation2(
                feats0, feats1, matches01, imagen0, imagen1, threshold=threshold
            )
            imagen0_warped = np.array(to_pil_image(imagen0_warped.cpu()))
        elif transformation_method == "rigid_ransac":
            imagen0_warped, points0, scores, metrics_local = apply_rigid_transformation_ransac(
                feats0, feats1, matches01, imagen0, imagen1, threshold=threshold
            )
        elif transformation_method == "translation_ransac":
            imagen0_warped, points0, scores, metrics_local = (
                apply_translation_transformation_ransac(
                    feats0, feats1, matches01, imagen0, imagen1, threshold=threshold
                )
            )

        # imagen0_warped_pil = Image.fromarray(imagen0_warped)

        # # Convertir la imagen warpada a tensor
        # imagen0_warped_tensor = to_tensor(imagen0_warped_pil).to(dispositivo)

        # # Redimensionar la imagen warpada si es necesario
        # imagen0_warped = resize(imagen0_warped, (480, 640))

        # # Convertir a PIL para retornar
        # resultado_pil = to_pil_image(imagen0_warped_tensor.cpu())

        print(
            f"[INFO] shape: {imagen0_warped.shape} - Matches: {len(points0)} - Score: {len(scores)}"
        )
        return imagen0_warped, points0, scores, metrics_local

    except Exception as e:
        print(f"Error al procesar las imágenes: {e}")
        return None, None, None, None

def mostrar_correspondencias_mejorada(
    ruta_imagen0,
    ruta_imagen1,
    extractor_tipo="superpoint",
    max_num_keypoints=8192,
    dispositivo=None,
    umbral_score=0.5,
    guardar_figura=False,
    ruta_guardado="correspondencias_mejoradas.png",
    filtro_imagen0=None,  # Nuevo parámetro para aplicar filtros a imagen0
    filtro_imagen1=None,  # Nuevo parámetro para aplicar filtros a imagen1
    mostrar_correspondencias=True,  # Nuevo parámetro para activar/desactivar las líneas de correspondencia
    mostrar_metricas=True,  # Nuevo parámetro para ocultar/mostrar las métricas
):
    """
    Muestra la correspondencia de los puntos clave entre dos imágenes después de aplicar una transformación de perspectiva.
    Además, distingue visualmente entre correspondencias inliers y outliers y reporta métricas de evaluación.

    Args:
        ruta_imagen0 (str): Ruta al primer archivo de imagen.
        ruta_imagen1 (str): Ruta al segundo archivo de imagen.
        extractor_tipo (str, opcional): Tipo de extractor a utilizar. Opciones: "superpoint", "disk", "sift", "aliked", "doghardnet". Por defecto es "superpoint".
        max_num_keypoints (int, opcional): Número máximo de puntos clave a extraer. Por defecto es 4096.
        dispositivo (torch.device, opcional): Dispositivo para ejecutar el modelo. Si es None, se selecciona automáticamente GPU si está disponible.
        umbral_score (float, opcional): Umbral de score para filtrar las correspondencias antes de visualizarlas. Valores entre 0 y 1. Por defecto es 0.5.
        guardar_figura (bool, opcional): Si se establece en True, guarda la figura en la ruta especificada. Por defecto es False.
        ruta_guardado (str, opcional): Ruta donde se guardará la figura si `guardar_figura` es True. Por defecto es "correspondencias_mejoradas.png".
        filtro_imagen0 (str, opcional): Tipo de filtro a aplicar a imagen0. Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none". Por defecto es None.
        filtro_imagen1 (str, opcional): Tipo de filtro a aplicar a imagen1. Opciones: "grayscale", "hsv", "blur", "contrast", "canny", "none". Por defecto es None.
        mostrar_correspondencias (bool, opcional): Si se establece en True, dibuja las líneas de correspondencia entre keypoints. Por defecto es True.
        mostrar_metricas (bool, opcional): Si se establece en True, muestra las métricas de evaluación en la figura y la consola. Por defecto es True.

    Returns:
        None. Muestra una figura con las correspondencias entre los puntos clave de las dos imágenes, diferenciando inliers y outliers y mostrando métricas de evaluación si está habilitado.
    """

    # Configurar el dispositivo
    if dispositivo is None:
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo utilizado:", dispositivo)

    # Inicializar extractor y matcher según el tipo especificado
    extractor_dict = {
        "superpoint": (
            SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(dispositivo),
            LightGlue(features="superpoint").eval().to(dispositivo),
        ),
        "disk": (
            DISK(max_num_keypoints=max_num_keypoints).eval().to(dispositivo),
            LightGlue(features="disk").eval().to(dispositivo),
        ),
        "sift": (
            SIFT(max_num_keypoints=max_num_keypoints).eval().to(dispositivo),
            LightGlue(features="sift").eval().to(dispositivo),
        ),
        "aliked": (
            ALIKED(max_num_keypoints=max_num_keypoints).eval().to(dispositivo),
            LightGlue(features="aliked").eval().to(dispositivo),
        ),
        "doghardnet": (
            DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(dispositivo),
            LightGlue(features="doghardnet").eval().to(dispositivo),
        ),
    }

    if extractor_tipo.lower() not in extractor_dict:
        raise ValueError(
            f"Extractor tipo '{extractor_tipo}' no es válido. Selecciona entre 'superpoint', 'disk', 'sift', 'aliked', 'doghardnet'."
        )

    extractor, matcher = extractor_dict[extractor_tipo.lower()]

    try:
        # Verificar que las rutas sean cadenas de texto
        if not isinstance(ruta_imagen0, (str, os.PathLike)) or not isinstance(
            ruta_imagen1, (str, os.PathLike)
        ):
            raise TypeError(
                "Las rutas de las imágenes deben ser cadenas de texto, bytes o objetos Path."
            )

        # Verificar que las rutas existan
        if not os.path.exists(ruta_imagen0):
            raise FileNotFoundError(f"La ruta de la imagen0 no existe: {ruta_imagen0}")
        if not os.path.exists(ruta_imagen1):
            raise FileNotFoundError(f"La ruta de la imagen1 no existe: {ruta_imagen1}")

        # Cargar y preparar las imágenes
        imagen0 = load_image(ruta_imagen0).to(dispositivo)
        imagen1 = load_image(ruta_imagen1).to(dispositivo)

        # Aplicar los filtros a cada imagen si se especifica
        imagen0 = apply_filter(imagen0, filtro_imagen0)
        imagen1 = apply_filter(imagen1, filtro_imagen1)

        # Extraer características del primer conjunto de imágenes
        feats0 = extractor.extract(imagen0)
        feats1 = extractor.extract(imagen1)

        # Emparejar características del primer conjunto
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        # Obtener los puntos clave y las correspondencias del primer conjunto
        kpts0, kpts1, matches = (
            feats0["keypoints"],
            feats1["keypoints"],
            matches01["matches"],
        )
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        # Filtrar las correspondencias por score
        scores = matches01["scores"]
        indices_filtrados = scores >= umbral_score
        m_kpts0_filtrados = m_kpts0[indices_filtrados]
        m_kpts1_filtrados = m_kpts1[indices_filtrados]
        scores_filtrados = scores[indices_filtrados]

        # Convertir a numpy para calcular la homografía
        pts0 = m_kpts0_filtrados.cpu().numpy()
        pts1 = m_kpts1_filtrados.cpu().numpy()

        # Calcular la matriz de homografía con RANSAC y obtener la máscara de inliers
        M, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)

        if M is None:
            raise ValueError("No se pudo calcular la homografía.")

        # Convertir tensor de PyTorch a imagen PIL para la transformación
        imagen0_pil = to_pil_image(imagen0.cpu())
        imagen1_pil = to_pil_image(imagen1.cpu())

        # Aplicar la transformación de perspectiva
        imagen0_warped = cv2.warpPerspective(
            np.array(imagen0_pil), M, (imagen1.shape[2], imagen1.shape[1])
        )
        imagen0_warped_pil = Image.fromarray(imagen0_warped)

        # Convertir la imagen warpada a tensor
        imagen0_warped_tensor = (
            torch.tensor(np.array(imagen0_warped_pil)).permute(2, 0, 1).float() / 255.0
        )
        imagen0_warped_tensor = imagen0_warped_tensor.to(dispositivo)

        # Extraer características de la imagen warpada y de la segunda imagen
        feats0_warped = extractor.extract(imagen0_warped_tensor)
        feats1_warped = extractor.extract(imagen1)

        # Emparejar características nuevamente
        matches01_warped = matcher({"image0": feats0_warped, "image1": feats1_warped})
        feats0_warped, feats1_warped, matches01_warped = [
            rbd(x) for x in [feats0_warped, feats1_warped, matches01_warped]
        ]

        # Obtener los puntos clave y las correspondencias del segundo conjunto
        kpts0_warped, kpts1_warped, matches_warped = (
            feats0_warped["keypoints"],
            feats1_warped["keypoints"],
            matches01_warped["matches"],
        )
        m_kpts0_warped, m_kpts1_warped = (
            kpts0_warped[matches_warped[..., 0]],
            kpts1_warped[matches_warped[..., 1]],
        )

        # Filtrar las correspondencias por score del segundo conjunto
        scores_warped = matches01_warped["scores"]
        indices_filtrados_warped = scores_warped >= umbral_score
        m_kpts0_warped_filtrados = m_kpts0_warped[indices_filtrados_warped]
        m_kpts1_warped_filtrados = m_kpts1_warped[indices_filtrados_warped]
        scores_filtrados_warped = scores_warped[indices_filtrados_warped]

        # Crear una imagen combinada horizontalmente
        imagen0_warped_np = imagen0_warped_tensor.cpu().permute(1, 2, 0).numpy()
        imagen1_np = imagen1.cpu().permute(1, 2, 0).numpy()
        combined_image = np.hstack((imagen0_warped_np, imagen1_np))

        # Calcular métricas para el primer conjunto de correspondencias
        total_matches = len(m_kpts0_filtrados)
        inliers = mask.sum()
        outliers = total_matches - inliers
        inlier_ratio = inliers / total_matches if total_matches > 0 else 0
        distancia_media = calcular_distancia_media(pts0, pts1)
        error_reproyeccion = evaluate_homography(M, pts0, pts1)

        # Dibujar las correspondencias del segundo conjunto (post warping)
        if mostrar_correspondencias:
            for i in range(len(m_kpts0_warped_filtrados)):
                # Determinar el color basado en el score
                score_tmp = scores_filtrados_warped[i].item()
                if score_tmp >= 0.90:
                    color = "g"  # Verde
                elif 0.75 <= score_tmp < 0.90:
                    color = "#FFFF00"  # Amarillo
                elif 0.60 <= score_tmp < 0.75:
                    color = "#FFA500"  # Naranja
                elif 0.50 <= score_tmp < 0.60:
                    color = "r"  # Rojo
                else:
                    continue  # Ignorar correspondencias con score < 0.50

                # Obtener las coordenadas en la imagen combinada
                x0_warped, y0_warped = m_kpts0_warped_filtrados[i].cpu().numpy()
                x1_warped, y1_warped = m_kpts1_warped_filtrados[i].cpu().numpy()
                x1_shifted_warped = (
                    x1_warped + imagen0_warped_np.shape[1]
                )  # Ajustar la posición horizontal de la segunda imagen

                # Dibujar línea entre los puntos correspondientes
                plt.plot(
                    [x0_warped, x1_shifted_warped],
                    [y0_warped, y1_warped],
                    c=color,
                    linewidth=0.5,
                )

                if score_tmp >= 0.50:
                    # Dibujar puntos en la imagen combinada
                    plt.scatter(x0_warped, y0_warped, c="b", s=10)
                    plt.scatter(x1_shifted_warped, y1_warped, c="b", s=10)

        # Añadir texto con métricas si está habilitado
        if mostrar_metricas:
            plt.text(
                10,
                30,
                f"Total Matches: {total_matches}",
                color="white",
                fontsize=12,
                bbox=dict(facecolor="black", alpha=0.5),
            )
            plt.text(
                10,
                50,
                f"Inliers: {int(inliers)}",
                color="green",
                fontsize=12,
                bbox=dict(facecolor="black", alpha=0.5),
            )
            plt.text(
                10,
                70,
                f"Outliers: {int(outliers)}",
                color="red",
                fontsize=12,
                bbox=dict(facecolor="black", alpha=0.5),
            )
            plt.text(
                10,
                90,
                f"Inlier Ratio: {inlier_ratio:.2f}",
                color="yellow",
                fontsize=12,
                bbox=dict(facecolor="black", alpha=0.5),
            )
            plt.text(
                10,
                110,
                f"Mean Euclidean Distance: {distancia_media:.2f} pixels",
                color="white",
                fontsize=12,
                bbox=dict(facecolor="black", alpha=0.5),
            )
            plt.text(
                10,
                130,
                f"Mean Reprojection Error: {error_reproyeccion:.2f} pixels",
                color="white",
                fontsize=12,
                bbox=dict(facecolor="black", alpha=0.5),
            )

        plt.axis("off")
        plt.title(
            "Correspondencias entre keypoints después de la transformación de perspectiva",
            fontsize=20,
            fontweight="bold",
        )

        if guardar_figura:
            # Asegurar que el directorio existe
            ruta_directorio = os.path.dirname(ruta_guardado)
            if ruta_directorio:
                os.makedirs(ruta_directorio, exist_ok=True)
            plt.savefig(ruta_guardado, bbox_inches="tight", pad_inches=0)
            print(f"Figura guardada en {ruta_guardado}")

        plt.show()
        plt.close()

        if mostrar_metricas:
            # Mostrar histograma de scores
            plt.figure(figsize=(10, 6))
            sns.histplot(
                scores_filtrados_warped.cpu().detach().numpy(), bins=50, kde=True
            )
            plt.title("Distribución de Scores de Correspondencia (Post Warping)")
            plt.xlabel("Score")
            plt.ylabel("Frecuencia")
            plt.show()

            # Mostrar métricas en la consola
            print(f"Total Matches: {total_matches}")
            print(f"Inliers: {inliers}")
            print(f"Outliers: {outliers}")
            print(f"Inlier Ratio: {inlier_ratio:.2f}")
            print(f"Mean Euclidean Distance: {distancia_media:.2f} pixels")
            print(f"Mean Reprojection Error: {error_reproyeccion:.2f} pixels")

    except Exception as e:
        print(f"Error al procesar y visualizar las correspondencias: {e}")