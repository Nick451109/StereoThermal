from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from scipy import ndimage


def invert_thermal():
    # Directorio de entrada (imágenes originales)
    input_dir  = "captures/temp/therm/"
    # Directorio de salida (imágenes invertidas)
    output_dir = "captures/temp/invert/"

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Listar todos los archivos en el directorio de entrada
    for filename in os.listdir(input_dir):
        # Verificar si el archivo es una imagen (puedes agregar más extensiones si es necesario)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Ruta completa de la imagen de entrada
            input_path = os.path.join(input_dir, filename)

            # Leer la imagen en escala de grises
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # Invertir la imagen (negativo)
                inverted_img = 255 - img

                # Ruta completa de la imagen de salida
                output_path = os.path.join(output_dir, filename)

                # Guardar la imagen invertida
                cv2.imwrite(output_path, inverted_img)
                print(f"Imagen procesada: {filename}")
            else:
                print(f"Error al leer la imagen: {filename}")

    print(f"Proceso completado! Transformadas {len(os.listdir(input_dir))} imágenes.")

def rgb2grayscale(ruta: str, ruta_salida: str):

    # Cargar la imagen
    imagen = cv2.imread(ruta)

    # if imagen is None:
    #     print("Error al cargar la imagen.")
    # else:
    #     # Crea una ventana y muestra la imagen
    #     cv2.imshow('Imagen', imagen)

    #     # Espera a que se presione una tecla
    #     cv2.waitKey(0)

    #     # Cierra la ventana
    #     cv2.destroyAllWindows()

    # Convertir a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(ruta_salida, imagen_gris)

def rgb2gray_all():
    # Directorio de entrada (imágenes originales)
    input_dir = "captures/temp/50right/"
    # Directorio de salida (imágenes en escala de grises)
    output_dir = "captures/temp/50gray/"

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Listar todos los archivos en el directorio de entrada
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            img = cv2.imread(input_path)

            if img is not None:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, gray_img)
                print(f"Imagen convertida a escala de grises: {filename}")
            else:
                print(f"Error al leer la imagen: {filename}")


def create_anaglyph():
    """
    Creates an anaglyph image from left and right eye images.
    """

    left_image_path  = "output_rectified/color1_rect/color1_000.png"
    right_image_path = "output_rectified/color2_rect/color2_000.png"

    left_img = Image.open(left_image_path).convert("RGB")
    right_img = Image.open(right_image_path).convert("RGB")

    # Ensure images are the same size
    if left_img.size != right_img.size:
        raise ValueError("Left and right images must have the same dimensions.")

    # Split channels
    left_red, _, _ = left_img.split()
    _, right_green, right_blue = right_img.split()

    # Create anaglyph image by combining channels
    anaglyph_img = Image.merge("RGB", (left_red, right_green, right_blue))
    plt.imshow(anaglyph_img)
    plt.title("Anaglyph 3D Image")
    plt.axis('off') # Hide axes for cleaner display
    plt.show()

def process_thermal_images(input_dir, output_dir, method='custom'):
    """
    Procesa todas las imágenes térmicas en un directorio
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # Aplicar mejoras
            enhanced_images = enhance_thermal_image(img)

            # Guardar la imagen procesada según el método elegido
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, enhanced_images[method])

            print(f"Procesada: {filename}")

def apply_convolution_kernels(img, kernel_type='edge_enhance', kernel_size=3, strength=1.0):
    """
    Aplica diferentes kernels convolucionales para mejorar contraste y detectar líneas
    """
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_normalized *= 255
    img_normalized = img_normalized.astype(np.uint8)

    # Definir diferentes kernels
    kernels = {
        # Detección de bordes
        'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'laplacian': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        'laplacian_diag': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),

        # Mejora de bordes y contraste
        'edge_enhance': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'sharpen_strong': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),

        # Detección de líneas direccionales
        'horizontal_lines': np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),
        'vertical_lines': np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]),
        'diagonal_lines_1': np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]),
        'diagonal_lines_2': np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),

        # Kernels personalizados para tableros de ajedrez
        'chess_pattern': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'grid_detect': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),

        # Kernels adaptativos de diferentes tamaños
        'large_edge': np.array([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ]),
    }

    # Crear kernel personalizado basado en el tamaño
    if kernel_type == 'custom_size':
        kernel = create_custom_kernel(kernel_size, strength)
    else:
        kernel = kernels.get(kernel_type, kernels['edge_enhance'])
        # Ajustar la intensidad del kernel
        kernel = kernel * strength

    # Aplicar convolución
    if kernel_type in ['sobel_x', 'sobel_y']:
        # Para Sobel, usar la función específica de OpenCV
        if kernel_type == 'sobel_x':
            filtered = cv2.Sobel(img_normalized, cv2.CV_64F, 1, 0, ksize=3)
        else:
            filtered = cv2.Sobel(img_normalized, cv2.CV_64F, 0, 1, ksize=3)
    else:
        # Aplicar convolución personalizada
        filtered = ndimage.convolve(img_normalized.astype(np.float64), kernel, mode='constant')

    # Normalizar resultado
    filtered = np.clip(filtered, 0, 255)
    return filtered.astype(np.uint8)

def create_custom_kernel(size, strength=1.0, kernel_type='edge'):
    """
    Crea kernels personalizados de diferentes tamaños
    """
    if kernel_type == 'edge':
        # Kernel de detección de bordes centrado
        kernel = np.ones((size, size)) * -1
        center = size // 2
        kernel[center, center] = (size * size - 1) * strength
        return kernel

    elif kernel_type == 'sharpen':
        # Kernel de afilado
        kernel = np.zeros((size, size))
        kernel[size//2, size//2] = strength * 5
        kernel[size//2-1, size//2] = -strength
        kernel[size//2+1, size//2] = -strength
        kernel[size//2, size//2-1] = -strength
        kernel[size//2, size//2+1] = -strength
        return kernel

    elif kernel_type == 'line_detect':
        # Kernel para detección de líneas
        kernel = np.ones((size, size)) * -1
        kernel[size//2, :] = strength * 2
        return kernel

def multi_kernel_enhancement(img, kernel_combination=['edge_enhance', 'horizontal_lines', 'vertical_lines'], weights=None):
    """
    Aplica múltiples kernels y combina los resultados
    """
    if weights is None:
        weights = [1.0] * len(kernel_combination)

    results = []
    for i, kernel_type in enumerate(kernel_combination):
        filtered = apply_convolution_kernels(img, kernel_type, strength=weights[i])
        results.append(filtered)

    # Combinar resultados (promedio ponderado)
    combined = np.zeros_like(results[0], dtype=np.float64)
    total_weight = sum(weights)

    for i, result in enumerate(results):
        combined += result.astype(np.float64) * weights[i] / total_weight

    return np.clip(combined, 0, 255).astype(np.uint8)

def adaptive_kernel_enhancement(img, tile_size=64):
    """
    Aplica kernels adaptativos por regiones de la imagen
    """
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_normalized *= 255
    img_normalized = img_normalized.astype(np.uint8)

    h, w = img_normalized.shape
    result = np.zeros_like(img_normalized)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            # Extraer tile
            tile = img_normalized[y:y+tile_size, x:x+tile_size]

            # Calcular varianza para determinar el kernel apropiado
            variance = np.var(tile)

            if variance < 100:  # Región uniforme
                kernel_type = 'edge_enhance'
                strength = 2.0
            elif variance < 500:  # Región con algo de detalle
                kernel_type = 'sharpen'
                strength = 1.5
            else:  # Región con mucho detalle
                kernel_type = 'laplacian'
                strength = 1.0

            # Aplicar kernel al tile
            enhanced_tile = apply_convolution_kernels(tile, kernel_type, strength=strength)

            # Colocar el tile procesado en el resultado
            result[y:y+tile_size, x:x+tile_size] = enhanced_tile

    return result

def enhance_thermal_image_kernel(img, use_kernels=True, kernel_type='chess_pattern', kernel_strength=1.5):
    """
    Mejora el contraste de una imagen térmica aplicando diferentes técnicas
    """
    # Tu normalización actual
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_normalized *= 255
    img_normalized = img_normalized.astype(np.uint8)

    # Aplicar kernels convolucionales si se solicita
    if use_kernels:
        img_kernel = apply_convolution_kernels(img, kernel_type, strength=kernel_strength)
        # Combinar imagen original con resultado del kernel
        img_combined = cv2.addWeighted(img_normalized, 0.6, img_kernel, 0.4, 0)
    else:
        img_combined = img_normalized

    # Resto de técnicas existentes aplicadas a la imagen mejorada con kernels
    # Opción 1: Umbralización adaptativa
    threshold_value = 128
    _, img_thresh = cv2.threshold(img_combined, threshold_value, 255, cv2.THRESH_BINARY)

    # Opción 2: Ecualización de histograma
    img_equalized = cv2.equalizeHist(img_combined)

    # Opción 3: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_combined)

    # Opción 4: Ajuste de gamma
    gamma = 0.5
    img_gamma = np.power(img_combined / 255.0, gamma) * 255
    img_gamma = img_gamma.astype(np.uint8)

    # Opción 5: Mapeo personalizado
    def custom_mapping(pixel_value):
        if pixel_value < 85:
            return pixel_value
        elif pixel_value < 170:
            return int(pixel_value * 0.3)
        else:
            return pixel_value

    img_custom = np.vectorize(custom_mapping)(img_combined)
    img_custom = img_custom.astype(np.uint8)

    # Opción 6: Stretching de contraste
    p2, p98 = np.percentile(img_combined, (2, 98))
    img_stretched = np.clip((img_combined - p2) * 255 / (p98 - p2), 0, 255)
    img_stretched = img_stretched.astype(np.uint8)

    # Nuevas opciones con kernels
    img_multi_kernel = multi_kernel_enhancement(img, ['edge_enhance', 'horizontal_lines', 'vertical_lines'], [1.0, 0.5, 0.5])
    img_adaptive = adaptive_kernel_enhancement(img)

    return {
        'original': img_normalized,
        'kernel_enhanced': img_kernel if use_kernels else img_normalized,
        'combined': img_combined,
        'threshold': img_thresh,
        'equalized': img_equalized,
        'clahe': img_clahe,
        'gamma': img_gamma,
        'custom': img_custom,
        'stretched': img_stretched,
        'multi_kernel': img_multi_kernel,
        'adaptive': img_adaptive
    }

def process_thermal_images_kernel(input_dir, output_dir, method='multi_kernel', kernel_type='chess_pattern', kernel_strength=1.5):
    """
    Procesa todas las imágenes térmicas en un directorio
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # Aplicar mejoras con kernels
            enhanced_images = enhance_thermal_image_kernel(img, use_kernels=True,
                                                 kernel_type=kernel_type,
                                                 kernel_strength=kernel_strength)

            # Guardar la imagen procesada según el método elegido
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, enhanced_images[method])

            print(f"Procesada: {filename}")


if __name__ == "__main__":

    # thermal_convertion()

    # input_directory = "captures/temp/50invert/"
    # output_directory = "captures/temp/50invert_new/"


    # process_thermal_images_kernel(input_directory, output_directory, method='kernel_enhanced', kernel_type='edge_enhance', kernel_strength=2.0)


    # input_directory = "captures/temp/50invert_new/"

    # Procesar todas las imágenes con el método personalizado


    # rgb2grayscale('captures/temp/right/20250714_152326.png' , "grayright.png")
    # rgb2grayscale('captures/temp/left/20250714_152326.png', "grayleft.png")

    # invert_thermal()

    create_anaglyph()