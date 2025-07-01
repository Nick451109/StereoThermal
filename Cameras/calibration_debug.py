import cv2
import numpy as np
import glob
import os

# Parámetros iniciales
CHECKERBOARD_SIZE = (6, 9)  # Ajustar según tu patrón
square_size = 20.0  # Tamaño de los cuadrados del checkerboard en mm

# Criterios para cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepara puntos del mundo (coordenadas 3D reales del patrón)
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * square_size

# Almacenar puntos 3D y puntos 2D de las imágenes
objpoints_stereo_color  = []  # Puntos 3D para calibración estéreo color
imgpoints_color1_stereo = []  # Puntos 2D de la cámara color 1 para estéreo
imgpoints_color2_stereo = []  # Puntos 2D de la cámara color 2 para estéreo

objpoints_stereo_thermal = []  # Puntos 3D para calibración estéreo térmica
imgpoints_color1_thermal = []  # Puntos 2D de la cámara color 1 para estéreo térmico
imgpoints_thermal = []  # Puntos 2D de la cámara térmica

# Para calibración individual
objpoints_color1 = []
imgpoints_color1 = []

objpoints_color2 = []
imgpoints_color2 = []

objpoints_thermal_individual = []
imgpoints_thermal_individual = []

# Carga las imágenes para cada cámara
color1_images  = glob.glob('Cameras/calibration_caps/left/*.png')
color2_images  = glob.glob('Cameras/calibration_caps/right/*.png')
thermal_images = glob.glob('Cameras/calibration_caps/thermal_invert/*.png')

print("Número de imágenes color 1:", len(color1_images))
print("Número de imágenes color 2:", len(color2_images))
print("Número de imágenes térmicas:", len(thermal_images))

# print(color1_images)
# print(color2_images)
# print(thermal_images)
# Asegúrate de que todas las listas de imágenes estén ordenadas y tengan la misma longitud
color1_images.sort()
color2_images.sort()
thermal_images.sort()

num_images = min(len(color1_images), len(color2_images), len(thermal_images))

# Contadores para debug
color1_detections = 0
color2_detections = 0
thermal_detections = 0

# Detectar esquinas del checkerboard y sincronizar detecciones
for i in range(num_images):
    color1_img_path = color1_images[i]
    color2_img_path = color2_images[i]
    thermal_img_path = thermal_images[i]

    print(f"Procesando imagen {i+1}/{num_images}: {os.path.basename(thermal_img_path)}")

    # Leer imágenes
    img_color1 = cv2.imread(color1_img_path, cv2.IMREAD_GRAYSCALE)
    img_color2 = cv2.imread(color2_img_path, cv2.IMREAD_GRAYSCALE)
    img_thermal = cv2.imread(thermal_img_path, cv2.IMREAD_GRAYSCALE)
    
    # Verificar que las imágenes se cargaron correctamente
    if img_color1 is None:
        print(f"Error: No se pudo cargar {color1_img_path}")
        continue
    if img_color2 is None:
        print(f"Error: No se pudo cargar {color2_img_path}")
        continue
    if img_thermal is None:
        print(f"Error: No se pudo cargar {thermal_img_path}")
        continue

    # Detectar esquinas en cámara color 1
    ret1, corners1 = cv2.findChessboardCorners(img_color1, CHECKERBOARD_SIZE, None)
    if ret1:
        corners_refined1 = cv2.cornerSubPix(img_color1, corners1, (11, 11), (-1, -1), criteria)
        imgpoints_color1.append(corners_refined1)
        objpoints_color1.append(objp)
        color1_detections += 1
        print(f"  ✓ Esquinas detectadas en color1")
    else:
        print(f"  ✗ No se detectaron esquinas en color1")

    # Detectar esquinas en cámara color 2
    ret2, corners2 = cv2.findChessboardCorners(img_color2, CHECKERBOARD_SIZE, None)
    if ret2:
        corners_refined2 = cv2.cornerSubPix(img_color2, corners2, (11, 11), (-1, -1), criteria)
        imgpoints_color2.append(corners_refined2)
        objpoints_color2.append(objp)
        color2_detections += 1
        print(f"  ✓ Esquinas detectadas en color2")
    else:
        print(f"  ✗ No se detectaron esquinas en color2")

    # Detectar esquinas en cámara térmica con parámetros adaptados
    ret3, corners3 = cv2.findChessboardCorners(img_thermal, CHECKERBOARD_SIZE, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                               cv2.CALIB_CB_NORMALIZE_IMAGE +
                                               cv2.CALIB_CB_FILTER_QUADS)
    if ret3:
        corners_refined3 = cv2.cornerSubPix(img_thermal, corners3, (11, 11), (-1, -1), criteria)
        imgpoints_thermal_individual.append(corners_refined3)
        objpoints_thermal_individual.append(objp)
        thermal_detections += 1
        print(f"  ✓ Esquinas detectadas en térmica")
    else:
        print(f"  ✗ No se detectaron esquinas en térmica")
        
        # Debug: Mostrar imagen térmica para inspección visual
        img_thermal_display = cv2.resize(img_thermal, (400, 300))
        cv2.imshow('Imagen Termica - No se detectaron esquinas', img_thermal_display)
        key = cv2.waitKey(1000)  # Mostrar por 1 segundo
        if key == 27:  # ESC para salir
            break

    # Sincronizar detecciones para calibración estéreo entre cámaras color
    if ret1 and ret2:
        objpoints_stereo_color.append(objp)
        imgpoints_color1_stereo.append(corners_refined1)
        imgpoints_color2_stereo.append(corners_refined2)

    # Sincronizar detecciones para calibración estéreo entre cámara térmica y color 1
    if ret1 and ret3:
        objpoints_stereo_thermal.append(objp)
        imgpoints_color1_thermal.append(corners_refined1)
        imgpoints_thermal.append(corners_refined3)

cv2.destroyAllWindows()

# Mostrar estadísticas de detección
print(f"\n=== ESTADÍSTICAS DE DETECCIÓN ===")
print(f"Color1: {color1_detections}/{num_images} detecciones ({color1_detections/num_images*100:.1f}%)")
print(f"Color2: {color2_detections}/{num_images} detecciones ({color2_detections/num_images*100:.1f}%)")
print(f"Térmica: {thermal_detections}/{num_images} detecciones ({thermal_detections/num_images*100:.1f}%)")
print(f"Estéreo color-color: {len(objpoints_stereo_color)} pares sincronizados")
print(f"Estéreo térmica-color1: {len(objpoints_stereo_thermal)} pares sincronizados")

# Verificar si hay suficientes detecciones para calibrar
min_images_required = 10  # Mínimo recomendado para calibración

if len(objpoints_color1) < min_images_required:
    print(f"\n⚠️  ADVERTENCIA: Solo {len(objpoints_color1)} detecciones en color1. Se recomiendan al menos {min_images_required}")
if len(objpoints_color2) < min_images_required:
    print(f"\n⚠️  ADVERTENCIA: Solo {len(objpoints_color2)} detecciones en color2. Se recomiendan al menos {min_images_required}")
if len(objpoints_thermal_individual) < min_images_required:
    print(f"\n❌ ERROR: Solo {len(objpoints_thermal_individual)} detecciones en térmica. Se recomiendan al menos {min_images_required}")
    print("   Posibles soluciones:")
    print("   1. Usar un tablero con mejor contraste térmico")
    print("   2. Ajustar la temperatura del tablero")
    print("   3. Verificar el tamaño del tablero (CHECKERBOARD_SIZE)")
    print("   4. Mejorar la iluminación o configuración de la cámara térmica")
    exit(1)

# Solo continuar con la calibración si hay suficientes detecciones
print(f"\n=== INICIANDO CALIBRACIÓN ===")

# Calibrar cámaras individualmente
print("Calibrando cámara color 1...")
ret1, mtx_color1, dist_color1, _, _ = cv2.calibrateCamera(objpoints_color1, imgpoints_color1, img_color1.shape[::-1], None, None)
print(f"Error de reproyección color1: {ret1:.3f}")

print("Calibrando cámara color 2...")
ret2, mtx_color2, dist_color2, _, _ = cv2.calibrateCamera(objpoints_color2, imgpoints_color2, img_color2.shape[::-1], None, None)
print(f"Error de reproyección color2: {ret2:.3f}")

print("Calibrando cámara térmica...")
ret3, mtx_thermal, dist_thermal, _, _ = cv2.calibrateCamera(objpoints_thermal_individual, imgpoints_thermal_individual, img_thermal.shape[::-1], None, None)
print(f"Error de reproyección térmica: {ret3:.3f}")

# Verificar que las listas para calibración estéreo tengan la misma longitud
print(f"\n=== CALIBRACIÓN ESTÉREO ===")
print("Número de vistas para calibración estéreo color-color:", len(objpoints_stereo_color))
print("Número de vistas para calibración estéreo térmica-color1:", len(objpoints_stereo_thermal))

# Solo hacer calibración estéreo si hay suficientes pares sincronizados
if len(objpoints_stereo_color) >= 5:
    print("Realizando calibración estéreo color-color...")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret_color, _, _, _, _, R_color, T_color, _, _ = cv2.stereoCalibrate(
        objpoints_stereo_color, imgpoints_color1_stereo, imgpoints_color2_stereo,
        mtx_color1, dist_color1, mtx_color2, dist_color2, img_color1.shape[::-1],
        flags=flags, criteria=criteria)
    print(f"Error estéreo color-color: {ret_color:.3f}")
else:
    print("❌ Insuficientes pares sincronizados para calibración estéreo color-color")
    R_color = T_color = None

if len(objpoints_stereo_thermal) >= 5:
    print("Realizando calibración estéreo térmica-color1...")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret_thermal, _, _, _, _, R_thermal, T_thermal, _, _ = cv2.stereoCalibrate(
        objpoints_stereo_thermal, imgpoints_color1_thermal, imgpoints_thermal,
        mtx_color1, dist_color1, mtx_thermal, dist_thermal, img_color1.shape[::-1],
        flags=flags, criteria=criteria)
    print(f"Error estéreo térmica-color1: {ret_thermal:.3f}")
else:
    print("❌ Insuficientes pares sincronizados para calibración estéreo térmica-color1")
    R_thermal = T_thermal = None

# Mostrar resultados clave
print(f"\n=== RESULTADOS ===")
print("Matriz intrínseca cámara color 1:")
print(mtx_color1)
print("\nMatriz intrínseca cámara térmica:")
print(mtx_thermal)
if R_color is not None:
    print("\nMatriz de traslación entre cámaras térmica y color 1:")
    print(T_color)
    print("\nMatriz de rotación entre cámaras color:")
    print(R_color)
if T_thermal is not None:
    print("\nMatriz de traslación entre cámaras térmica y color 1:")
    print(T_thermal)
    print("\nMatriz de rotación entre cámaras térmica y color 1:")
    print(R_thermal)

print(f"\n✓ Calibración completada exitosamente!")

# Guardar los resultados de calibración en un archivo .npz
np.savez('Cameras/calibration_data.npz',
    mtx_color1=mtx_color1, dist_color1=dist_color1,
    mtx_color2=mtx_color2, dist_color2=dist_color2,
    mtx_thermal=mtx_thermal, dist_thermal=dist_thermal,
    R_color=R_color, T_color=T_color,
    R_thermal=R_thermal, T_thermal=T_thermal)