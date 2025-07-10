import cv2
import numpy as np
import os

class StereoCamera:
    def __init__(self, calibration_results):
        """
        Inicializa el sistema de cámaras estéreo con los resultados de calibración
        """
        # Parámetros intrínsecos
        self.mtx_color1 = calibration_results['mtx_color1']
        self.mtx_color2 = calibration_results['mtx_color2'] 
        self.mtx_thermal = calibration_results['mtx_thermal']
        
        self.dist_color1 = calibration_results['dist_color1']
        self.dist_color2 = calibration_results['dist_color2']
        self.dist_thermal = calibration_results['dist_thermal']
        
        # Parámetros extrínsecos
        self.R_color = calibration_results['R_color']
        self.T_color = calibration_results['T_color']
        self.R_thermal = calibration_results['R_thermal'] 
        self.T_thermal = calibration_results['T_thermal']
        
        # Mapas de rectificación (se calculan una vez)
        self.maps_initialized = False
        self.setup_rectification_maps()
    
    def setup_rectification_maps(self):
        """
        Calcula los mapas de rectificación para las cámaras
        """
        # Asumiendo resolución típica (ajustar según tus imágenes)
        image_size = (640, 480)  # Ajusta según tu resolución real
        
        # Rectificación para par estéreo color-color
        self.R1_color, self.R2_color, self.P1_color, self.P2_color, self.Q_color, _, _ = cv2.stereoRectify(
            self.mtx_color1, self.dist_color1,
            self.mtx_color2, self.dist_color2,
            image_size, self.R_color, self.T_color
        )
        
        # Rectificación para par estéreo color1-térmica
        self.R1_thermal, self.R2_thermal, self.P1_thermal, self.P2_thermal, self.Q_thermal, _, _ = cv2.stereoRectify(
            self.mtx_color1, self.dist_color1,
            self.mtx_thermal, self.dist_thermal,
            image_size, self.R_thermal, self.T_thermal
        )
        
        # Mapas de rectificación para cámaras color
        self.map1_color1, self.map2_color1 = cv2.initUndistortRectifyMap(
            self.mtx_color1, self.dist_color1, self.R1_color, self.P1_color, image_size, cv2.CV_32FC1
        )
        self.map1_color2, self.map2_color2 = cv2.initUndistortRectifyMap(
            self.mtx_color2, self.dist_color2, self.R2_color, self.P2_color, image_size, cv2.CV_32FC1
        )
        
        # Mapas de rectificación para cámara térmica con color1
        self.map1_color1_thermal, self.map2_color1_thermal = cv2.initUndistortRectifyMap(
            self.mtx_color1, self.dist_color1, self.R1_thermal, self.P1_thermal, image_size, cv2.CV_32FC1
        )
        self.map1_thermal, self.map2_thermal = cv2.initUndistortRectifyMap(
            self.mtx_thermal, self.dist_thermal, self.R2_thermal, self.P2_thermal, image_size, cv2.CV_32FC1
        )
        
        self.maps_initialized = True
        print("✅ Mapas de rectificación inicializados")
    
    def rectify_stereo_color(self, img_left, img_right):
        """
        Rectifica un par de imágenes de las cámaras color
        """
        if not self.maps_initialized:
            raise Exception("Los mapas de rectificación no están inicializados")
        
        img_left_rect = cv2.remap(img_left, self.map1_color1, self.map2_color1, cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right, self.map1_color2, self.map2_color2, cv2.INTER_LINEAR)
        
        return img_left_rect, img_right_rect
    
    def rectify_thermal_color(self, img_color1, img_thermal):
        """
        Rectifica un par de imágenes color1-térmica
        """
        if not self.maps_initialized:
            raise Exception("Los mapas de rectificación no están inicializados")
        
        img_color1_rect = cv2.remap(img_color1, self.map1_color1_thermal, self.map2_color1_thermal, cv2.INTER_LINEAR)
        img_thermal_rect = cv2.remap(img_thermal, self.map1_thermal, self.map2_thermal, cv2.INTER_LINEAR)
        
        return img_color1_rect, img_thermal_rect
    
    def process_images_batch(self, color1_folder, color2_folder, thermal_folder, output_folder):
        """
        Procesa un lote de imágenes y guarda las versiones rectificadas
        """
        # Crear directorios de salida
        os.makedirs(f"{output_folder}/color1_rect", exist_ok=True)
        os.makedirs(f"{output_folder}/color2_rect", exist_ok=True)
        os.makedirs(f"{output_folder}/thermal_rect", exist_ok=True)
        os.makedirs(f"{output_folder}/color_stereo", exist_ok=True)
        os.makedirs(f"{output_folder}/thermal_color_stereo", exist_ok=True)
        
        # Obtener listas de imágenes
        color1_images = sorted([f for f in os.listdir(color1_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        color2_images = sorted([f for f in os.listdir(color2_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        thermal_images = sorted([f for f in os.listdir(thermal_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        num_images = min(len(color1_images), len(color2_images), len(thermal_images))
        print(f"Procesando {num_images} imágenes...")
        
        for i in range(num_images):
            # Cargar imágenes
            img_color1 = cv2.imread(os.path.join(color1_folder, color1_images[i]))
            img_color2 = cv2.imread(os.path.join(color2_folder, color2_images[i]))
            img_thermal = cv2.imread(os.path.join(thermal_folder, thermal_images[i]))
            
            if img_color1 is None or img_color2 is None or img_thermal is None:
                print(f"⚠️  Error cargando imágenes {i}")
                continue
            
            # Rectificar pares estéreo
            color1_rect, color2_rect = self.rectify_stereo_color(img_color1, img_color2)
            color1_thermal_rect, thermal_rect = self.rectify_thermal_color(img_color1, img_thermal)
            
            # Guardar imágenes individuales rectificadas
            cv2.imwrite(f"{output_folder}/color1_rect/color1_{i:03d}.png", color1_rect)
            cv2.imwrite(f"{output_folder}/color2_rect/color2_{i:03d}.png", color2_rect)
            cv2.imwrite(f"{output_folder}/thermal_rect/thermal_{i:03d}.png", thermal_rect)
            
            # Crear y guardar pares estéreo lado a lado
            stereo_color = np.hstack((color1_rect, color2_rect))
            stereo_thermal_color = np.hstack((color1_thermal_rect, thermal_rect))
            
            # Dibujar líneas de epipolar para verificación visual
            height = stereo_color.shape[0]
            for y in range(0, height, height//10):
                cv2.line(stereo_color, (0, y), (stereo_color.shape[1], y), (0, 255, 0), 1)
                cv2.line(stereo_thermal_color, (0, y), (stereo_thermal_color.shape[1], y), (0, 255, 0), 1)
            
            cv2.imwrite(f"{output_folder}/color_stereo/stereo_color_{i:03d}.png", stereo_color)
            cv2.imwrite(f"{output_folder}/thermal_color_stereo/stereo_thermal_{i:03d}.png", stereo_thermal_color)
            
            if i % 10 == 0:
                print(f"Procesadas {i+1}/{num_images} imágenes")
        
        print(f"✅ Procesamiento completo. Imágenes guardadas en {output_folder}")
    
    def capture_and_rectify_live(self, color1_cam_id=0, thermal_cam_id=1):
        """
        Captura en vivo y muestra imágenes rectificadas (requiere cámaras conectadas)
        """
        # Inicializar cámaras
        cap_visible = cv2.VideoCapture(color1_cam_id)
        cap_visible.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_visible.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # cap_color2 = cv2.VideoCapture(color2_cam_id)
        cap_thermal = cv2.VideoCapture(thermal_cam_id)
        cap_thermal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_thermal.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not all([cap_visible.isOpened(), cap_thermal.isOpened()]):
            print("❌ Error: No se pudieron abrir todas las cámaras")
            return
        
        print("🎥 Captura en vivo iniciada. Presiona 'q' para salir, 's' para guardar imagen actual")
        
        save_counter = 0
        
        while True:
            # Leer la imagen de la cámara visible
            ret_visible, img_visible = cap_visible.read()
            if not ret_visible:
                print("No se pudo capturar la imagen de la cámara visible.")
                self.capturing = False
                return
            # Dividir la imagen visible en dos imágenes: izquierda y derecha
            height, width, _ = img_visible.shape
            mid_width = width // 2  # Mitad del ancho

            # Imagen izquierda
            cap_color1 = img_visible[:, :mid_width]

            # Imagen derecha
            cap_color2 = img_visible[:, mid_width:]
            # Capturar frames
            # ret1, frame_color1 = cap_color1.read()
            # ret2, frame_color2 = cap_color2.read()
            ret3, frame_thermal = cap_thermal.read()
            
            # if not all([ret1, ret2, ret3]):
            #     print("❌ Error capturando frames")
            #     break
            
            # Rectificar frames
            color1_rect, color2_rect = self.rectify_stereo_color(cap_color1, cap_color2)
            color1_thermal_rect, thermal_rect = self.rectify_thermal_color(cap_color1, cv2.bitwise_not((frame_thermal)))
            
            # Crear visualizaciones
            stereo_color = np.hstack((color1_rect, color2_rect))
            stereo_thermal = np.hstack((color1_thermal_rect, thermal_rect))

            for y in range(0, height, height//10):
                cv2.line(stereo_color, (0, y), (stereo_color.shape[1], y), (0, 255, 0), 1)
                cv2.line(stereo_thermal, (0, y), (stereo_thermal.shape[1], y), (0, 255, 0), 1)
            
            # Mostrar
            cv2.imshow('Stereo Color Rectified', stereo_color)
            cv2.imshow('Thermal-Color Rectified', stereo_thermal)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Guardar imagen actual
                cv2.imwrite(f'captured_stereo_color_{save_counter:03d}.png', stereo_color)
                cv2.imwrite(f'captured_thermal_color_{save_counter:03d}.png', stereo_thermal)
                print(f"💾 Imagen {save_counter} guardada")
                save_counter += 1
        
        # Limpiar
        cap_color1.release()
        cap_color2.release()
        cap_thermal.release()
        cv2.destroyAllWindows()

# Ejemplo de uso con tus resultados de calibración
if __name__ == "__main__":
    calibration_data = np.load('Cameras/calibration_data.npz')
    print(calibration_data['R_thermal'])
    # Aquí pondrías los resultados reales de tu calibración
    calibration_results = {
        'mtx_color1':   calibration_data['mtx_color1'],  
        'mtx_color2':   calibration_data['mtx_color2'],  
        'mtx_thermal':  calibration_data['mtx_thermal'], 
        'dist_color1':  calibration_data['dist_color1'], 
        'dist_color2':  calibration_data['dist_color2'], 
        'dist_thermal': calibration_data['dist_thermal'],
        'R_color':      calibration_data['R_color'],
        'T_color':      calibration_data['T_color'],
        'R_thermal':    calibration_data['R_thermal'],
        'T_thermal':    calibration_data['T_thermal']
    }
    
    # Crear sistema de cámaras
    stereo_system = StereoCamera(calibration_results)
    
    # Procesar imágenes existentes
    stereo_system.process_images_batch(
        color1_folder="Cameras/calibration_caps/calibration_dataset/rgb",
        color2_folder="Cameras/calibration_caps/right", 
        thermal_folder="Cameras/calibration_caps/calibration_dataset/rgb",
        output_folder="output_rectified"
    )

    # stereo_system.capture_and_rectify_live()
