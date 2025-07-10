import cv2
import os
import datetime
import numpy as np
import time

class CameraInterface:
    def __init__(self):
        # Crear carpetas para las capturas si no existen
        os.makedirs("captures/thermal", exist_ok=True)
        os.makedirs("captures/visible/left", exist_ok=True)
        os.makedirs("captures/visible/right", exist_ok=True)
        
        # Variables de estado
        self.running = True
        self.capturing_photos = False
        self.message = ""
        self.message_timer = 0
        
        # Configurar cámaras
        self.setup_cameras()
        
        # Dimensiones de la interfaz
        self.display_width = 1400
        self.display_height = 800
        
        # Colores para la interfaz
        self.bg_color = (45, 45, 45)
        self.button_color = (70, 130, 180)
        self.button_hover_color = (100, 149, 237)
        self.text_color = (255, 255, 255)
        self.success_color = (0, 255, 0)
        self.error_color = (0, 0, 255)
        
        # Posiciones y tamaños de botones
        self.button_width = 150
        self.button_height = 50
        self.button_spacing = 20
        
        # Crear ventana principal
        cv2.namedWindow('Camera Interface', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Interface', self.display_width, self.display_height)
        cv2.setMouseCallback('Camera Interface', self.mouse_callback)
        
        # Imprimir instrucciones en terminal
        print("=" * 60)
        print("INTERFAZ DE CÁMARA - INSTRUCCIONES")
        print("=" * 60)
        print("• Hacer clic en 'Tomar Fotos' para capturar 4 imágenes")
        print("• Presionar 'c' como atajo para capturar")
        print("• Presionar 'q' o hacer clic en 'Salir' para terminar")
        print("• Las imágenes se guardan automáticamente en /captures/")
        print("• Se capturan 4 fotos con intervalo de 4 segundos")
        print("=" * 60)
        
    def setup_cameras(self):
        """Configurar las cámaras"""
        # IMPORTANTE: Cambiar estos IDs según como se conecten las cámaras
        self.cap_visible = cv2.VideoCapture(1)
        self.cap_thermal = cv2.VideoCapture(2)
        
        if not self.cap_visible.isOpened():
            print("Error: No se pudo abrir la cámara visible.")
            self.running = False
            return
            
        if not self.cap_thermal.isOpened():
            print("Error: No se pudo abrir la cámara térmica.")
            self.running = False
            return
            
        # Configurar resolución de la cámara visible
        self.cap_visible.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap_visible.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Cámaras configuradas correctamente")
        
    def create_button(self, img, x, y, width, height, text, is_hovered=False):
        """Crear un botón con estilo moderno"""
        color = self.button_hover_color if is_hovered else self.button_color
        
        # Dibujar botón con bordes redondeados (simulado)
        cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 255), 2)
        
        # Añadir texto centrado
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, self.text_color, thickness)
        
        return (x, y, width, height)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Manejar eventos del mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Verificar si se hizo clic en algún botón
            if self.is_point_in_button(x, y, self.capture_button_rect):
                self.capture_photos()
            elif self.is_point_in_button(x, y, self.exit_button_rect):
                self.exit_app()
    
    def is_point_in_button(self, x, y, button_rect):
        """Verificar si un punto está dentro de un botón"""
        bx, by, bw, bh = button_rect
        return bx <= x <= bx + bw and by <= y <= by + bh
    
    def capture_photos(self):
        """Capturar 4 fotos con intervalo de 4 segundos"""
        if self.capturing_photos:
            return
            
        time.sleep(4)
        self.capturing_photos = True
        self.message = "Iniciando captura..."
        self.message_timer = time.time()

        iterations = 9
        
        for i in range(iterations):
            # Obtener timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Definir nombres de archivo
            left_filename = f"captures/visible/left/LEFT_visible_{timestamp}.png"
            right_filename = f"captures/visible/right/RIGHT_visible_{timestamp}.png"
            thermal_filename = f"captures/thermal/thermal_{timestamp}.png"
            
            # Leer imágenes de las cámaras
            ret_visible, img_visible = self.cap_visible.read()
            ret_thermal, img_thermal = self.cap_thermal.read()
            
            if ret_visible and ret_thermal:
                # Dividir imagen visible
                height, width, _ = img_visible.shape
                mid_width = width // 2
                
                img_left = img_visible[:, :mid_width]
                img_right = img_visible[:, mid_width:]
                
                # Convertir imagen térmica a escala de grises
                img_thermal_gray = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)
                
                # Guardar imágenes
                cv2.imwrite(left_filename, img_left)
                cv2.imwrite(right_filename, img_right)
                cv2.imwrite(thermal_filename, img_thermal_gray)
                
                self.message = f"Captura {i+1}/{iterations} completada"
                self.message_timer = time.time()
                print(f"Captura {i+1} realizada: {left_filename}, {right_filename}, {thermal_filename}")
                
                # Esperar 4 segundos antes de la siguiente captura (excepto la última)
                if i < iterations - 1:
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        self.update_display()
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            else:
                self.message = "Error al capturar imágenes"
                self.message_timer = time.time()
                print("Error al capturar imágenes")
                break
        
        self.message = "¡Todas las capturas completadas!"
        self.message_timer = time.time()
        self.capturing_photos = False
    
    def exit_app(self):
        """Salir de la aplicación"""
        self.running = False
    
    def update_display(self):
        """Actualizar la pantalla principal"""
        # Crear imagen de fondo
        display_img = np.full((self.display_height, self.display_width, 3), self.bg_color, dtype=np.uint8)
        
        # Leer imágenes de las cámaras
        ret_visible, img_visible = self.cap_visible.read()
        ret_thermal, img_thermal = self.cap_thermal.read()
        
        if ret_visible and ret_thermal:
            # Procesar imagen visible
            height, width, _ = img_visible.shape
            mid_width = width // 2
            
            img_left = img_visible[:, :mid_width]
            img_right = img_visible[:, mid_width:]
            
            # Procesar imagen térmica (escala de grises para tiempo real)
            img_thermal_gray = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)
            img_thermal_display = cv2.cvtColor(img_thermal_gray, cv2.COLOR_GRAY2BGR)
            
            # Redimensionar imágenes para la visualización (más grandes)
            display_size_color = (400, 300)  # Tamaño más grande para cámaras color
            display_size_thermal = (400, 300)  # Tamaño para cámara térmica
            
            img_left_resized = cv2.resize(img_left, display_size_color)
            img_right_resized = cv2.resize(img_right, display_size_color)
            img_thermal_resized = cv2.resize(img_thermal_display, display_size_thermal)
            
            # Posicionar imágenes - cámaras color arriba
            y_offset_color = 30
            x_positions_color = [50, 500]  # Izquierda y derecha
            
            # Posicionar imagen térmica - abajo centrada
            y_offset_thermal = 380
            x_position_thermal = (self.display_width - display_size_thermal[0]) // 2
            
            # Colocar imágenes color arriba
            display_img[y_offset_color:y_offset_color+display_size_color[1], 
                       x_positions_color[0]:x_positions_color[0]+display_size_color[0]] = img_left_resized
            display_img[y_offset_color:y_offset_color+display_size_color[1], 
                       x_positions_color[1]:x_positions_color[1]+display_size_color[0]] = img_right_resized
            
            # Colocar imagen térmica abajo
            display_img[y_offset_thermal:y_offset_thermal+display_size_thermal[1], 
                       x_position_thermal:x_position_thermal+display_size_thermal[0]] = img_thermal_resized
            
            # Añadir etiquetas
            labels_color = ["Left Camera", "Right Camera"]
            for i, label in enumerate(labels_color):
                cv2.putText(display_img, label, (x_positions_color[i], y_offset_color - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)
            
            # Etiqueta para cámara térmica
            cv2.putText(display_img, "Thermal Camera", (x_position_thermal, y_offset_thermal - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)
        
        # Crear botones (en la parte inferior)
        button_y = 720
        button_x_start = (self.display_width - (2 * self.button_width + self.button_spacing)) // 2
        
        # Botón de captura
        self.capture_button_rect = self.create_button(
            display_img, button_x_start, button_y, 
            self.button_width, self.button_height, 
            "Tomar Fotos", self.capturing_photos
        )
        
        # Botón de salir
        self.exit_button_rect = self.create_button(
            display_img, button_x_start + self.button_width + self.button_spacing, 
            button_y, self.button_width, self.button_height, 
            "Salir", False
        )
        
        # Mostrar mensaje si existe
        if self.message and time.time() - self.message_timer < 3:
            color = self.success_color if "completada" in self.message or "completadas" in self.message else self.text_color
            # Centrar mensaje
            text_size = cv2.getTextSize(self.message, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = (self.display_width - text_size[0]) // 2
            cv2.putText(display_img, self.message, (text_x, 700), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Mostrar información de estado
        status_text = "Listo para capturar" if not self.capturing_photos else "Capturando..."
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        status_x = (self.display_width - status_size[0]) // 2
        cv2.putText(display_img, status_text, (status_x, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
        
        # Mostrar la imagen
        cv2.imshow('Camera Interface', display_img)
    
    def run(self):
        """Ejecutar la aplicación principal"""
        if not self.running:
            print("Error: No se pudieron inicializar las cámaras")
            return
            
        print("Interfaz de cámara iniciada")
        print("Presiona 'q' para salir")
        
        while self.running:
            self.update_display()
            
            # Manejar eventos de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.exit_app()
            elif key == ord('c'):  # Atajo de teclado para capturar
                self.capture_photos()
        
        # Limpiar recursos
        self.cap_visible.release()
        self.cap_thermal.release()
        cv2.destroyAllWindows()
        print("Aplicación cerrada")

# Ejecutar la aplicación
if __name__ == "__main__":
    app = CameraInterface()
    app.run()