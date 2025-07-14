import cv2
import os

def invert_thermal():
    # Directorio de entrada (imágenes originales)
    input_dir = "captures/thermal/"
    # Directorio de salida (imágenes invertidas)
    output_dir = "captures/inverse/"

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


def rgb2grayscale():

    # Cargar la imagen
    imagen = cv2.imread('captures/right2/RIGHT_visible_20250710_110814.png')

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

    cv2.imwrite('gray.png', imagen_gris)


if __name__ == "__main__":

    rgb2grayscale()