
import torch
from util import *
from Image_registration import registration
from Fusion.fusion import *
import tifffile as tf
from Disparity.disparity import compute_disparity
def main(imageLeft, imageRight, imageThermal, threshold=200, transformation_method="homography"): 
    
    # Procesar las imágenes utilizando el registro
    image_warped, matches, scores, error = registration.procesar_imagenes(ruta_imagen0=imageThermal, ruta_imagen1=imageLeft, threshold=threshold, transformation_method=transformation_method)
    # show_image(image_registered_bgr, 'Warped Image')

    
    disparity = compute_disparity(img_left=imageLeft,img_right=imageRight)

    fusioned_image_bgrt = fusion_bgr_lwir(imageLeft, image_warped)
    fusioned_image_bgrtd = fusion_bgr_lwir_disparity(imagen_bgr=imageLeft, imagen_lwir=image_warped, imagen_disparity=disparity)

    # rgbt_image = extract_channels(fusioned_image_bgrt, [2,1,0,3])

    rgbtd_image = extract_channels(fusioned_image_bgrtd, [2,1,0,3,4])

    tf.imwrite(
        "rgbtd.tiff", 
        rgbtd_image, 
        photometric='rgb', 
        metadata={'description':'RGB Image + Thermal LWIR Channel + Disparity'},
        compression=None)
    
    cv2.imwrite("disparity_map.tiff", disparity.astype(np.float32))

    

    
    

if __name__ == "__main__":
    
    # Rutas de las imágenes
    # image_thermal_path = "Cameras/captures/video_image_extractor_results/thermal/image_00005.png"
    # image_left_path = "Cameras/captures/video_image_extractor_results/left/image_00005.png"
    # image_right_path = "Cameras/captures/video_image_extractor_results/right/image_00005.png"

    #image_thermal_path = "Cameras/captures/thermal/thermal_20241030_130522.png"
    #image_left_path = "Cameras/captures/visible/left_rect/LEFT_visible_20241030_130522.png"
    #image_right_path = "Cameras/captures/visible/right_rect/RIGHT_visible_20241030_130522.png"

    #pocos matches
    #image_thermal_path = "./captures/thermal/thermal_20250612_152008.png"
    #image_left_path = "./captures/visible/left/LEFT_visible_20250612_152008.png"
    #image_right_path = "./captures/visible/right/RIGHT_visible_20250612_152008.png"

    image_thermal_path = "./captures/thermal/thermal_20250625_102056.png"
    image_left_path = "./captures/visible/left/LEFT_visible_20250625_102056.png"
    image_right_path = "./captures/visible/right/RIGHT_visible_20250625_102056.png"

    transformation_name = "homography"
    # Umbral para el registro
    threshold = 100
    #cambiar a threshold=200

    main(imageLeft=image_left_path, imageRight=image_right_path,imageThermal=image_thermal_path, threshold=threshold, transformation_method=transformation_name)

    

    print("END")





