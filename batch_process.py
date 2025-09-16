import os
import glob
from tqdm import tqdm
from Image_registration.util import histogram_match_visible_to_thermal
from main import main  # Importamos tu función principal

def run_batch(
    rgb_dir="./Datasets/parallax_Tardal/rgb_out",
    thermal_dir="./Datasets/parallax_Tardal/thermal_out",
    aplicar_histogram_matching=True,
    extractor="disk",
    threshold=0,
):
    # Buscar imágenes
    rgb_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    thermal_images = sorted(glob.glob(os.path.join(thermal_dir, "*.png")))

    # Transformaciones a aplicar
    transformations = [
        "homography",
        "affine",
        "rigid",
        "similarity",
        "translation",
        "translation2",
        "rigid_ransac",
        "translation_ransac",
    ]

    print(f"[INFO] Encontradas {len(rgb_images)} imágenes RGB y {len(thermal_images)} térmicas")

    for rgb_path, thermal_path in tqdm(zip(rgb_images, thermal_images), total=len(rgb_images), desc="Procesando pares"):
        base_filename = os.path.splitext(os.path.basename(thermal_path))[0]

        # Histogram matching
        if aplicar_histogram_matching:
            matched_path = histogram_match_visible_to_thermal(
                visible_rgb_path=rgb_path,
                thermal_gray_path=thermal_path,
                mostrar=False,  # No mostrar gráficos en batch
                output_path=f"./Datasets/parallax_Tardal/hist_matched/{base_filename}.png"
            )
            image_left_path = matched_path
            image_right_path = matched_path
        else:
            image_left_path = rgb_path
            image_right_path = rgb_path

        # Aplicar todas las transformaciones
        for transformation_name in transformations:
            main(
                imageLeft=image_left_path,
                imageRight=image_right_path,
                imageThermal=thermal_path,
                output_name=base_filename,
                extractor=extractor,
                transformation_method=transformation_name,
                threshold=threshold,
                usar_histogram_matching=aplicar_histogram_matching,
            )

    print("[INFO] Procesamiento por lotes finalizado.")


if __name__ == "__main__":
    run_batch()
