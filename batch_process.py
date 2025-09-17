import os
import glob
from datetime import datetime
from tqdm import tqdm
from Image_registration.util import histogram_match_visible_to_thermal, save_metrics_to_excel
from main import main

def run_batch(
    rgb_dir="./Datasets/parallax_Tardal/rgb_out",
    thermal_dir="./Datasets/parallax_Tardal/thermal_out",
    aplicar_histogram_matching=True,
    threshold=0,
):
    # Crear carpeta de resultados única con timestamp
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join("results", f"results_batch_{fecha}")
    os.makedirs(results_root, exist_ok=True)

    # Excel global
    global_xlsx = os.path.join(results_root, "metrics_results.xlsx")

    # Buscar imágenes
    rgb_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    thermal_images = sorted(glob.glob(os.path.join(thermal_dir, "*.png")))

    # Extractores a evaluar
    extractors = ["disk", "aliked", "superpoint"]

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

    for extractor in extractors:
        print(f"\n[INFO] Procesando extractor: {extractor}")

        # Carpeta para este extractor
        extractor_dir = os.path.join(results_root, extractor)
        os.makedirs(extractor_dir, exist_ok=True)
        matches_dir = os.path.join(extractor_dir, "matches")
        os.makedirs(matches_dir, exist_ok=True)

        # Excel por extractor
        extractor_xlsx = os.path.join(extractor_dir, f"metrics_results_{extractor}.xlsx")

        # Carpeta para histogram matching (una sola global)
        if aplicar_histogram_matching:
            hist_dir = os.path.join(results_root, "hist_matched")
            os.makedirs(hist_dir, exist_ok=True)
        else:
            hist_dir = None

        for rgb_path, thermal_path in tqdm(zip(rgb_images, thermal_images), total=len(rgb_images), desc=extractor):
            base_filename = os.path.splitext(os.path.basename(thermal_path))[0]

            # Histogram matching
            if aplicar_histogram_matching:
                matched_path = histogram_match_visible_to_thermal(
                    visible_rgb_path=rgb_path,
                    thermal_gray_path=thermal_path,
                    mostrar=False,
                    output_path=os.path.join(hist_dir, f"{base_filename}.png")
                )
                image_left_path = matched_path
                image_right_path = matched_path
            else:
                image_left_path = rgb_path
                image_right_path = rgb_path

            for transformation_name in transformations:
                # Subcarpeta para cada transformación
                trans_dir = os.path.join(extractor_dir, transformation_name)
                os.makedirs(trans_dir, exist_ok=True)

                # === Ejecutar main.py ===
                image_warped, matches, scores, error = main(
                    imageLeft=image_left_path,
                    imageRight=image_right_path,
                    imageThermal=thermal_path,
                    output_name=base_filename,
                    extractor=extractor,
                    transformation_method=transformation_name,
                    threshold=threshold,
                    usar_histogram_matching=aplicar_histogram_matching,
                )

                # === Guardar métricas en Excel global y por extractor ===
                fieldnames = [
                    "output_name", "extractor", "transformation",
                    "Matches",
                    "NMI_pre", "RMSE_pre", "Error_reprojection", "NMI_post", "RMSE_post", "NRMSE",
                    "NCC_Sobel", "PSNR_Sobel", "SSIM_Sobel",
                    "NCC_Canny", "PSNR_Canny", "SSIM_Canny"
                ]

                # Armamos el diccionario de datos igual que en main
                data = {
                    "output_name": base_filename,
                    "extractor": extractor,
                    "transformation": transformation_name,
                    "Matches": len(matches) if matches is not None else 0,
                    "NMI_pre": None,  # si quieres, puedes recalcular aquí
                    "RMSE_pre": None,
                    "Error_reprojection": error,
                    "NMI_post": None,
                    "RMSE_post": None,
                    "NRMSE": None,
                    "NCC_Sobel": None,
                    "PSNR_Sobel": None,
                    "SSIM_Sobel": None,
                    "NCC_Canny": None,
                    "PSNR_Canny": None,
                    "SSIM_Canny": None,
                }

                # Guardar tanto en global como en por-extractor
                save_metrics_to_excel(global_xlsx, data, fieldnames)
                save_metrics_to_excel(extractor_xlsx, data, fieldnames)

    print(f"\n[INFO] Resultados guardados en {results_root}")


if __name__ == "__main__":
    run_batch()
