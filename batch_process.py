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
                trans_dir = os.path.join(extractor_dir, transformation_name)
                os.makedirs(trans_dir, exist_ok=True)

                result = main(
                    imageLeft=image_left_path,
                    imageRight=image_right_path,
                    imageThermal=thermal_path,
                    output_name=base_filename,
                    extractor=extractor,
                    transformation_method=transformation_name,
                    threshold=threshold,
                    usar_histogram_matching=aplicar_histogram_matching,
                    base_dir=trans_dir  # ✅ más ordenado
                )

                if result is None:
                    print(f"[WARN] Se omitió {base_filename} con {transformation_name} ({extractor}) por falta de matches.")
                    continue

                image_warped, matches, scores, metrics_local, data = result
                error = metrics_local.get("Error_reprojection", None)

                # Guardar en Excel global
                save_metrics_to_excel(global_xlsx, data, list(data.keys()))


    print(f"\n[INFO] Resultados guardados en {results_root}")


if __name__ == "__main__":
    run_batch()
