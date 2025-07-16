code = '20250714_152902';
% Calibración de cámaras estéreo con imágenes térmicas y visibles
I1 = imread(strcat('captures\temp\50gray\', code, '.png'));
I2 = imread(strcat('captures\temp\50invert\', code, '.png'));

% Calibración de cámaras estéreo con imágenes visibles
% I1 = imread(strcat('captures\visible\right\RIGHT_visible_', code, '.png'));
% I2 = imread(strcat('captures\visible\right\LEFT_visible_', code, '.png'));

% [J1, J2] = rectifyStereoImages(I1, I2, cam_params);
% figure; imshow(stereoAnaglyph(I1, I2)); title('Imágenes rectificadas (anaglifo)');


% Extraer parámetros actuales
cam1_params = cam_params.CameraParameters1;
cam2_params = cam_params.CameraParameters2;

% Rectificar con parámetros originales
[J1_original, J2_original] = rectifyStereoImages(I1, I2, cam_params);

% Calcular el desplazamiento necesario
% 3 cuadros de 20mm = 60mm de desplazamiento
% Necesitamos convertir mm a píxeles
focal_length = cam_params.CameraParameters1.FocalLength(1); % En píxeles
baseline = norm(cam_params.PoseCamera2.Translation); % En mm
pixel_displacement = 42;% (60 * focal_length) / baseline; % Conversión mm a píxeles

% Aplicar traslación horizontal a la imagen derecha
translation_matrix = [1, 0, pixel_displacement; 0, 1, 0; 0, 0, 1];
tform = projtform2d(translation_matrix);
J2_shifted = imwarp(J2_original, tform, 'OutputView', imref2d(size(J2_original)));

% Crear anaglifo corregido
corrected_anaglyph = stereoAnaglyph(J1_original, J2_shifted);

% Mostrar comparación
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1); imshow(stereoAnaglyph(J1_original, J2_original)); title('Original');
subplot(1,2,2); imshow(corrected_anaglyph); title('Corregido con traslación');

% % Función para ajuste fino manual
% function J2_adjusted = fine_tune_alignment(J1, J2, pixel_shift)
%     translation_matrix = [1, 0, pixel_shift; 0, 1, 0; 0, 0, 1];
%     tform = projtform2d(translation_matrix);
%     J2_adjusted = imwarp(J2, tform, 'OutputView', imref2d(size(J2)));
% end

% Prueba con diferentes desplazamientos
% shifts_to_test = [-80, -60, -40, -20, 0, 20, 40, 60, 80];
% figure('Position', [100, 100, 1400, 800]);
% for i = 1:length(shifts_to_test)
%     shift = shifts_to_test(i);
%     J2_test = fine_tune_alignment(J1_original, J2_original, shift);
%     anaglyph_test = stereoAnaglyph(J1_original, J2_test);
% 
%     subplot(3, 3, i);
%     imshow(anaglyph_test);
%     title(sprintf('Shift: %d px', shift));
% end