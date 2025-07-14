I1 = imread('C:\StereoThermal\captures\visible\left\LEFT_visible_20250710_110814.png');
I2 = imread('C:\StereoThermal\captures\visible\right\RIGHT_visible_20250710_110814.png');

% Ejemplo: Rectificar y visualizar un par de imágenes
[J1, J2] = rectifyStereoImages(I2, I1, cam_params);
figure; imshow(stereoAnaglyph(J1, J2)); title('Imágenes rectificadas (anaglifo)');

