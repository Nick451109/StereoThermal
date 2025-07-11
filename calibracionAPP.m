I1 = imread('C:/StereoThermal/captures/inverse/thermal_20250710_110814.png');
I2 = imread('C:/StereoThermal/gray.png');

% Ejemplo: Rectificar y visualizar un par de imágenes
[J1, J2] = rectifyStereoImages(I1, I2, cam_params);
figure; imshow(stereoAnaglyph(J1, J2)); title('Imágenes rectificadas (anaglifo)');