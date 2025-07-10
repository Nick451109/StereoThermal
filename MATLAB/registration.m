clc
clear all
close all

orthophoto = imread('LEFT_visible_20241113_141719.png');
figure, imshow(orthophoto)

uregistered = imread('thermal_20241113_141719.png');
figure, imshow(uregistered)


cpselect(uregistered, orthophoto)
pause
mytform = cp2tform(movingPoints, fixedPoints, 'projective');

registered = imtransform(uregistered, mytform, 'XData', [1 size(orthophoto,2)], 'YData', [1 size(orthophoto,1)]);

figure, imshow(registered);
imwrite(registered, 'output_image2.png');  % Guardar como PNG