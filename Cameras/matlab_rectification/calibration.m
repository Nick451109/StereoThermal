% Calibración estéreo automatizada entre cámaras visible y térmica
% Versión mejorada con directorios automáticos

%% CONFIGURACIÓN DE DIRECTORIOS
% Solo necesitas modificar estos directorios base
baseCalibrationDir = 'captures/';
baseCaptureDir     = 'captures/';
baseOutputDir      = 'captures/';

%% PARTE 1: CALIBRACIÓN ESTÉREO

% Directorios automáticos para calibración
leftCalibDir = fullfile(baseCalibrationDir, 'visible/left');
thermalCalibDir = fullfile(baseCalibrationDir, 'inverse');

% Obtener automáticamente todas las imágenes de calibración
leftCalibFiles = dir(fullfile(leftCalibDir, '*.png'));
thermalCalibFiles = dir(fullfile(thermalCalibDir, '*.png'));

% Verificar que existan imágenes
if isempty(leftCalibFiles) || isempty(thermalCalibFiles)
    error('No se encontraron imágenes de calibración en los directorios especificados.');
end

% Verificar que haya el mismo número de imágenes
if numel(leftCalibFiles) ~= numel(thermalCalibFiles)
    error('El número de imágenes de calibración no coincide entre las carpetas.');
end

% Generar rutas completas automáticamente
imageFileNames1 = cell(1, numel(leftCalibFiles));
imageFileNames2 = cell(1, numel(thermalCalibFiles));

for i = 1:numel(leftCalibFiles)
    imageFileNames1{i} = fullfile(leftCalibDir, leftCalibFiles(i).name);
    imageFileNames2{i} = fullfile(thermalCalibDir, thermalCalibFiles(i).name);
end

fprintf('Encontradas %d imágenes de calibración en cada carpeta.\n', numel(leftCalibFiles));

% Detectar patrón de calibración en imágenes
detector = vision.calibration.stereo.CheckerboardDetector();
minCornerMetric = 0.150000;
[imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames1, imageFileNames2, 'MinCornerMetric', minCornerMetric);

% Generar coordenadas del mundo para el patrón
squareSize = 20.000000;  % en milímetros
worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% Leer una imagen para obtener dimensiones
I1 = imread(imageFileNames1{1});
[mrows, ncols, ~] = size(I1);

% Calibrar la cámara
[stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% Mostrar errores de reproyección
h1 = figure; showReprojectionErrors(stereoParams);
title('Errores de Reproyección');

% Visualizar ubicaciones del patrón
h2 = figure; showExtrinsics(stereoParams, 'CameraCentric');
title('Parámetros Extrínsecos');

% Mostrar errores de estimación de parámetros
displayErrors(estimationErrors, stereoParams);

fprintf('Calibración completada. Imágenes utilizadas: %d de %d\n', sum(pairsUsed), numel(imageFileNames1));

%% PARTE 2: RECTIFICACIÓN DE NUEVAS IMÁGENES

% Directorios automáticos para rectificación
inputDir1 = fullfile(baseCaptureDir, 'visible', 'left');
inputDir2 = fullfile(baseCaptureDir, 'thermal');
outputDir1 = fullfile(baseOutputDir, 'left_thermal_rectified');
outputDir2 = fullfile(baseOutputDir, 'thermal_rectified');

% Crear carpetas de salida si no existen
if ~exist(outputDir1, 'dir')
    mkdir(outputDir1);
    fprintf('Creada carpeta: %s\n', outputDir1);
end
if ~exist(outputDir2, 'dir')
    mkdir(outputDir2);
    fprintf('Creada carpeta: %s\n', outputDir2);
end

% Obtener lista de imágenes automáticamente
imageFiles1 = dir(fullfile(inputDir1, '*.png'));
imageFiles2 = dir(fullfile(inputDir2, '*.png'));

% Verificar que existan imágenes
if isempty(imageFiles1) || isempty(imageFiles2)
    error('No se encontraron imágenes en los directorios de entrada.');
end

% Verificar que haya la misma cantidad de imágenes
if numel(imageFiles1) ~= numel(imageFiles2)
    warning('El número de imágenes en las carpetas no coincide. Procesando las comunes.');
    % Tomar el mínimo número de imágenes
    numImages = min(numel(imageFiles1), numel(imageFiles2));
    imageFiles1 = imageFiles1(1:numImages);
    imageFiles2 = imageFiles2(1:numImages);
end

fprintf('Iniciando rectificación de %d pares de imágenes...\n', numel(imageFiles1));

% Rectificar cada par de imágenes
for i = 1:numel(imageFiles1)
    try
        % Leer imágenes originales
        inputFile1 = fullfile(inputDir1, imageFiles1(i).name);
        inputFile2 = fullfile(inputDir2, imageFiles2(i).name);
        
        I1 = imread(inputFile1);
        I2 = imread(inputFile2);

        % Validar y ajustar dimensiones
        if size(I1, 1) ~= size(I2, 1) || size(I1, 2) ~= size(I2, 2)
            fprintf('Ajustando dimensiones de la imagen %d: %s y %s\n', i, imageFiles1(i).name, imageFiles2(i).name);
            I2 = imresize(I2, [size(I1, 1), size(I1, 2)]);
        end

        % Validar número de canales
        if size(I1, 3) ~= size(I2, 3)
            fprintf('Ajustando canales de la imagen %d\n', i);
            if size(I1, 3) == 1
                I1 = cat(3, I1, I1, I1);
            end
            if size(I2, 3) == 1
                I2 = cat(3, I2, I2, I2);
            end
        end

        % Validar tipo de datos
        if ~isa(I1, 'uint8')
            I1 = im2uint8(I1);
        end
        if ~isa(I2, 'uint8')
            I2 = im2uint8(I2);
        end

        % Rectificar imágenes
        [rectifiedI1, rectifiedI2] = rectifyStereoImages(I1, I2, stereoParams);
        
        % Ajustar resolución a la original
        targetSize = [size(I1, 2), size(I1, 1)];
        rectifiedI1 = imresize(rectifiedI1, targetSize);
        rectifiedI2 = imresize(rectifiedI2, targetSize);

        % Girar las imágenes rectificadas 90 grados en sentido horario
        rectifiedI1 = imrotate(rectifiedI1, 90);
        rectifiedI2 = imrotate(rectifiedI2, 90);
            
        % Construir las rutas de salida
        outputFile1 = fullfile(outputDir1, imageFiles1(i).name);
        outputFile2 = fullfile(outputDir2, imageFiles2(i).name);
        
        % Guardar imágenes rectificadas
        imwrite(rectifiedI1, outputFile1);
        imwrite(rectifiedI2, outputFile2);
        
        % Mensaje de progreso cada 10 imágenes
        if mod(i, 10) == 0 || i == numel(imageFiles1)
            fprintf('Progreso: %d/%d imágenes procesadas (%.1f%%)\n', i, numel(imageFiles1), (i/numel(imageFiles1))*100);
        end
        
    catch ME
        fprintf('Error procesando imagen %d (%s): %s\n', i, imageFiles1(i).name, ME.message);
        continue;
    end
end

fprintf('\n¡Rectificación completada! Todas las imágenes han sido procesadas y guardadas.\n');
fprintf('Imágenes rectificadas guardadas en:\n');
fprintf('  - %s\n', outputDir1);
fprintf('  - %s\n', outputDir2);

%% FUNCIÓN AUXILIAR: Mostrar estadísticas de calibración
function showCalibrationStats(stereoParams, pairsUsed)
    fprintf('\n=== ESTADÍSTICAS DE CALIBRACIÓN ===\n');
    fprintf('Imágenes utilizadas: %d\n', sum(pairsUsed));
    fprintf('Error medio de reproyección: %.4f píxeles\n', stereoParams.MeanReprojectionError);
    fprintf('Tamaño del cuadrado: %.2f mm\n', stereoParams.WorldUnits);
    fprintf('Resolución de imagen: %d x %d\n', stereoParams.ImageSize(2), stereoParams.ImageSize(1));
    fprintf('=====================================\n\n');
end

% Llamar a la función de estadísticas
showCalibrationStats(stereoParams, pairsUsed);