
% Crear una estructura con estas variables
parameters.cameraMatrix = cameraParamsRightOnly.K;
parameters.distCoeffs = [cameraParamsRightOnly.RadialDistortion, cameraParamsRightOnly.TangentialDistortion];
parameters.imageSize = cameraParamsRightOnly.ImageSize;
parameters.stereoR = cameraParamsRightOnly.RotationVectors;
parameters.stereoT = [0,0,0];
parameters.flCamera = cameraParamsRightOnly.FocalLength;

% Codificar la estructura en JSON
jsonData = jsonencode(parameters);

% Guardar el JSON en un archivo
fid = fopen('ParamsRightOnly.json', 'w');
if fid == -1, error('No se puede crear el archivo.'); end
fwrite(fid, jsonData, 'char');
fclose(fid);