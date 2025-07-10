
% Crear una estructura con estas variables
stereoParameters.cameraMatrix1 = stereoParamsThermal.CameraParameters1.K;
stereoParameters.distCoeffs1 = [stereoParamsThermal.CameraParameters1.RadialDistortion, stereoParamsThermal.CameraParameters1.TangentialDistortion];
stereoParameters.cameraMatrix2 = stereoParamsThermal.CameraParameters2.K;
stereoParameters.distCoeffs2 = [stereoParamsThermal.CameraParameters2.RadialDistortion, stereoParamsThermal.CameraParameters2.TangentialDistortion];
stereoParameters.imageSize = stereoParamsThermal.CameraParameters1.ImageSize;
stereoParameters.stereoR = stereoParamsThermal.PoseCamera2.R;
stereoParameters.stereoT = stereoParamsThermal.PoseCamera2.Translation;
stereoParameters.flCamera1 = stereoParamsThermal.CameraParameters1.FocalLength;
stereoParameters.flCamera2 = stereoParamsThermal.CameraParameters2.FocalLength;

% Codificar la estructura en JSON
jsonData = jsonencode(stereoParameters);

% Guardar el JSON en un archivo
fid = fopen('ParamsThermal.json', 'w');
if fid == -1, error('No se puede crear el archivo.'); end
fwrite(fid, jsonData, 'char');
fclose(fid);

%% 

% Codificar la estructura en JSON
jsonData = jsonencode(stereoParamsThermal);

% Guardar el JSON en un archivo
fid = fopen('including_Y_rotation_random_stereoParamrs.json', 'w');
if fid == -1, error('No se puede crear el archivo.'); end
fwrite(fid, jsonData, 'char');
fclose(fid);