# Imagen base con Python 3.10
FROM python:3.10-slim

# Instala utilidades del sistema necesarias para dependencias como pypylon, OpenCV y otras
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de tu proyecto
COPY . .

# Instala seaborn, stackview y otras dependencias adicionales
RUN pip install --upgrade pip

# Instala torch compatible con CUDA 11.8 (cu118)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instala dependencias del requirements.txt
RUN pip install -r requirements.txt

# Instala librerías adicionales no incluidas en requirements.txt
RUN pip install seaborn stackview

# Comando por defecto
CMD ["bash"]
