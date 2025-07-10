import os

def listar_rutas(directorio):
    rutas = []
    for raiz, directorios, archivos in os.walk(directorio):
        for archivo in archivos:
            ruta_completa = os.path.join(raiz, archivo)
            # Escapar las barras invertidas para Windows
            ruta_completa = ruta_completa.replace('\\', '\\\\')
            rutas.append(ruta_completa)
    
    # Imprimir todas las rutas separadas por comas
    print(', '.join(f"'{r}'" for r in rutas))

if __name__ == "__main__":
    listar_rutas('captures/visible/right/')