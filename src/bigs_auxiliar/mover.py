import os
import shutil

def mover_archivos(origen, destino):
    if not os.path.exists(destino):
        os.makedirs(destino)
    
    for root, dirs, files in os.walk(origen):
        for file in files:
            ruta_origen = os.path.join(root, file)
            ruta_destino = os.path.join(destino, file)
            shutil.move(ruta_origen, ruta_destino)
            print(f'Movido: {ruta_origen} -> {ruta_destino}')

if __name__ == "__main__":
    directorio_origen = '/media/jorge/Expansion/medicina/patologia_digital/datos/histology/perfil_molecular/publicas/TCGA-BRCA/formalin_fixed_paraffin_embedded'
    directorio_destino = '/media/jorge/Expansion/medicina/patologia_digital/datos/histology/perfil_molecular/publicas/TCGA-BRCA/diagnostico'
    mover_archivos(directorio_origen, directorio_destino)