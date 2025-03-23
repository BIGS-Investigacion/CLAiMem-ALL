import cptac


# Obtener los niveles proteómicos de MKI67
def get_cptac_ki67_expression(brca, sample_id):
    proteomics = brca.get_dataframe('proteomics', 'umich')
    
    # Asegurar que MKI67 está en la tabla
    if "MKI67" in proteomics.columns:
        if sample_id in proteomics.index:
            return f"Expresión proteómica de Ki-67 en {sample_id}: {proteomics.loc[sample_id, 'MKI67']}"
        else:
            return f"Muestra {sample_id} no encontrada en CPTAC-BRCA."
    else:
        return "Ki-67 no está disponible en los datos proteómicos de CPTAC."


# Descargar y cargar datos de CPTAC-BRCA
#cptac.download(source="umich", dtype="proteomics", cancer="brca")  
#print(cptac.get_source_options())

brca = cptac.Brca()
#print(brca.list_data_sources())

#  Ejecutar para una muestra específica
sample_id = "01BR008"  # Cambia esto por tu identificador real
ki67_protein = get_cptac_ki67_expression(brca, sample_id)
print(ki67_protein)

