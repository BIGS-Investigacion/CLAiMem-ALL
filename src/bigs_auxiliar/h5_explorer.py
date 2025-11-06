import h5py
import numpy as np
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
file1 = ".features_20x/tcga/features_virchow/h5_files/TCGA-WT-AB44-01A-01-TS1.B6C0EEDB-E5B9-4B0D-8599-23879A0419EB.h5"

file2 = "old/.features_20x/features_virchow/TCGA-WT-AB44-01A-01-TS1.B6C0EEDB-E5B9-4B0D-8599-23879A0419EB.h5"

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def compare_datasets(ds1, ds2, name, tolerance=1e-6):
    """
    Compara dos datasets de HDF5
    
    Returns:
        dict con resultados de comparación
    """
    results = {
        'name': name,
        'same_shape': False,
        'same_dtype': False,
        'same_values': False,
        'max_diff': None,
        'num_different': None
    }
    
    # Comparar shapes
    if ds1.shape != ds2.shape:
        results['shape1'] = ds1.shape
        results['shape2'] = ds2.shape
        return results
    
    results['same_shape'] = True
    results['shape'] = ds1.shape
    
    # Comparar dtypes
    if ds1.dtype != ds2.dtype:
        results['dtype1'] = ds1.dtype
        results['dtype2'] = ds2.dtype
        return results
    
    results['same_dtype'] = True
    results['dtype'] = ds1.dtype
    
    # Comparar valores
    data1 = ds1[:]
    data2 = ds2[:] 

    # Eliminar datos NaN para comparación
    if np.isnan(data1).any() or np.isnan(data2).any():
        mask = ~(np.isnan(data1) | np.isnan(data2))
        data1 = data1[mask]
        data2 = data2[mask]
    
    if np.array_equal(data1, data2):
        results['same_values'] = True
        results['max_diff'] = 0.0
        results['num_different'] = 0
    else:
        # Calcular diferencias
        max_1= np.max(data1)
        max_2= np.max(data2)
        min_1= np.min(data1)
        min_2= np.min(data2)
        diff = np.abs(data1 - data2)
        results['max_diff'] = np.max(diff)
        results['mean_diff'] = np.mean(diff)
        results['num_different'] = np.sum(diff > tolerance)
        results['pct_different'] = (results['num_different'] / data1.size) * 100
        
        # Son "iguales" si la diferencia es pequeña
        results['same_values'] = results['max_diff'] < tolerance
    
    return results

# ============================================================================
# COMPARACIÓN PRINCIPAL
# ============================================================================
print("="*80)
print("HDF5 FILE COMPARISON")
print("="*80)

print(f"\nFile 1: {file1}")
print(f"File 2: {file2}")

# Verificar existencia
if not os.path.exists(file1):
    print(f"\n❌ File 1 does not exist!")
    exit(1)

if not os.path.exists(file2):
    print(f"\n❌ File 2 does not exist!")
    exit(1)

# Comparar tamaños
size1 = os.path.getsize(file1)
size2 = os.path.getsize(file2)

print(f"\nFile sizes:")
print(f"  File 1: {size1:,} bytes ({size1/1024/1024:.2f} MB)")
print(f"  File 2: {size2:,} bytes ({size2/1024/1024:.2f} MB)")
print(f"  Difference: {abs(size1 - size2):,} bytes ({abs(size1 - size2)/1024:.1f} KB)")

# Abrir archivos
with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
    
    # Comparar keys
    keys1 = set(f1.keys())
    keys2 = set(f2.keys())
    
    print("\n" + "="*80)
    print("STRUCTURE COMPARISON")
    print("="*80)
    
    print(f"\nKeys in File 1: {sorted(keys1)}")
    print(f"Keys in File 2: {sorted(keys2)}")
    
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    if only_in_1:
        print(f"\n⚠️ Keys ONLY in File 1: {sorted(only_in_1)}")
    
    if only_in_2:
        print(f"\n⚠️ Keys ONLY in File 2: {sorted(only_in_2)}")
    
    if common_keys == keys1 == keys2:
        print(f"\n✅ Both files have the same keys: {sorted(common_keys)}")
    
    # Comparar datasets comunes
    print("\n" + "="*80)
    print("DATASET COMPARISON")
    print("="*80)
    
    all_same = True
    
    for key in sorted(common_keys):
        ds1 = f1[key]
        ds2 = f2[key]
        
        # Solo comparar datasets, no grupos
        if not isinstance(ds1, h5py.Dataset) or not isinstance(ds2, h5py.Dataset):
            print(f"\n⚠️ '{key}' is not a dataset in both files, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Dataset: '{key}'")
        print(f"{'='*80}")
        
        results = compare_datasets(ds1, ds2, key)
        
        # Mostrar resultados
        if not results['same_shape']:
            print(f"❌ DIFFERENT SHAPES:")
            print(f"   File 1: {results['shape1']}")
            print(f"   File 2: {results['shape2']}")
            all_same = False
            continue
        
        print(f"✅ Shape: {results['shape']}")
        
        if not results['same_dtype']:
            print(f"❌ DIFFERENT DTYPES:")
            print(f"   File 1: {results['dtype1']}")
            print(f"   File 2: {results['dtype2']}")
            all_same = False
            continue
        
        print(f"✅ Dtype: {results['dtype']}")
        
        if results['same_values']:
            print(f"✅ Values: IDENTICAL (max diff: {results['max_diff']})")
        else:
            print(f"❌ Values: DIFFERENT")
            print(f"   Max difference: {results['max_diff']:.6e}")
            print(f"   Mean difference: {results['mean_diff']:.6e}")
            print(f"   Elements different: {results['num_different']:,} ({results['pct_different']:.2f}%)")
            print(f"   Total elements: {ds1.size:,}")
            all_same = False
            
            # Mostrar algunos ejemplos de diferencias
            if ds1.ndim <= 2:
                data1 = ds1[:]
                data2 = ds2[:]
                diff = np.abs(data1 - data2)
                
                # Índices con mayor diferencia
                if ds1.ndim == 1:
                    top_indices = np.argsort(diff)[-5:][::-1]
                    print(f"\n   Top 5 differences:")
                    for idx in top_indices:
                        print(f"     [{idx}]: {data1[idx]:.6f} vs {data2[idx]:.6f} (diff: {diff[idx]:.6e})")
                
                elif ds1.ndim == 2:
                    flat_diff = diff.flatten()
                    top_flat_indices = np.argsort(flat_diff)[-5:][::-1]
                    print(f"\n   Top 5 differences:")
                    for flat_idx in top_flat_indices:
                        i, j = np.unravel_index(flat_idx, ds1.shape)
                        print(f"     [{i}, {j}]: {data1[i,j]:.6f} vs {data2[i,j]:.6f} (diff: {diff[i,j]:.6e})")
    
    # Comparar atributos
    print("\n" + "="*80)
    print("ATTRIBUTES COMPARISON")
    print("="*80)
    
    attrs1 = dict(f1.attrs)
    attrs2 = dict(f2.attrs)
    
    if attrs1 or attrs2:
        if attrs1 == attrs2:
            print("✅ Root attributes are identical")
            if attrs1:
                for k, v in attrs1.items():
                    print(f"   {k}: {v}")
        else:
            print("❌ Root attributes differ:")
            all_attrs = set(attrs1.keys()) | set(attrs2.keys())
            for attr in sorted(all_attrs):
                v1 = attrs1.get(attr, "NOT PRESENT")
                v2 = attrs2.get(attr, "NOT PRESENT")
                if v1 == v2:
                    print(f"   ✅ {attr}: {v1}")
                else:
                    print(f"   ❌ {attr}:")
                    print(f"      File 1: {v1}")
                    print(f"      File 2: {v2}")
    else:
        print("No root attributes in either file")
    
    # Comparar atributos de datasets
    for key in sorted(common_keys):
        if isinstance(f1[key], h5py.Dataset) and isinstance(f2[key], h5py.Dataset):
            attrs1 = dict(f1[key].attrs)
            attrs2 = dict(f2[key].attrs)
            
            if attrs1 or attrs2:
                if attrs1 != attrs2:
                    print(f"\n❌ Attributes differ for dataset '{key}':")
                    all_attrs = set(attrs1.keys()) | set(attrs2.keys())
                    for attr in sorted(all_attrs):
                        v1 = attrs1.get(attr, "NOT PRESENT")
                        v2 = attrs2.get(attr, "NOT PRESENT")
                        if v1 != v2:
                            print(f"   {attr}: {v1} vs {v2}")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

if all_same and not only_in_1 and not only_in_2:
    print("\n✅ FILES ARE IDENTICAL")
    print("   Same structure, same data, same attributes")
else:
    print("\n❌ FILES ARE DIFFERENT")
    
    if only_in_1 or only_in_2:
        print("   - Different keys/datasets")
    
    if not all_same:
        print("   - Different values in one or more datasets")
    
    print("\nPOSSIBLE REASONS:")
    print("   1. Different compression settings (HDF5)")
    print("   2. Different extraction parameters")
    print("   3. Different processing pipelines")
    print("   4. Data corruption")
    print("   5. Intentionally different versions")

print("\n" + "="*80)