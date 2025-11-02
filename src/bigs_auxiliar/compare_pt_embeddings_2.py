from pathlib import Path
import torch
import sys

def compare_pt_files(file1, file2):
    """
    Compare two .pt files and verify that the second has double the elements.
    """
    try:
        # Load the tensors
        data1 = torch.load(file1)
        data2 = torch.load(file2)
        
        # Get number of elements
        if isinstance(data1, dict):
            count1 = sum(v.numel() for v in data1.values() if isinstance(v, torch.Tensor))
        else:
            count1 = data1.numel()
        
        if isinstance(data2, dict):
            count2 = sum(v.numel() for v in data2.values() if isinstance(v, torch.Tensor))
        else:
            count2 = data2.numel()
        
        print(f"File 1 ({file1}): {count1} elements")
        print(f"File 2 ({file2}): {count2} elements")
        
        # Check if file2 has double the elements
        if count2 == count1 * 2:
            print("✓ File 2 has exactly double the elements")
            return True
        else:
            print(f"✗ File 2 does NOT have double the elements (ratio: {count2/count1:.2f}x)")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    media_pearson = 0
    media_cosine = 0
    pt_dir = "/media/jorge/MASIVO_PORTATIL_1/features_20x_he_her2/tcga/features_virchow/pt_files"
    pt_files = sorted(Path(pt_dir).glob("*.pt"))

    for pt_file in pt_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pt_file.name}")
        print('='*60)
        
        try:
            data = torch.load(str(pt_file))
            
            if isinstance(data, dict):
                first_vector = next(v for v in data.values() if isinstance(v, torch.Tensor))
            else:
                first_vector = data
            
            vector_flat = first_vector.flatten()
            mid = len(vector_flat) // 2
            first_half = vector_flat[:mid]
            second_half = vector_flat[mid:]
            
            print(f"Total elements: {len(vector_flat)}")
            print(f"First 5 elements: {first_half[:5]}")
            print(f"Last 5 elements: {second_half[:5]}")
            
            pearson_corr = torch.corrcoef(torch.stack([first_half, second_half]))[0, 1]
            print(f"Pearson correlation: {pearson_corr.item():.4f}")
            
            cosine_sim = torch.nn.functional.cosine_similarity(first_half.unsqueeze(0), second_half.unsqueeze(0))
            print(f"Cosine similarity: {cosine_sim.item():.4f}")
            media_cosine += cosine_sim
            media_pearson += pearson_corr
        except Exception as e:
            print(f"Error processing {pt_file.name}: {e}")
    
    print(media_pearson/len(pt_files))
    print(media_cosine/len(pt_files))