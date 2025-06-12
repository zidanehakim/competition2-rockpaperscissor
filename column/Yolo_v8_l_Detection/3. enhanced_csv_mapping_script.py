import pandas as pd
import csv
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Any
import time

def map_large_csv_files(
    classification_csv="classification_results.csv",
    extraction_csv="extracted_boxes_report.csv", 
    output_csv="mapped_results.csv",
    batch_size=1000,
    verbose=True
):
    """
    Enhanced CSV mapping for large datasets with multiple similar entries
    
    Args:
        classification_csv (str): Path to classification results CSV
        extraction_csv (str): Path to extraction report CSV
        output_csv (str): Path for output mapped CSV
        batch_size (int): Number of records to process in each batch
        verbose (bool): Print progress information
    """
    
    start_time = time.time()
    
    if verbose:
        print("🚀 ENHANCED CSV MAPPING TOOL (Large Dataset Support)")
        print("=" * 60)
    
    # Step 1: Load and validate CSV files
    if verbose:
        print("📄 Loading CSV files...")
    
    try:
        df_classification = pd.read_csv(classification_csv)
        df_extraction = pd.read_csv(extraction_csv)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading CSV files: {e}")
        return
    
    if verbose:
        print(f"   Classification CSV: {len(df_classification):,} rows")
        print(f"   Extraction CSV: {len(df_extraction):,} rows")
    
    # Step 2: Build efficient lookup structures
    if verbose:
        print("🔍 Building lookup structures...")
    
    classification_lookup = build_classification_lookup(df_classification, verbose)
    extraction_data = process_extraction_data_efficient(df_extraction, verbose)
    
    # Step 3: Process mapping in batches
    if verbose:
        print("🔗 Processing mappings...")
    
    mapped_results = process_mappings_batch(
        extraction_data, 
        classification_lookup, 
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Step 4: Save results
    if verbose:
        print("💾 Saving results...")
    
    save_results_efficient(mapped_results, output_csv, verbose)
    
    end_time = time.time()
    if verbose:
        print(f"\n✅ Mapping completed in {end_time - start_time:.2f} seconds!")
        print(f"📁 Output: {output_csv}")
        print(f"📊 Generated: {len(mapped_results):,} mapped records")

def build_classification_lookup(df_classification: pd.DataFrame, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Build efficient lookup dictionary for classification data
    
    Args:
        df_classification: Classification DataFrame
        verbose: Print progress
        
    Returns:
        Dictionary with image names as keys and classification data as values
    """
    
    lookup = {}
    
    for _, row in df_classification.iterrows():
        image_name = str(row['Image_Name']).strip()
        
        lookup[image_name] = {
            'crack_class': str(row.get('Crack_Class', '')).strip(),
            'crack_name': str(row.get('Crack_Name', '')).strip(), 
            'crack_confidence': str(row.get('Crack_Confidence', '')).strip(),
            'damage_class': str(row.get('Damage_Class', '')).strip(),
            'damage_name': str(row.get('Damage_Name', '')).strip(),
            'damage_confidence': str(row.get('Damage_Confidence', '')).strip(),
            'status': str(row.get('Status', '')).strip()
        }
    
    if verbose:
        print(f"   Built classification lookup: {len(lookup):,} entries")
    
    return lookup

def process_extraction_data_efficient(df_extraction: pd.DataFrame, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Process extraction data efficiently for large datasets
    
    Args:
        df_extraction: Extraction DataFrame
        verbose: Print progress
        
    Returns:
        List of processed extraction records
    """
    
    extraction_records = []
    
    for _, row in df_extraction.iterrows():
        main_image = str(row['Input Id']).strip()
        detection_string = str(row.get('Detection', '')).strip()
        
        # Parse detection IDs
        if detection_string and detection_string != 'nan':
            detection_ids = [d.strip() for d in detection_string.split(',') if d.strip()]
        else:
            detection_ids = []
        
        # Extract sub-image data efficiently
        sub_images_data = extract_sub_images_data(row, df_extraction.columns)
        
        if sub_images_data:  # Only process if there are sub-images
            record = {
                'main_image': main_image,
                'detection_ids': detection_ids,
                'sub_images_data': sub_images_data
            }
            extraction_records.append(record)
    
    if verbose:
        print(f"   Processed extraction data: {len(extraction_records):,} main images")
        total_sub_images = sum(len(rec['sub_images_data']) for rec in extraction_records)
        print(f"   Total sub-images: {total_sub_images:,}")
    
    return extraction_records

def extract_sub_images_data(row: pd.Series, columns: pd.Index) -> List[Dict[str, str]]:
    """
    Extract sub-image data from extraction row
    
    Args:
        row: DataFrame row
        columns: DataFrame columns
        
    Returns:
        List of sub-image data dictionaries
    """
    
    sub_images_data = []
    
    # Find all id_X columns and their corresponding detect_X columns
    id_columns = [col for col in columns if col.startswith('id_')]
    
    for id_col in sorted(id_columns, key=lambda x: int(x.split('_')[1])):
        if pd.notna(row[id_col]) and str(row[id_col]).strip():
            sub_image_name = str(row[id_col]).strip()
            
            # Get corresponding detect_X column
            id_num = id_col.split('_')[1]
            detect_col = f"detect_{id_num}"
            
            if detect_col in columns and pd.notna(row[detect_col]):
                detect_class = str(row[detect_col]).strip()
            else:
                detect_class = ""
            
            sub_images_data.append({
                'sub_image_name': sub_image_name,
                'detect_class': detect_class,
                'sub_image_id': Path(sub_image_name).stem if sub_image_name.endswith('.jpg') else sub_image_name
            })
    
    return sub_images_data

def process_mappings_batch(
    extraction_records: List[Dict],
    classification_lookup: Dict[str, Dict],
    batch_size: int = 1000,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Process mappings in batches for memory efficiency
    
    Args:
        extraction_records: List of extraction records
        classification_lookup: Classification lookup dictionary
        batch_size: Number of records per batch
        verbose: Print progress
        
    Returns:
        List of mapped records
    """
    
    mapped_results = []
    total_records = len(extraction_records)
    missing_classifications = set()
    record_id = 1
    
    for i in range(0, total_records, batch_size):
        batch_end = min(i + batch_size, total_records)
        batch_records = extraction_records[i:batch_end]
        
        if verbose:
            print(f"   Processing batch {i//batch_size + 1}: records {i+1}-{batch_end} of {total_records}")
        
        for extraction_record in batch_records:
            main_image = extraction_record['main_image']
            detection_ids = extraction_record['detection_ids']
            sub_images_data = extraction_record['sub_images_data']
            
            # Collect classification data for each sub-image
            crack_classes = []
            damage_classes = []
            overall_ids = []
            found_classifications = 0
            
            for sub_data in sub_images_data:
                sub_image_name = sub_data['sub_image_name']
                sub_image_id = sub_data['sub_image_id']
                overall_ids.append(sub_image_id)
                
                if sub_image_name in classification_lookup:
                    class_data = classification_lookup[sub_image_name]
                    crack_classes.append(class_data['crack_class'])
                    damage_classes.append(class_data['damage_class'])
                    found_classifications += 1
                else:
                    crack_classes.append("")
                    damage_classes.append("")
                    missing_classifications.add(sub_image_name)
            
            # Create mapped record
            mapped_record = {
                'id': record_id,
                'image_id': main_image,
                'overall_ids': ','.join(overall_ids),
                'detection': ','.join(detection_ids),
                'crack_classes': ','.join(crack_classes),
                'damage_classes': ','.join(damage_classes),
                'sub_images_count': len(sub_images_data),
                'found_classifications': found_classifications,
                'missing_classifications': len(sub_images_data) - found_classifications
            }
            
            mapped_results.append(mapped_record)
            record_id += 1
    
    if verbose and missing_classifications:
        print(f"   ⚠️  Warning: {len(missing_classifications)} sub-images missing classification data")
        if len(missing_classifications) <= 10:
            print(f"      Missing: {', '.join(list(missing_classifications)[:10])}")
        else:
            print(f"      First 10 missing: {', '.join(list(missing_classifications)[:10])}")
    
    return mapped_results

def save_results_efficient(mapped_results: List[Dict], output_csv: str, verbose: bool = True) -> None:
    """
    Save results efficiently for large datasets
    
    Args:
        mapped_results: List of mapped records
        output_csv: Output file path
        verbose: Print progress
    """
    
    if not mapped_results:
        print("❌ No data to save")
        return
    
    # Define headers (excluding metadata columns)
    headers = ['id', 'image_id', 'overall_ids', 'detection', 'crack_classes', 'damage_classes']
    
    # Write in chunks for memory efficiency
    chunk_size = 1000
    total_records = len(mapped_results)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for i in range(0, total_records, chunk_size):
            chunk_end = min(i + chunk_size, total_records)
            chunk = mapped_results[i:chunk_end]
            
            # Filter out metadata columns for output
            filtered_chunk = [
                {key: record[key] for key in headers if key in record}
                for record in chunk
            ]
            
            writer.writerows(filtered_chunk)
            
            if verbose and total_records > chunk_size:
                print(f"   Saved chunk: {i+1}-{chunk_end} of {total_records}")

def analyze_data_quality(
    classification_csv: str,
    extraction_csv: str,
    sample_size: int = 1000
) -> None:
    """
    Analyze data quality and provide insights
    
    Args:
        classification_csv: Path to classification CSV
        extraction_csv: Path to extraction CSV  
        sample_size: Number of records to analyze
    """
    
    print("🔍 DATA QUALITY ANALYSIS")
    print("=" * 40)
    
    try:
        # Load samples
        df_class = pd.read_csv(classification_csv, nrows=sample_size)
        df_extract = pd.read_csv(extraction_csv, nrows=sample_size)
        
        print(f"\n📊 Classification CSV Analysis (first {len(df_class)} rows):")
        print(f"   Total rows: {len(df_class):,}")
        print(f"   Columns: {list(df_class.columns)}")
        print(f"   Unique images: {df_class['Image_Name'].nunique():,}")
        print(f"   Missing values per column:")
        for col in df_class.columns:
            missing = df_class[col].isna().sum()
            if missing > 0:
                print(f"      {col}: {missing}")
        
        print(f"\n📊 Extraction CSV Analysis (first {len(df_extract)} rows):")
        print(f"   Total rows: {len(df_extract):,}")
        print(f"   Columns: {list(df_extract.columns)}")
        print(f"   Unique main images: {df_extract['Input Id'].nunique():,}")
        
        # Analyze sub-image patterns
        id_columns = [col for col in df_extract.columns if col.startswith('id_')]
        print(f"   Max sub-images per main image: {len(id_columns)}")
        
        # Sample data patterns
        print(f"\n📋 Sample Data Patterns:")
        if len(df_class) > 0:
            sample_class = df_class.iloc[0]
            print(f"   Sample classification: {sample_class['Image_Name']} -> Class: {sample_class.get('Crack_Class', 'N/A')}")
        
        if len(df_extract) > 0:
            sample_extract = df_extract.iloc[0]
            print(f"   Sample extraction: {sample_extract['Input Id']} -> Detections: {sample_extract.get('Detection', 'N/A')}")
        
        print("\n✅ Data quality analysis completed!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

def create_large_sample_data(
    num_main_images: int = 100,
    avg_sub_images_per_main: int = 5,
    output_classification: str = "large_classification.csv",
    output_extraction: str = "large_extraction.csv"
) -> None:
    """
    Create large sample datasets for testing
    
    Args:
        num_main_images: Number of main images to create
        avg_sub_images_per_main: Average sub-images per main image
        output_classification: Output classification CSV
        output_extraction: Output extraction CSV
    """
    
    print(f"🏗️  Creating large sample data...")
    print(f"   Main images: {num_main_images}")
    print(f"   Avg sub-images per main: {avg_sub_images_per_main}")
    
    # Generate classification data
    classification_data = []
    extraction_data = []
    
    record_id = 1
    
    for main_id in range(1, num_main_images + 1):
        # Random number of sub-images (varying between 1 and 2*avg)
        num_sub_images = np.random.randint(1, avg_sub_images_per_main * 2)
        
        main_image_name = f"{main_id}.jpg"
        detection_ids = []
        sub_image_names = []
        detect_classes = []
        
        # Generate sub-images
        for sub_id in range(1, num_sub_images + 1):
            sub_image_name = f"{main_id}_{sub_id}.jpg"
            
            # Random classification data
            crack_class = np.random.choice([0, 1, 2, 6])
            damage_class = np.random.choice(['A', 'B', 'C'])
            crack_names = {0: 'No_Crack', 1: 'Hairline_Crack', 2: 'Minor_Crack', 6: 'Severe_Crack'}
            damage_names = {'A': 'Light_Damage', 'B': 'Moderate_Damage', 'C': 'Severe_Damage'}
            
            classification_data.append([
                record_id,
                sub_image_name,
                crack_class,
                crack_names[crack_class],
                round(np.random.uniform(0.7, 1.0), 3),
                damage_class,
                damage_names[damage_class],
                round(np.random.uniform(0.8, 1.0), 3),
                "Success"
            ])
            
            detection_ids.append(str(crack_class))
            sub_image_names.append(sub_image_name)
            detect_classes.append(crack_class)
            record_id += 1
        
        # Create extraction record
        extraction_record = [main_image_name, ','.join(detection_ids)]
        
        # Add sub-image columns (up to 12 pairs)
        for i in range(12):
            if i < len(sub_image_names):
                extraction_record.extend([sub_image_names[i], detect_classes[i]])
            else:
                extraction_record.extend(["", ""])
        
        extraction_data.append(extraction_record)
    
    # Save classification CSV
    classification_headers = ["ID", "Image_Name", "Crack_Class", "Crack_Name", "Crack_Confidence", 
                             "Damage_Class", "Damage_Name", "Damage_Confidence", "Status"]
    
    with open(output_classification, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(classification_headers)
        writer.writerows(classification_data)
    
    # Save extraction CSV
    extraction_headers = ["Input Id", "Detection"]
    for i in range(1, 13):
        extraction_headers.extend([f"id_{i}", f"detect_{i}"])
    
    with open(output_extraction, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(extraction_headers)
        writer.writerows(extraction_data)
    
    print(f"✅ Large sample data created:")
    print(f"   {output_classification}: {len(classification_data):,} classification records")
    print(f"   {output_extraction}: {len(extraction_data):,} extraction records")

# Configuration and main execution
if __name__ == "__main__":
    # Configuration for large datasets
    CONFIG = {
        "classification_csv": "/root/ak/results_detailed.csv",
        "extraction_csv": "/root/ak/extracted_boxes_report.csv", 
        "output_csv": "mapped_results.csv",
        "batch_size": 1000,  # Adjust based on memory
        "verbose": True
    }
    
    print("🚀 ENHANCED CSV MAPPING TOOL")
    print("=" * 40)
    
    # Uncomment to create large sample data for testing
    # create_large_sample_data(num_main_images=500, avg_sub_images_per_main=4)
    
    # Uncomment to analyze data quality before mapping
    # analyze_data_quality(CONFIG["classification_csv"], CONFIG["extraction_csv"])
    
    # Run enhanced mapping
    map_large_csv_files(**CONFIG)
    
    print("\n🎉 Enhanced CSV mapping completed!")

# Example usage for very large datasets
def process_huge_dataset():
    """Example for processing very large datasets"""
    map_large_csv_files(
        classification_csv="huge_classification.csv",
        extraction_csv="huge_extraction.csv",
        output_csv="huge_mapped_results.csv",
        batch_size=5000,  # Larger batches for better performance
        verbose=True
    )
