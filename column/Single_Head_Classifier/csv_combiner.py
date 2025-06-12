#!/usr/bin/env python3
"""
CSV Combiner for Crack and Damage Predictions
Combines crack sub-image predictions with main damage image predictions
"""

import pandas as pd
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import os

class CrackDamageCombiner:
    def __init__(self, crack_csv: str, damage_csv: str, output_csv: str):
        """
        Initialize the combiner
        
        Args:
            crack_csv: Path to crack predictions CSV
            damage_csv: Path to damage predictions CSV  
            output_csv: Path for output combined CSV
        """
        self.crack_csv = crack_csv
        self.damage_csv = damage_csv
        self.output_csv = output_csv
        
        print(f"🔗 CSV Combiner initialized")
        print(f"   Crack CSV: {crack_csv}")
        print(f"   Damage CSV: {damage_csv}")
        print(f"   Output CSV: {output_csv}")
    
    def extract_main_image_id(self, sub_image_name: str) -> str:
        """
        Extract main image ID from sub-image name
        
        Args:
            sub_image_name: Sub-image name like "1_4.jpg", "2_3.jpg"
            
        Returns:
            Main image ID like "1", "2"
        """
        # Extract number before underscore
        match = re.match(r'(\d+)_\d+\.', sub_image_name)
        if match:
            return match.group(1)
        
        # Fallback: try to extract any leading digits
        match = re.match(r'(\d+)', sub_image_name)
        if match:
            return match.group(1)
        
        return "unknown"
    
    def load_and_group_crack_data(self) -> Dict[str, List[Dict]]:
        """
        Load crack CSV and group by main image ID
        
        Returns:
            Dictionary mapping main image ID to list of crack predictions
        """
        print("📄 Loading crack predictions...")
        
        try:
            df_crack = pd.read_csv(self.crack_csv)
            print(f"   Loaded {len(df_crack)} crack predictions")
        except Exception as e:
            print(f"❌ Error loading crack CSV: {e}")
            return {}
        
        # Group crack predictions by main image
        crack_groups = defaultdict(list)
        
        for _, row in df_crack.iterrows():
            sub_image_name = str(row['Image_Name']).strip()
            main_id = self.extract_main_image_id(sub_image_name)
            
            crack_info = {
                'sub_image_name': sub_image_name,
                'crack_class': str(row['Crack_Class']).strip(),
                'crack_name': str(row['Crack_Name']).strip(),
                'confidence': str(row['Confidence']).strip(),
                'status': str(row['Status']).strip()
            }
            
            crack_groups[main_id].append(crack_info)
        
        print(f"   Grouped into {len(crack_groups)} main images")
        
        # Show grouping summary
        for main_id in sorted(crack_groups.keys(), key=lambda x: int(x) if x.isdigit() else 999)[:5]:
            sub_count = len(crack_groups[main_id])
            sub_names = [info['sub_image_name'] for info in crack_groups[main_id]]
            print(f"     Main {main_id}: {sub_count} sub-images ({', '.join(sub_names[:3])}{'...' if sub_count > 3 else ''})")
        
        return crack_groups
    
    def load_damage_data(self) -> Dict[str, Dict]:
        """
        Load damage CSV and create lookup by image ID
        
        Returns:
            Dictionary mapping image ID to damage prediction
        """
        print("\n📄 Loading damage predictions...")
        
        try:
            df_damage = pd.read_csv(self.damage_csv)
            print(f"   Loaded {len(df_damage)} damage predictions")
        except Exception as e:
            print(f"❌ Error loading damage CSV: {e}")
            return {}
        
        # Create damage lookup
        damage_lookup = {}
        
        for _, row in df_damage.iterrows():
            image_name = str(row['Image_Name']).strip()
            
            # Extract ID from image name (e.g., "1.jpg" -> "1")
            image_id = re.match(r'(\d+)\.', image_name)
            if image_id:
                image_id = image_id.group(1)
            else:
                # Fallback: use filename without extension
                image_id = os.path.splitext(image_name)[0]
            
            damage_info = {
                'image_name': image_name,
                'damage_class': str(row['Damage_Class']).strip(),
                'damage_name': str(row['Damage_Name']).strip(),
                'confidence': str(row['Confidence']).strip(),
                'status': str(row['Status']).strip()
            }
            
            damage_lookup[image_id] = damage_info
        
        print(f"   Created lookup for {len(damage_lookup)} damage images")
        
        # Show damage summary
        for image_id in sorted(damage_lookup.keys(), key=lambda x: int(x) if x.isdigit() else 999)[:5]:
            damage_info = damage_lookup[image_id]
            print(f"     ID {image_id}: {damage_info['image_name']} -> Class {damage_info['damage_class']}")
        
        return damage_lookup
    
    def combine_predictions(self, crack_groups: Dict[str, List[Dict]], 
                          damage_lookup: Dict[str, Dict]) -> List[Dict]:
        """
        Combine crack and damage predictions
        
        Args:
            crack_groups: Grouped crack predictions by main image ID
            damage_lookup: Damage predictions lookup
            
        Returns:
            List of combined prediction records
        """
        print("\n🔗 Combining predictions...")
        
        # Define mapping rules
        damage_crack_mapping = {
            '18': ['3', '4', '5', '6'],  # 18 only contain with 3,4,5,6
            '19': ['5', '7'],            # 19 only contain with 5,7  
            '20': ['9']                  # 20 only contain with 9
        }
        
        print("📋 Mapping Rules Applied:")
        for damage, allowed_cracks in damage_crack_mapping.items():
            print(f"   Damage {damage} → Only allows crack classes: {', '.join(allowed_cracks)}")
        print()
        
        combined_records = []
        missing_damage = []
        missing_crack = []
        
        # Get all unique image IDs from both sources
        all_ids = set(crack_groups.keys()) | set(damage_lookup.keys())
        
        record_id = 1
        
        for image_id in sorted(all_ids, key=lambda x: int(x) if x.isdigit() else 999):
            
            # Get damage info
            if image_id in damage_lookup:
                damage_info = damage_lookup[image_id]
                damage_image_name = damage_info['image_name']
                damage_class = damage_info['damage_class']
            else:
                missing_damage.append(image_id)
                damage_image_name = f"{image_id}.jpg"  # Assume standard naming
                damage_class = ""
            
            # Get crack info
            if image_id in crack_groups:
                crack_infos = crack_groups[image_id]
                
                # Sort crack sub-images numerically by sub-image number
                def extract_sub_number(crack_info):
                    match = re.search(r'_(\d+)\.', crack_info['sub_image_name'])
                    return int(match.group(1)) if match else 999
                
                crack_infos.sort(key=extract_sub_number)
                
                crack_image_names = [info['sub_image_name'] for info in crack_infos]
                crack_classes = [info['crack_class'] for info in crack_infos]
                crack_confidences = [float(info['confidence']) if info['confidence'].replace('.','').isdigit() else 0.0 for info in crack_infos]
                
                crack_images_str = ','.join(crack_image_names)
                crack_classes_str = ','.join(crack_classes)
            else:
                missing_crack.append(image_id)
                crack_images_str = ""
                crack_classes_str = ""
                crack_classes = []
                crack_confidences = []
            
            # Create combined classes (damage + crack classes)
            combined_classes = []
            if damage_class:
                combined_classes.append(damage_class)
            if crack_classes_str:
                combined_classes.extend(crack_classes_str.split(','))
            
            combined_classes_str = ','.join(combined_classes)
            
            # Apply mapping rules for "classes with mapping"
            mapped_classes = []
            if damage_class:
                mapped_classes.append(damage_class)
                
                # Filter crack classes based on damage class mapping rules
                if damage_class in damage_crack_mapping and crack_classes:
                    allowed_crack_classes = damage_crack_mapping[damage_class]
                    
                    # Only add crack classes that are allowed for this damage class
                    for crack_class in crack_classes:
                        crack_class = crack_class.strip()  # Remove any whitespace
                        if crack_class in allowed_crack_classes:
                            mapped_classes.append(crack_class)
            
            mapped_classes_str = ','.join(mapped_classes) if mapped_classes else ""
            
            # Find highest probability crack class for "classes with high prob"
            high_prob_classes = []
            if damage_class:
                high_prob_classes.append(damage_class)
                
                # Find highest confidence crack class from mapped classes only
                if damage_class in damage_crack_mapping and crack_classes and crack_confidences:
                    allowed_crack_classes = damage_crack_mapping[damage_class]
                    
                    # Filter to get only allowed crack classes with their confidences
                    allowed_crack_with_conf = []
                    for crack_class, confidence in zip(crack_classes, crack_confidences):
                        crack_class = crack_class.strip()  # Remove any whitespace
                        if crack_class in allowed_crack_classes:
                            allowed_crack_with_conf.append((crack_class, confidence))
                    
                    # Find highest confidence crack class from allowed ones only
                    if allowed_crack_with_conf:
                        best_crack = max(allowed_crack_with_conf, key=lambda x: x[1])
                        high_prob_classes.append(best_crack[0])
            
            high_prob_classes_str = ','.join(high_prob_classes) if high_prob_classes else ""
            
            # Create combined record
            record = {
                'ID': record_id,
                'Image_Name_Damage': damage_image_name,
                'Image_Name_Crack': crack_images_str,
                'Damage_Class': damage_class,
                'Crack_Class': crack_classes_str,
                'Classes': combined_classes_str,
                'Classes_With_Mapping': mapped_classes_str,
                'Classes_With_High_Prob': high_prob_classes_str
            }
            
            combined_records.append(record)
            record_id += 1
        
        print(f"   Combined {len(combined_records)} records")
        
        if missing_damage:
            print(f"   ⚠️  {len(missing_damage)} images missing damage predictions: {', '.join(missing_damage[:5])}{'...' if len(missing_damage) > 5 else ''}")
        
        if missing_crack:
            print(f"   ⚠️  {len(missing_crack)} images missing crack predictions: {', '.join(missing_crack[:5])}{'...' if len(missing_crack) > 5 else ''}")
        
        return combined_records
    
    def save_combined_csv(self, combined_records: List[Dict]):
        """
        Save combined records to CSV
        
        Args:
            combined_records: List of combined prediction records
        """
        print(f"\n💾 Saving combined CSV...")
        
        try:
            df_combined = pd.DataFrame(combined_records)
            df_combined.to_csv(self.output_csv, index=False)
            
            print(f"   ✅ Saved {len(combined_records)} records to: {self.output_csv}")
            
            # Show sample records
            print(f"\n📋 Sample records:")
            for i, record in enumerate(combined_records[:3]):
                print(f"   {i+1}. Damage: {record['Image_Name_Damage']} (Class: {record['Damage_Class']})")
                print(f"      Crack: {record['Image_Name_Crack']} (Classes: {record['Crack_Class']})")
                print(f"      All Classes: {record['Classes']}")
                print(f"      Mapped Classes: {record['Classes_With_Mapping']}")
                print(f"      High Prob Classes: {record['Classes_With_High_Prob']}")
                print()
                
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
    
    def print_summary_statistics(self, combined_records: List[Dict]):
        """Print summary statistics of the combined dataset"""
        print(f"\n📊 Summary Statistics:")
        print(f"   Total combined records: {len(combined_records)}")
        
        # Count records with damage predictions
        with_damage = sum(1 for r in combined_records if r['Damage_Class'])
        print(f"   Records with damage predictions: {with_damage}")
        
        # Count records with crack predictions  
        with_crack = sum(1 for r in combined_records if r['Crack_Class'])
        print(f"   Records with crack predictions: {with_crack}")
        
        # Count records with both
        with_both = sum(1 for r in combined_records if r['Damage_Class'] and r['Crack_Class'])
        print(f"   Records with both predictions: {with_both}")
        
        # Count records with mapped classes
        with_mapped = sum(1 for r in combined_records if len(r['Classes_With_Mapping'].split(',')) > 1)
        print(f"   Records with valid mapped classes: {with_mapped}")
        
        # Count records with high prob classes
        with_high_prob = sum(1 for r in combined_records if len(r['Classes_With_High_Prob'].split(',')) > 1)
        print(f"   Records with high prob predictions: {with_high_prob}")
        
        # Average number of sub-images per main image
        sub_image_counts = []
        for record in combined_records:
            if record['Image_Name_Crack']:
                count = len(record['Image_Name_Crack'].split(','))
                sub_image_counts.append(count)
        
        if sub_image_counts:
            avg_sub_images = sum(sub_image_counts) / len(sub_image_counts)
            print(f"   Average sub-images per main image: {avg_sub_images:.1f}")
            print(f"   Sub-image count range: {min(sub_image_counts)}-{max(sub_image_counts)}")
        
        # Show mapping rule effectiveness
        print(f"\n🎯 Mapping Rule Results:")
        damage_mapping_stats = {'18': 0, '19': 0, '20': 0}
        
        for record in combined_records:
            damage_class = record['Damage_Class']
            if damage_class in damage_mapping_stats:
                mapped_classes = record['Classes_With_Mapping']
                if ',' in mapped_classes:  # Has both damage and crack classes
                    damage_mapping_stats[damage_class] += 1
        
        for damage_class, count in damage_mapping_stats.items():
            print(f"   Damage {damage_class}: {count} records successfully mapped")
        
        # Show detailed mapping examples
        print(f"\n📝 Mapping Examples:")
        for record in combined_records[:3]:
            if record['Damage_Class'] and record['Crack_Class']:
                damage = record['Damage_Class']
                original_cracks = record['Crack_Class'].split(',')
                mapped_cracks = record['Classes_With_Mapping'].replace(damage + ',', '').split(',') if ',' in record['Classes_With_Mapping'] else []
                
                if original_cracks and mapped_cracks and mapped_cracks != ['']:
                    filtered_out = [c for c in original_cracks if c not in mapped_cracks]
                    print(f"   Image {record['Image_Name_Damage']}: Damage {damage}")
                    print(f"     Original cracks: {', '.join(original_cracks)}")
                    print(f"     Mapped cracks: {', '.join(mapped_cracks)}")
                    if filtered_out:
                        print(f"     Filtered out: {', '.join(filtered_out)} (not allowed with damage {damage})")
                    print()
    
    def combine_csv_files(self):
        """
        Main method to combine CSV files
        """
        print(f"🚀 Starting CSV combination process...")
        
        # Load and group crack data
        crack_groups = self.load_and_group_crack_data()
        if not crack_groups:
            print("❌ No crack data loaded. Aborting.")
            return
        
        # Load damage data
        damage_lookup = self.load_damage_data()
        if not damage_lookup:
            print("❌ No damage data loaded. Aborting.")
            return
        
        # Combine predictions
        combined_records = self.combine_predictions(crack_groups, damage_lookup)
        if not combined_records:
            print("❌ No records to combine. Aborting.")
            return
        
        # Save results
        self.save_combined_csv(combined_records)
        
        # Print statistics
        self.print_summary_statistics(combined_records)
        
        print(f"\n🎉 CSV combination completed successfully!")

def test_mapping_logic():
    """
    Test function to demonstrate mapping logic
    """
    print("🧪 Testing Mapping Logic")
    print("=" * 30)
    
    # Define mapping rules
    damage_crack_mapping = {
        '18': ['3', '4', '5', '6'],  # 18 only contain with 3,4,5,6
        '19': ['5', '7'],            # 19 only contain with 5,7  
        '20': ['9']                  # 20 only contain with 9
    }
    
    # Test cases
    test_cases = [
        {"damage": "18", "cracks": ["3", "6", "7", "8", "9"], "expected_mapped": ["18", "3", "6"]},
        {"damage": "19", "cracks": ["5", "6", "7", "8"], "expected_mapped": ["19", "5", "7"]},
        {"damage": "20", "cracks": ["5", "7", "9"], "expected_mapped": ["20", "9"]},
        {"damage": "18", "cracks": ["1", "2", "7", "8"], "expected_mapped": ["18"]},  # No valid cracks
    ]
    
    for i, test in enumerate(test_cases, 1):
        damage_class = test["damage"]
        crack_classes = test["cracks"]
        expected = test["expected_mapped"]
        
        # Apply mapping logic
        mapped_classes = [damage_class]
        if damage_class in damage_crack_mapping:
            allowed_crack_classes = damage_crack_mapping[damage_class]
            for crack_class in crack_classes:
                if crack_class in allowed_crack_classes:
                    mapped_classes.append(crack_class)
        
        # Check result
        result = "✅ PASS" if mapped_classes == expected else "❌ FAIL"
        
        print(f"Test {i}: {result}")
        print(f"  Damage: {damage_class}, Cracks: {crack_classes}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {mapped_classes}")
        print()

def main():
    """Main function with example usage"""
    
    # Example file paths - modify these to match your files
    CRACK_CSV = "crack_results_detailed.csv"      # Your crack CSV file
    DAMAGE_CSV = "damage_results_detailed.csv"    # Your damage CSV file  
    OUTPUT_CSV = "combined_predictions.csv"  # Output combined file
    
    print("🔗 Crack & Damage CSV Combiner")
    print("=" * 40)
    
    # Uncomment to test mapping logic
    # test_mapping_logic()
    # return
    
    # Check if input files exist
    if not os.path.exists(CRACK_CSV):
        print(f"❌ Crack CSV not found: {CRACK_CSV}")
        print(f"   Please update CRACK_CSV path in the script")
        return
    
    if not os.path.exists(DAMAGE_CSV):
        print(f"❌ Damage CSV not found: {DAMAGE_CSV}")
        print(f"   Please update DAMAGE_CSV path in the script")
        return
    
    # Create combiner and run
    combiner = CrackDamageCombiner(CRACK_CSV, DAMAGE_CSV, OUTPUT_CSV)
    combiner.combine_csv_files()

def combine_csv_files_simple(crack_csv: str, damage_csv: str, output_csv: str):
    """
    Simple wrapper function for easy usage
    
    Args:
        crack_csv: Path to crack predictions CSV
        damage_csv: Path to damage predictions CSV
        output_csv: Path for output combined CSV
    """
    combiner = CrackDamageCombiner(crack_csv, damage_csv, output_csv)
    combiner.combine_csv_files()

if __name__ == "__main__":
    main()

# Quick usage examples:
"""
# Method 1: Update paths in main() and run
python csv_combiner.py

# Method 2: Import and use directly
from csv_combiner import combine_csv_files_simple

combine_csv_files_simple(
    crack_csv="crack_results.csv",
    damage_csv="damage_results.csv", 
    output_csv="combined_results.csv"
)

# Method 3: Use class directly for more control
from csv_combiner import CrackDamageCombiner

combiner = CrackDamageCombiner("crack.csv", "damage.csv", "output.csv")
combiner.combine_csv_files()

# Output CSV will have these columns:
# - ID: Sequential record number
# - Image_Name_Damage: Main damage image (e.g., "1.jpg")
# - Image_Name_Crack: Sub-crack images (e.g., "1_1.jpg,1_2.jpg,1_3.jpg")
# - Damage_Class: Damage class (18, 19, or 20)
# - Crack_Class: All crack classes (e.g., "6,5,3,4")
# - Classes: Combined all classes (e.g., "18,6,5,3,4")
# - Classes_With_Mapping: Filtered by mapping rules:
#   * 18 only with 3,4,5,6
#   * 19 only with 5,7
#   * 20 only with 9
# - Classes_With_High_Prob: Damage + highest confidence crack from mapping
"""