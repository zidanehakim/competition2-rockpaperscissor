import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import argparse
import re

# Model definition (same as training)
class SingleHeadResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SingleHeadResNet, self).__init__()
        
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class CrackInference:
    def __init__(self, model_path, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        
        # Load model
        self.model = SingleHeadResNet(num_classes=7)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Class mappings
        self.idx_to_class = {0: '3', 1: '4', 2: '5', 3: '6', 4: '7', 5: '8', 6: '9'}
        
        # Updated class names
        self.class_names = {
            '3': 'X_Shape',
            '4': 'Continuous_Diagonal_Crack',
            '5': 'Discontinuous_Diagonal_Crack',
            '6': 'Continuous_Vertical_Crack',
            '7': 'Discontinuous_Vertical_Crack',
            '8': 'Continuous_Horizontal_Crack',
            '9': 'Discontinuous_Horizontal_Crack'
        }
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def find_images(self, image_dir):
        """Find all images in directory with numerical sorting"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        
        for ext in extensions:
            pattern = os.path.join(image_dir, '**', ext)
            image_paths.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(image_dir, '**', ext.upper())
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        # Remove duplicates
        image_paths = list(set(image_paths))
        
        # Sort numerically by extracting numbers from filename
        def numeric_sort_key(path):
            filename = os.path.basename(path)
            numbers = re.findall(r'\d+', filename)
            if numbers:
                return int(numbers[0])
            return 0
        
        return sorted(image_paths, key=numeric_sort_key)
    
    def predict_single_image(self, image_path):
        """Predict crack class for single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                
                # Get probabilities
                prob = torch.softmax(output, dim=1)
                
                # Get prediction
                pred_idx = torch.argmax(prob, dim=1).item()
                
                # Convert to original label
                pred_class = self.idx_to_class[pred_idx]
                
                # Get confidence score
                confidence = prob[0][pred_idx].item()
                
                return {
                    'success': True,
                    'crack_class': pred_class,
                    'crack_name': self.class_names[pred_class],
                    'confidence': confidence,
                    'error': None
                }
        
        except Exception as e:
            return {
                'success': False,
                'crack_class': None,
                'crack_name': None,
                'confidence': None,
                'error': str(e)
            }
    
    def predict_batch(self, image_paths):
        """Predict on batch of images"""
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        for img_path in tqdm(image_paths):
            result = self.predict_single_image(img_path)
            result['image_path'] = img_path
            result['image_name'] = os.path.basename(img_path)
            results.append(result)
        
        return results
    
    def create_simple_csv(self, results, output_file):
        """Create simple CSV: ID, Crack_Class"""
        
        with open(output_file, 'w') as f:
            f.write('ID,Crack_Class\n')
            
            for i, result in enumerate(results, 1):
                if result['success']:
                    f.write(f'{i},{result["crack_class"]}\n')
                else:
                    f.write(f'{i},-1\n')  # Error indicator
        
        print(f"Simple CSV saved to: {output_file}")
    
    def create_detailed_csv(self, results, output_file):
        """Create detailed CSV with confidence scores and names"""
        csv_data = []
        
        for i, result in enumerate(results, 1):
            if result['success']:
                csv_data.append({
                    'ID': i,
                    'Image_Name': result['image_name'],
                    'Crack_Class': result['crack_class'],
                    'Crack_Name': result['crack_name'],
                    'Confidence': f"{result['confidence']:.3f}",
                    'Status': 'Success'
                })
            else:
                csv_data.append({
                    'ID': i,
                    'Image_Name': result['image_name'],
                    'Crack_Class': 'Error',
                    'Crack_Name': 'Error',
                    'Confidence': '0.000',
                    'Status': f"Error: {result['error']}"
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        print(f"Detailed CSV saved to: {output_file}")
        
        return df
    
    def print_summary(self, results):
        """Print prediction summary"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n{'='*60}")
        print(f"CRACK CLASSIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Total Images: {len(results)}")
        print(f"Successful Predictions: {len(successful)}")
        print(f"Failed Predictions: {len(failed)}")
        
        if successful:
            # Count predictions by class
            class_counts = {}
            
            for result in successful:
                crack_class = result['crack_class']
                class_counts[crack_class] = class_counts.get(crack_class, 0) + 1
            
            print(f"\nCrack Classification Distribution:")
            for crack_class in sorted(class_counts.keys()):
                count = class_counts[crack_class]
                percentage = count / len(successful) * 100
                crack_name = self.class_names.get(crack_class, crack_class)
                print(f"  Class {crack_class} ({crack_name}): {count} ({percentage:.1f}%)")
            
            # Average confidence score
            avg_conf = np.mean([r['confidence'] for r in successful])
            
            print(f"\nAverage Confidence Score: {avg_conf:.3f}")
        
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='Crack Classification Inference')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', default='crack_results.csv', help='Output CSV file')
    parser.add_argument('--simple', action='store_true', help='Create simple CSV (ID, Crack only)')
    parser.add_argument('--detailed', action='store_true', help='Create detailed CSV with confidence scores')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = CrackInference(args.model)
    
    # Single image or directory
    if os.path.isfile(args.input):
        # Single image
        print(f"Processing single image: {args.input}")
        result = inference.predict_single_image(args.input)
        
        if result['success']:
            print(f"\nPrediction Results:")
            print(f"Input: 1 image")
            print(f"Output: 1 label")
            print(f"  Crack: Class {result['crack_class']} - {result['crack_name']} (confidence: {result['confidence']:.3f})")
        else:
            print(f"Error: {result['error']}")
    
    elif os.path.isdir(args.input):
        # Directory of images
        image_paths = inference.find_images(args.input)
        
        if not image_paths:
            print(f"No images found in {args.input}")
            return
        
        # Process all images
        results = inference.predict_batch(image_paths)
        
        # Create output based on flags
        if args.simple:
            inference.create_simple_csv(results, args.output.replace('.csv', '_simple.csv'))
        
        if args.detailed:
            df = inference.create_detailed_csv(results, args.output.replace('.csv', '_detailed.csv'))
        
        if not args.simple and not args.detailed:
            # Default: create simple CSV
            inference.create_simple_csv(results, args.output)
        
        # Print summary
        inference.print_summary(results)
        
        # Show sample results
        successful_results = [r for r in results if r['success']]
        if successful_results:
            print(f"\nSample Results (first 5):")
            for i, result in enumerate(successful_results[:5], 1):
                print(f"  {i}. {result['image_name']}: Crack={result['crack_class']}")
    
    else:
        print(f"Input path not found: {args.input}")

if __name__ == "__main__":
    main()

#python crack_inference.py --model ./best_crack_model.pth --input /root/ak/extracted_boxes --output ./crack_results.csv --detailed