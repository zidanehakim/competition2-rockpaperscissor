from models.yolo_model import test_yolo_model
from models.multi_label_model import test_multi_label_model
from models.single_label_model import test_single_label_model
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mapping YOLO class indices to class names for reference
YOLO_MAPPING = {
    0: "Exposed Rebar",
    1: "Cracks",
    2: "Huge Spalling",
}

def generate_submission(results):
    """
    Generate a submission CSV file from the classification results.
    
    Args:
        results: Dictionary mapping file_id to list of integer classes [int, int, int]
    """
    try:
        submission = pd.DataFrame(columns=["ID", "class"])
        
        for file_id, classes in results.items():
            # Directly use the list of integers
            # Convert the list into comma-separated values without spaces
            class_str = ','.join(map(str, classes))
            row = pd.DataFrame([{"ID": file_id, "class": class_str}])
            submission = pd.concat([submission, row], ignore_index=True)
        
        # Sort by ID to ensure order
        submission = submission.sort_values(by="ID").reset_index(drop=True)
        
        # Save to CSV
        submission.to_csv("submission.csv", index=False)
        logger.info(f"Submission file generated successfully with {len(submission)} entries.")
        
    except Exception as e:
        logger.error(f"Error generating submission file: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting classification pipeline...")
        
        # Step 1: Run YOLO detection model
        logger.info("Running YOLO model for initial detection...")
        yolo_results, yolo_cropped_images = test_yolo_model()
        print(f"YOLO results: {yolo_results}")
        logger.info(f"YOLO detection completed for {len(yolo_results)} images.")
        
        # Step 2: Run multi-label classification on detected cracks
        logger.info("Running multi-label model for crack classification...")
        multi_label_results = test_multi_label_model(yolo_results, yolo_cropped_images)
        logger.info(f"Multi-label classification completed for {len(multi_label_results)} images.")
        
        # Step 3: Run single-label classification for damage level
        logger.info("Running single-label model for damage classification...")
        final_results = test_single_label_model(multi_label_results)
        logger.info(f"Single-label classification completed for {len(final_results)} images.")
        
        # Generate submission file
        logger.info("Generating submission file...")
        generate_submission(final_results)
        logger.info("Classification pipeline completed successfully!")        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        exit(1)