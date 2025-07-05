import os
import json
import logging
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Gemini client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

client = genai.Client(api_key=GOOGLE_API_KEY)

class LaneDetectionSystem:
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth", device=None):
        """Initialize the lane detection system with SAM ViT-H"""
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using Apple Metal Performance Shaders")
            else:
                self.device = "cpu"
                logger.info("Using CPU (this will be slow)")
        else:
            self.device = device
            
        # Download SAM checkpoint if needed
        if not os.path.exists(sam_checkpoint):
            logger.info("Downloading SAM ViT-H checkpoint...")
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            urllib.request.urlretrieve(url, sam_checkpoint)
            logger.info("Download complete!")
        
        # Load SAM model
        logger.info("Loading SAM ViT-H model...")
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        logger.info("SAM model loaded successfully!")
        
        # Color mapping for lane types
        self.color_map = {
            "white_solid": (255, 255, 255, 200),
            "white_dashed": (255, 255, 255, 200),
            "yellow_solid": (255, 255, 0, 200),
            "yellow_dashed": (255, 255, 0, 200),
            "road_edge": (255, 0, 0, 200)
        }
        
    def get_lane_descriptions(self, image_path):
        """Use Gemini to get semantic descriptions of lanes"""
        
        # Load image
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        prompt = """
        Analyze this road image and describe each lane marking and road edge.
        
        For each lane/edge, provide:
        1. A clear, specific description of its location
        2. Its type: white_solid, white_dashed, yellow_solid, yellow_dashed, or road_edge
        3. Approximate location hints (e.g., "left third of image", "center", etc.)
        
        Output as JSON array:
        [
          {
            "description": "solid white line along the left shoulder separating the leftmost lane from the road edge",
            "type": "white_solid",
            "location_hint": "left side, running from bottom to top"
          }
        ]
        
        Be specific about:
        - Which side of the road (left/right)
        - What it separates (e.g., "between lane 1 and lane 2")
        - Any unique features (curves, merges, etc.)
        """
        
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=1024,
                include_thoughts=True
            )
        )
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, image],
                config=config
            )
            
            # Parse response
            json_output = self.parse_json(response.text)
            descriptions = json.loads(json_output)
            
            logger.info(f"Gemini identified {len(descriptions)} lane elements")
            return descriptions
            
        except Exception as e:
            logger.error(f"Failed to get lane descriptions: {e}")
            return []
    
    def parse_json(self, text):
        """Extract JSON from markdown fencing"""
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == "```":
                        return "\n".join(lines[i + 1:j])
                return "\n".join(lines[i + 1:])
        return text
    
    def segment_with_text(self, image_np, text_prompt, location_hint=None):
        """Use SAM to segment based on text description"""
        
        # Set the image
        self.predictor.set_image(image_np)
        
        # For text-based segmentation, we'll use a grid of points
        # and filter based on the location hint
        h, w = image_np.shape[:2]
        
        # Generate point grid based on location hint
        if location_hint:
            if "left" in location_hint.lower():
                x_range = (0, w // 3)
            elif "right" in location_hint.lower():
                x_range = (2 * w // 3, w)
            elif "center" in location_hint.lower():
                x_range = (w // 3, 2 * w // 3)
            else:
                x_range = (0, w)
                
            if "top" in location_hint.lower():
                y_range = (0, h // 2)
            elif "bottom" in location_hint.lower():
                y_range = (h // 2, h)
            else:
                y_range = (0, h)
        else:
            x_range = (0, w)
            y_range = (0, h)
        
        # Create a grid of points
        points = []
        for y in range(y_range[0], y_range[1], 50):
            for x in range(x_range[0], x_range[1], 50):
                points.append([x, y])
        
        if not points:
            logger.warning(f"No points generated for prompt: {text_prompt}")
            return None
            
        points = np.array(points)
        labels = np.ones(len(points))  # All positive points
        
        # Get masks
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        return masks[best_idx]
    
    def process_image(self, image_path, output_dir):
        """Complete pipeline for processing a single image"""
        
        logger.info(f"Processing {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Get lane descriptions from Gemini
        descriptions = self.get_lane_descriptions(image_path)
        
        if not descriptions:
            logger.warning("No lane descriptions received from Gemini")
            return
        
        # Create overlay for all lanes
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Statistics
        stats = {
            "white_solid": 0,
            "white_dashed": 0,
            "yellow_solid": 0,
            "yellow_dashed": 0,
            "road_edge": 0
        }
        
        # Process each description
        results = []
        for i, desc in enumerate(descriptions):
            lane_type = desc.get("type", "unknown")
            description = desc.get("description", "")
            location_hint = desc.get("location_hint", "")
            
            logger.info(f"Processing: {description[:50]}...")
            
            # Get segmentation mask
            mask = self.segment_with_text(image_rgb, description, location_hint)
            
            if mask is not None:
                # Update statistics
                if lane_type in stats:
                    stats[lane_type] += 1
                
                # Get color
                color = self.color_map.get(lane_type, (128, 128, 128, 200))
                
                # Apply mask to overlay
                mask_indices = mask.astype(bool)
                overlay[mask_indices] = color
                
                # Store result
                results.append({
                    "description": description,
                    "type": lane_type,
                    "location_hint": location_hint,
                    "mask_area": np.sum(mask)
                })
                
                logger.info(f"✓ Segmented {lane_type}")
            else:
                logger.warning(f"✗ Failed to segment: {description}")
        
        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Convert overlay to PIL Image
        overlay_pil = Image.fromarray(overlay)
        image_pil = Image.fromarray(image_rgb)
        
        # Composite overlay on original
        composite = Image.alpha_composite(
            image_pil.convert('RGBA'),
            overlay_pil
        )
        
        # Save composite
        composite_path = os.path.join(output_dir, f"{base_name}_lanes_sam.png")
        composite.save(composite_path)
        logger.info(f"Saved composite: {composite_path}")
        
        # Save mask only
        mask_only = Image.new('RGBA', (w, h), (0, 0, 0, 255))
        mask_only = Image.alpha_composite(mask_only, overlay_pil)
        mask_path = os.path.join(output_dir, f"{base_name}_mask_sam.png")
        mask_only.save(mask_path)
        logger.info(f"Saved mask: {mask_path}")
        
        # Save results JSON
        output_data = {
            "image": os.path.basename(image_path),
            "statistics": stats,
            "lanes": results,
            "method": "SAM_ViT_H"
        }
        
        json_path = os.path.join(output_dir, f"{base_name}_results_sam.json")
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Detection complete: {sum(stats.values())} lanes found")
        return results


# Alternative implementation using GroundingDINO + SAM for better text understanding
class GroundedSAMDetection:
    """Advanced implementation using GroundingDINO for text grounding"""
    
    def __init__(self):
        # This would require additional setup
        logger.info("GroundedSAM requires GroundingDINO installation")
        logger.info("Install with: pip install groundingdino-py")
        
    # Implementation would go here...


def main():
    """Main execution function"""
    
    # Initialize system
    detector = LaneDetectionSystem()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"sam_lane_detection_{timestamp}"
    
    # Find images
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        logger.error("No image files found")
        return
    
    logger.info(f"Found {len(image_files)} images")
    
    # Process each image
    for image_file in image_files:
        try:
            detector.process_image(image_file, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            continue
    
    logger.info("All processing complete!")


if __name__ == "__main__":
    main()
