import os
import json
import logging
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
from google import genai
from google.genai import types
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LaneDescription:
    """Structured lane description from Gemini"""
    description: str
    type: str
    location_hint: str
    reference_points: List[Tuple[int, int]] = None  # Optional reference points

class EnhancedSAMLaneDetector:
    """Enhanced SAM-based lane detection with better prompting strategy"""
    
    def __init__(self, model_type="vit_h", checkpoint_path=None):
        """Initialize with specified SAM model"""
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple MPS")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU (consider using GPU for better performance)")
        
        # Download checkpoint if needed
        if checkpoint_path is None:
            checkpoint_path = f"sam_{model_type}_checkpoint.pth"
            if not os.path.exists(checkpoint_path):
                self._download_checkpoint(model_type, checkpoint_path)
        
        # Load SAM
        logger.info(f"Loading SAM {model_type}...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        logger.info("SAM loaded successfully!")
        
        # Initialize Gemini
        self.gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Lane colors
        self.colors = {
            "white_solid": (255, 255, 255, 200),
            "white_dashed": (255, 255, 255, 200),
            "yellow_solid": (255, 255, 0, 200),
            "yellow_dashed": (255, 255, 0, 200),
            "road_edge": (255, 0, 0, 200)
        }
    
    def _download_checkpoint(self, model_type, save_path):
        """Download SAM checkpoint"""
        import urllib.request
        
        urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        if model_type not in urls:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Downloading SAM {model_type} checkpoint...")
        urllib.request.urlretrieve(urls[model_type], save_path)
        logger.info("Download complete!")
    
    def get_lane_descriptions_with_points(self, image_path):
        """Get lane descriptions with reference points from Gemini"""
        
        image = Image.open(image_path)
        
        prompt = """
        Analyze this road image and identify all lane markings and road edges.
        
        For each lane/edge, provide:
        1. A detailed description for finding it
        2. Type: white_solid, white_dashed, yellow_solid, yellow_dashed, or road_edge
        3. 2-3 reference points along the line (in pixel coordinates)
        
        Format as JSON:
        [
          {
            "description": "white solid line separating the left shoulder from the main travel lane",
            "type": "white_solid",
            "location_hint": "left side of the road, running parallel to traffic",
            "reference_points": [[150, 380], [160, 250], [170, 100]]
          }
        ]
        
        Important:
        - Reference points should follow the line from bottom to top
        - Be specific about what each line separates
        - Include context (e.g., "near the white car", "before the merge")
        """
        
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1024)
        )
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, image],
                config=config
            )
            
            # Parse JSON
            json_str = self._extract_json(response.text)
            lane_data = json.loads(json_str)
            
            # Convert to LaneDescription objects
            descriptions = []
            for item in lane_data:
                desc = LaneDescription(
                    description=item["description"],
                    type=item["type"],
                    location_hint=item.get("location_hint", ""),
                    reference_points=item.get("reference_points", [])
                )
                descriptions.append(desc)
            
            logger.info(f"Gemini identified {len(descriptions)} lanes")
            return descriptions
            
        except Exception as e:
            logger.error(f"Error getting descriptions: {e}")
            return []
    
    def _extract_json(self, text):
        """Extract JSON from response"""
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == "```":
                        return "\n".join(lines[i + 1:j])
        return text
    
    def segment_lane_with_points(self, image_np, lane_desc):
        """Segment a lane using reference points and description"""
        
        # Set image for SAM
        self.predictor.set_image(image_np)
        
        h, w = image_np.shape[:2]
        
        # Use reference points if available
        if lane_desc.reference_points and len(lane_desc.reference_points) > 0:
            # Convert reference points to numpy array
            points = np.array(lane_desc.reference_points)
            
            # Ensure points are within bounds
            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, h - 1)
            
            # All positive labels
            labels = np.ones(len(points))
            
            logger.info(f"Using {len(points)} reference points for segmentation")
        else:
            # Fallback: Generate points based on location hint
            points, labels = self._generate_search_points(w, h, lane_desc.location_hint)
            logger.info(f"Generated {len(points)} search points from location hint")
        
        # Get predictions
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # Refine mask for lane shapes (thin, elongated)
        refined_mask = self._refine_lane_mask(best_mask, lane_desc.type)
        
        return refined_mask
    
    def _generate_search_points(self, w, h, location_hint):
        """Generate search points based on location hint"""
        points = []
        
        # Parse location hint
        hint_lower = location_hint.lower()
        
        # Determine x range
        if "left" in hint_lower:
            x_start, x_end = 0, w // 3
        elif "right" in hint_lower:
            x_start, x_end = 2 * w // 3, w
        elif "center" in hint_lower or "middle" in hint_lower:
            x_start, x_end = w // 3, 2 * w // 3
        else:
            x_start, x_end = 0, w
        
        # Generate points along likely lane positions
        for y in range(h - 50, 50, -100):  # Bottom to top
            for x in range(x_start, x_end, 50):
                points.append([x, y])
        
        return np.array(points), np.ones(len(points))
    
    def _refine_lane_mask(self, mask, lane_type):
        """Refine mask to better match lane characteristics"""
        
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Apply morphological operations
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Close small gaps
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        # Keep only large components (lanes are typically long)
        min_area = mask_uint8.shape[0] * 10  # Minimum area threshold
        refined_mask = np.zeros_like(mask_uint8)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                refined_mask[labels == i] = 255
        
        return refined_mask > 0
    
    def process_image(self, image_path, output_dir):
        """Process a single image"""
        
        logger.info(f"\nProcessing: {image_path}")
        
        # Load image
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Get descriptions from Gemini
        lane_descriptions = self.get_lane_descriptions_with_points(image_path)
        
        if not lane_descriptions:
            logger.warning("No lanes identified by Gemini")
            return
        
        # Create overlay
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Statistics
        stats = {t: 0 for t in self.colors.keys()}
        results = []
        
        # Process each lane
        for i, lane_desc in enumerate(lane_descriptions):
            logger.info(f"\nSegmenting lane {i+1}/{len(lane_descriptions)}: {lane_desc.type}")
            logger.info(f"Description: {lane_desc.description[:100]}...")
            
            try:
                # Get segmentation mask
                mask = self.segment_lane_with_points(image_rgb, lane_desc)
                
                if mask is not None and np.any(mask):
                    # Update stats
                    stats[lane_desc.type] += 1
                    
                    # Apply color
                    color = self.colors.get(lane_desc.type, (128, 128, 128, 200))
                    overlay[mask] = color
                    
                    # Calculate mask properties
                    mask_area = np.sum(mask)
                    y_coords, x_coords = np.where(mask)
                    
                    if len(y_coords) > 0:
                        results.append({
                            "description": lane_desc.description,
                            "type": lane_desc.type,
                            "mask_area": int(mask_area),
                            "bbox": [
                                int(np.min(x_coords)),
                                int(np.min(y_coords)),
                                int(np.max(x_coords)),
                                int(np.max(y_coords))
                            ]
                        })
                    
                    logger.info(f"✓ Successfully segmented (area: {mask_area} pixels)")
                else:
                    logger.warning("✗ Failed to segment - no mask generated")
                    
            except Exception as e:
                logger.error(f"✗ Error during segmentation: {e}")
        
        # Save outputs
        self._save_outputs(image_path, image_rgb, overlay, stats, results, output_dir)
        
        return results
    
    def _save_outputs(self, image_path, image_rgb, overlay, stats, results, output_dir):
        """Save all output files"""
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Convert to PIL
        image_pil = Image.fromarray(image_rgb)
        overlay_pil = Image.fromarray(overlay)
        
        # Create composite
        composite = Image.alpha_composite(
            image_pil.convert('RGBA'),
            overlay_pil
        )
        
        # Save files
        composite.save(os.path.join(output_dir, f"{base_name}_composite.png"))
        overlay_pil.save(os.path.join(output_dir, f"{base_name}_overlay.png"))
        
        # Save JSON results
        output_data = {
            "image": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "detected_lanes": results,
            "total_lanes": sum(stats.values())
        }
        
        with open(os.path.join(output_dir, f"{base_name}_results.json"), 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\n=== Summary for {base_name} ===")
        logger.info(f"Total lanes detected: {sum(stats.values())}")
        for lane_type, count in stats.items():
            if count > 0:
                logger.info(f"  {lane_type}: {count}")


def main():
    """Main execution"""
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Please set GOOGLE_API_KEY environment variable")
        return
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Lane detection using SAM and Gemini')
    parser.add_argument('--image', type=str, help='Process a single specific image')
    parser.add_argument('--count', type=int, default=None, help='Number of images to process (default: all)')
    parser.add_argument('--input-dir', type=str, default='examples', help='Input directory (default: examples)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: auto-generated with timestamp)')
    args = parser.parse_args()
    
    # Initialize detector
    detector = EnhancedSAMLaneDetector(model_type="vit_h")
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"sam_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Determine which images to process
    if args.image:
        # Process single specified image
        if os.path.exists(args.image):
            logger.info(f"Processing single image: {args.image}")
            detector.process_image(args.image, output_dir)
        else:
            logger.error(f"Image not found: {args.image}")
            return
    else:
        # Look for images in the input directory
        input_dir = args.input_dir
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            logger.warning(f"Input directory '{input_dir}' not found, checking current directory...")
            input_dir = '.'
        
        # Find all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = []
        
        for file in sorted(os.listdir(input_dir)):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(input_dir, file))
        
        if not image_files:
            logger.error(f"No image files found in {input_dir}")
            logger.info(f"Supported formats: {', '.join(image_extensions)}")
            return
        
        # Limit number of images if specified
        if args.count and args.count > 0:
            image_files = image_files[:args.count]
            logger.info(f"Processing first {args.count} image(s) out of {len(image_files)} found")
        else:
            logger.info(f"Processing all {len(image_files)} images found")
        
        # Process each image
        for idx, img_path in enumerate(image_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing image {idx}/{len(image_files)}: {os.path.basename(img_path)}")
            logger.info(f"{'='*60}")
            
            try:
                detector.process_image(img_path, output_dir)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                if args.count == 1:
                    # If only processing one image, don't continue on error
                    return
    
    logger.info(f"\n{'='*60}")
    logger.info("All processing complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
