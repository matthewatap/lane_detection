import os
import json
import logging
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
import cv2
from google import genai
from google.genai import types
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LaneSegment:
    """Represents a detected lane segment"""
    type: str
    points: List[Tuple[int, int]]
    confidence: float
    description: str

class HybridLaneDetector:
    """Combines Gemini's semantic understanding with edge detection for line tracing"""
    
    def __init__(self):
        """Initialize the hybrid detector"""
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set GOOGLE_API_KEY environment variable")
        
        self.gemini_client = genai.Client(api_key=api_key)
        
        # Color mapping for different lane types
        self.colors = {
            "white_solid": (255, 255, 255),      # White
            "white_dashed": (200, 200, 200),     # Light gray
            "yellow_solid": (0, 255, 255),       # Yellow (BGR)
            "yellow_dashed": (0, 200, 200),      # Darker yellow
            "road_edge": (0, 0, 255)             # Red
        }
        
        # Line thickness settings
        self.line_thickness = {
            "white_solid": 4,
            "white_dashed": 3,
            "yellow_solid": 4,
            "yellow_dashed": 3,
            "road_edge": 5
        }
    
    def get_lane_regions_from_gemini(self, image_path):
        """Get semantic understanding of lane locations from Gemini"""
        
        image = Image.open(image_path)
        
        prompt = """
        Analyze this road image and identify regions where lane markings and road edges are located.
        
        For each lane marking or road edge, provide:
        1. Type: white_solid, white_dashed, yellow_solid, yellow_dashed, or road_edge
        2. Description of its location
        3. Approximate bounding region as percentage of image (left_x, top_y, right_x, bottom_y)
        4. Whether it's vertical (running up/down) or horizontal
        
        Return as JSON array:
        [
          {
            "type": "white_dashed",
            "description": "Left lane divider between lanes 1 and 2",
            "region": [20, 0, 30, 100],
            "orientation": "vertical",
            "curved": false
          }
        ]
        
        Notes:
        - Region coordinates are percentages (0-100)
        - Include ALL visible lane markings
        - Road edges are where pavement meets grass/dirt
        - Be specific about dashed vs solid lines
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
            regions = json.loads(json_str)
            
            logger.info(f"Gemini identified {len(regions)} lane regions")
            return regions
            
        except Exception as e:
            logger.error(f"Error getting lane regions: {e}")
            return []
    
    def _extract_json(self, text):
        """Extract JSON from response"""
        lines = text.splitlines()
        json_start = -1
        json_end = -1
        
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_start = i + 1
            elif line.strip() == "```" and json_start != -1:
                json_end = i
                break
        
        if json_start != -1 and json_end != -1:
            return "\n".join(lines[json_start:json_end])
        return text
    
    def detect_lines_in_region(self, image, region_info, img_width, img_height):
        """Detect actual line pixels within a region using edge detection"""
        
        # Convert region percentages to pixels
        left = int(region_info["region"][0] * img_width / 100)
        top = int(region_info["region"][1] * img_height / 100)
        right = int(region_info["region"][2] * img_width / 100)
        bottom = int(region_info["region"][3] * img_height / 100)
        
        # Ensure valid bounds
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)
        
        # Extract region
        region = image[top:bottom, left:right]
        
        if region.size == 0:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply different preprocessing based on lane type
        lane_type = region_info["type"]
        
        if "yellow" in lane_type:
            # Enhance yellow detection
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        elif "white" in lane_type:
            # Enhance white detection
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            gray = mask
        elif lane_type == "road_edge":
            # Use gradient for road edges
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gray = np.sqrt(grad_x**2 + grad_y**2).astype(np.uint8)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Use Hough transform to find lines
        if region_info["orientation"] == "vertical":
            # For vertical lines, use standard Hough
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, 
                                   minLineLength=30, maxLineGap=50)
        else:
            # For horizontal lines, adjust parameters
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, 
                                   minLineLength=20, maxLineGap=30)
        
        # Convert line segments back to image coordinates
        line_points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Convert back to full image coordinates
                x1 += left
                y1 += top
                x2 += left
                y2 += top
                line_points.append([(x1, y1), (x2, y2)])
        
        return line_points
    
    def connect_line_segments(self, segments, lane_type):
        """Connect individual line segments into continuous lanes"""
        
        if not segments:
            return []
        
        # For dashed lines, connect segments with gaps
        if "dashed" in lane_type:
            # Sort segments by their start point
            segments.sort(key=lambda seg: seg[0][1])  # Sort by y-coordinate
            
            connected_points = []
            for seg in segments:
                connected_points.extend(seg)
            
            return connected_points
        else:
            # For solid lines, try to form one continuous line
            # Extract all points
            all_points = []
            for seg in segments:
                all_points.extend(seg)
            
            if len(all_points) < 2:
                return all_points
            
            # Sort points to form a continuous line
            # Simple approach: sort by y-coordinate (works for mostly vertical lines)
            all_points.sort(key=lambda p: p[1])
            
            return all_points
    
    def draw_lane_overlay(self, image, lane_segments):
        """Draw detected lanes on the image"""
        
        # Create overlay
        overlay = image.copy()
        
        for lane in lane_segments:
            color = self.colors.get(lane.type, (128, 128, 128))
            thickness = self.line_thickness.get(lane.type, 3)
            
            if "dashed" in lane.type:
                # Draw dashed line
                self.draw_dashed_line(overlay, lane.points, color, thickness)
            else:
                # Draw solid line
                if len(lane.points) >= 2:
                    # Draw as polyline
                    points = np.array(lane.points, dtype=np.int32)
                    cv2.polylines(overlay, [points], False, color, thickness)
        
        # Blend with original
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        return result
    
    def draw_dashed_line(self, image, points, color, thickness):
        """Draw a dashed line through points"""
        
        if len(points) < 2:
            return
        
        # Draw short segments with gaps
        dash_length = 20
        gap_length = 15
        
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i + 1]
            
            # Calculate distance
            dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            
            if dist == 0:
                continue
            
            # Draw dashes along the line
            num_dashes = int(dist / (dash_length + gap_length))
            
            for j in range(num_dashes + 1):
                start_ratio = j * (dash_length + gap_length) / dist
                end_ratio = min((j * (dash_length + gap_length) + dash_length) / dist, 1.0)
                
                if start_ratio >= 1.0:
                    break
                
                start_x = int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio)
                start_y = int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
                end_x = int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio)
                end_y = int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
    
    def process_image(self, image_path, output_dir):
        """Process a single image to detect and draw lanes"""
        
        logger.info(f"\nProcessing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        height, width = image.shape[:2]
        
        # Get lane regions from Gemini
        regions = self.get_lane_regions_from_gemini(image_path)
        
        if not regions:
            logger.warning("No lane regions identified")
            return
        
        # Detect lines in each region
        all_lane_segments = []
        
        for region_info in regions:
            logger.info(f"Processing {region_info['type']}: {region_info['description']}")
            
            # Detect line segments in this region
            line_segments = self.detect_lines_in_region(image, region_info, width, height)
            
            if line_segments:
                # Connect segments
                connected_points = self.connect_line_segments(line_segments, region_info['type'])
                
                if connected_points:
                    lane_segment = LaneSegment(
                        type=region_info['type'],
                        points=connected_points,
                        confidence=0.9,
                        description=region_info['description']
                    )
                    all_lane_segments.append(lane_segment)
                    logger.info(f"  ✓ Detected {len(connected_points)} points")
            else:
                logger.warning(f"  ✗ No lines detected in region")
        
        # Draw overlay
        result_image = self.draw_lane_overlay(image, all_lane_segments)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save output image
        output_path = os.path.join(output_dir, f"{base_name}_lanes.jpg")
        cv2.imwrite(output_path, result_image)
        logger.info(f"Saved result to: {output_path}")
        
        # Save detection data
        detection_data = {
            "image": os.path.basename(image_path),
            "lanes_detected": len(all_lane_segments),
            "lanes": [
                {
                    "type": lane.type,
                    "description": lane.description,
                    "num_points": len(lane.points),
                    "confidence": lane.confidence
                }
                for lane in all_lane_segments
            ]
        }
        
        json_path = os.path.join(output_dir, f"{base_name}_lanes.json")
        with open(json_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        # Summary
        logger.info(f"\nSummary for {base_name}:")
        lane_counts = {}
        for lane in all_lane_segments:
            lane_counts[lane.type] = lane_counts.get(lane.type, 0) + 1
        
        for lane_type, count in lane_counts.items():
            logger.info(f"  {lane_type}: {count}")


def main():
    """Main execution function"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Hybrid lane detection')
    parser.add_argument('--image', type=str, help='Single image to process')
    parser.add_argument('--count', type=int, help='Number of images to process')
    parser.add_argument('--input-dir', type=str, default='examples', help='Input directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = HybridLaneDetector()
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"lane_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Process images
    if args.image:
        # Single image
        detector.process_image(args.image, output_dir)
    else:
        # Multiple images from directory
        input_dir = args.input_dir if os.path.exists(args.input_dir) else '.'
        
        image_files = []
        for file in sorted(os.listdir(input_dir)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(input_dir, file))
        
        if args.count:
            image_files = image_files[:args.count]
        
        logger.info(f"Processing {len(image_files)} images...")
        
        for idx, img_path in enumerate(image_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Image {idx}/{len(image_files)}")
            logger.info(f"{'='*60}")
            
            try:
                detector.process_image(img_path, output_dir)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
    
    logger.info(f"\nAll done! Results in: {output_dir}")


if __name__ == "__main__":
    main()
