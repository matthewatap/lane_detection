import os
import base64
import json
from PIL import Image, ImageDraw, ImageFont
import openai

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def draw_overlay(image_path, regions, output_path):
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for region in regions:
        # Convert percent bbox to absolute pixels
        left, top, right, bottom = region["region"]
        left = int(left * w / 100)
        top = int(top * h / 100)
        right = int(right * w / 100)
        bottom = int(bottom * h / 100)
        color = {
            "white_solid": (255,255,255),
            "white_dashed": (200,200,200),
            "yellow_solid": (255,255,0),
            "yellow_dashed": (180,180,0),
            "road_edge": (255,0,0)
        }.get(region["type"], (0,255,0))
        # Draw rectangle
        draw.rectangle([left, top, right, bottom], outline=color, width=4)
        # Label
        label = region["type"]
        draw.text((left+4, top+4), label, fill=color, font=font)
    im.save(output_path)
    print(f"Overlay saved: {output_path}")

def get_lane_regions_with_openai(image_path, api_key):
    openai.api_key = api_key
    image_b64 = encode_image_to_base64(image_path)
    prompt = (
        "Analyze this road image and return all visible lane markings and road edges. "
        "For each, return a JSON with: "
        "type (white_solid, white_dashed, yellow_solid, yellow_dashed, road_edge), "
        "description, region as [left_x, top_y, right_x, bottom_y] in percentage (0-100), "
        "orientation (vertical/horizontal), curved (true/false). "
        "Output ONLY a JSON array."
    )
    response = openai.chat.completions.create(
        model="o3",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ],
            }
        ],
        max_tokens=2048,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    # Extract JSON array
    json_start = content.find('[')
    json_end = content.rfind(']')
    if json_start == -1 or json_end == -1:
        raise Exception("Failed to parse OpenAI response as JSON")
    return json.loads(content[json_start:json_end+1])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lane overlay with OpenAI Vision")
    parser.add_argument("--image", required=True, help="Input image file")
    parser.add_argument("--output", default="lane_overlay.jpg", help="Output image file")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    print("Calling OpenAI Vision model...")
    regions = get_lane_regions_with_openai(args.image, api_key)
    print(json.dumps(regions, indent=2))

    draw_overlay(args.image, regions, args.output)
