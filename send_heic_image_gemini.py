
import base64
from pathlib import Path
from PIL import Image
import pillow_heif
from typing import Dict
import os
import re
import tempfile
from google import genai
from google.genai import types


GEMINI_API_KEY = 'AIzaSyAADGhj_yxeBq7--Y6lU82xg2n26j8u_LM'

SEARCH_PROMPT = 'You are an expert in identifying the make of all items whch can be bought, and their model and year of release. You have been trained all your life for this! Always use the web to find your answers: scour the internet and find the correct answer!'

client = genai.Client(api_key=GEMINI_API_KEY)


def generate(base64_image):
    client = genai.Client(
        api_key=GEMINI_API_KEY,
    )

    model = "gemini-2.5-pro-preview-05-06"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(
                        base64_image
                    ),
                ),
                types.Part.from_text(text="""You are an expert in identifying the make of all items whch can be bought, and their model and year of release. You have been trained all your life for this! Always use the web to find your answers: scour the internet and find the correct answer!

Tell me what this product is. Give me the exact model, make and year of release. Return nothing else apart from this information!

Think deeply and use your judgement highly effectively. This is your top priority!"""),
            ],
        ),
    ]
    tools = [
        types.Tool(url_context=types.UrlContext()),
        types.Tool(google_search=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
        tools=tools,
        response_mime_type="text/plain",
    )

    out = ''
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        out = out + chunk.text

    return out



def resize_heic_to_500x500(input_path, output_path):
    """
    Resize a HEIC image to 500x500 pixels
    """
    try:
        # Open the HEIC image
        with Image.open(input_path) as img:
            # Convert to RGB if needed (HEIC might be in different color space)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Simple resize to exactly 500x500
            img = img.resize((500, 500), Image.Resampling.LANCZOS)

            # Save as JPEG (more compatible than HEIC for API)
            img.save(output_path, 'JPEG', quality=85)
            return True
    except Exception as e:
        print(f"Error resizing image: {e}")
        return False


def send_message_with_heic_image(heic_file_path: str) -> Dict:
    """
    Send a message to Claude API with an HEIC image attachment
    Args:
        api_key: Your Anthropic API key
        message: The message to send
        heic_file_path: Path to the HEIC image file
    Returns:
        Response dictionary
    """

    try:
        # Register HEIF opener with Pillow
        pillow_heif.register_heif_opener()

        # Convert HEIC to JPEG for Claude API compatibility
        heic_path = Path(heic_file_path)
        if not heic_path.exists():
            raise FileNotFoundError(f"HEIC file not found: {heic_file_path}")

        # Open HEIC and convert to JPEG
        with Image.open(heic_path) as img:
            # Convert to RGB if necessary (HEIC might be in different color space)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Save as JPEG to bytes
            import io
            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format='JPEG', quality=95)
            jpeg_bytes = jpeg_buffer.getvalue()

            ## rest of the below is to convert the image
            img = Image.open(io.BytesIO(jpeg_buffer.getvalue()))

            # Resize to 500x500
            # LANCZOS is high quality resampling (PIL.Image.Resampling.LANCZOS in newer versions)
            resized_img = img.resize((500, 500), Image.LANCZOS)

            # Convert back to JPEG buffer
            output_buffer = io.BytesIO()
            resized_img.save(output_buffer, format='JPEG')
            output_buffer.seek(0)
            jpeg_bytes = output_buffer.getvalue()

        # Encode to base64
        image_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')

        text_response = generate(image_base64)

        return {
            "response": text_response,
            #"usage": response.usage.__dict__,
            "success": True
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }

# Usage
if __name__ == "__main__":

    # Directory containing HEIC files
    DOWNLOADS_DIR = "../../downloads"

    # Get all .heic files that don't end with a number before the extension
    heic_files = []
    for filename in os.listdir(DOWNLOADS_DIR):
        if filename.lower().endswith('.heic'):
            # Check if filename ends with number before .heic (e.g., "file1.heic", "IMG_123.heic")
            # This regex checks if there's NOT a digit immediately before .heic
            if not re.search(r'\d\.heic$', filename.lower()):
                heic_files.append(os.path.join(DOWNLOADS_DIR, filename))

    print(f"Found {len(heic_files)} HEIC files not ending with numbers:")
    for file_path in heic_files:
        print(f"  {os.path.basename(file_path)}")

    # Process each file
    for HEIC_FILE_PATH in heic_files:
        print(f"\nProcessing: {os.path.basename(HEIC_FILE_PATH)}")

        keywords = ['lamp', 'adapter', 'dale', 'frame']
        if not any(keyword in HEIC_FILE_PATH for keyword in keywords):
            continue

        # Send the message with image
        result = send_message_with_heic_image(HEIC_FILE_PATH)

        # Print result
        if result['success']:
            print(f"Response: {result['response']}")
            #print(f"Tokens used: {result['usage']}")

            # Save to file with unique name
            base_name = os.path.splitext(os.path.basename(HEIC_FILE_PATH))[0]
            output_filename = f"outputs/geminiflash_with_image_{base_name}_search.txt"

            os.makedirs("outputs", exist_ok=True)  # Create outputs directory if it doesn't exist
            with open(output_filename, 'w') as output:
                output.write(result['response'])
            print(f"Saved to: {output_filename}")
        else:
            print(f"Failed: {result['error']}")
