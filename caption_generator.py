# caption_generator.py
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioningModel:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """
        Initializes the BLIP image captioning model.
        Args:
            model_name (str): The name of the pre-trained BLIP model to use.
                              "Salesforce/blip-image-captioning-base" is a good default.
        """
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"BLIP model loaded on device: {self.device}")

    def generate_caption(self, image_path, prompt=None):
        """
        Generates a caption for the given image.
        Args:
            image_path (str): Path to the input image file.
            prompt (str, optional): An optional text prompt to guide the caption generation.
                                    For example, "a photography of".
        Returns:
            str: The generated image caption.
        """
        try:
            raw_image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            return f"Error: Image not found at {image_path}"
        except Exception as e:
            return f"Error opening image: {e}"

        if prompt:
            # Conditional captioning
            inputs = self.processor(raw_image, text=prompt, return_tensors="pt").to(self.device)
            # You might experiment with num_beams and max_length for better results
            out = self.model.generate(**inputs, num_beams=3, max_length=50)
        else:
            # Unconditional captioning
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, num_beams=3, max_length=50)

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

# Example usage (for testing)
if __name__ == "__main__":
    # Create a dummy image for testing
    from PIL import Image, ImageDraw, ImageFont
    import os

    dummy_image_path = "test_image.png"
    if not os.path.exists(dummy_image_path):
        img = Image.new('RGB', (600, 400), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        try:
            # Try to load a default font
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            # Fallback if font not found (e.g., on some Linux systems)
            font = ImageFont.load_default()
        d.text((50,150), "A red car parked on a street with trees.", fill=(255,255,0), font=font)
        img.save(dummy_image_path)
        print(f"Created dummy image: {dummy_image_path}")

    caption_model = ImageCaptioningModel()

    # Generate unconditional caption
    print(f"\nCaption for '{dummy_image_path}' (unconditional):")
    caption = caption_model.generate_caption(dummy_image_path)
    print(caption)

    # Generate conditional caption
    print(f"\nCaption for '{dummy_image_path}' (with prompt 'a photo of'):")
    caption_with_prompt = caption_model.generate_caption(dummy_image_path, prompt="a photo of")
    print(caption_with_prompt)

    # Test with a non-existent image
    print("\nTesting with non-existent image:")
    non_existent_caption = caption_model.generate_caption("non_existent_image.jpg")
    print(non_existent_caption)

    # Clean up dummy image
    if os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)
        print(f"Removed dummy image: {dummy_image_path}")