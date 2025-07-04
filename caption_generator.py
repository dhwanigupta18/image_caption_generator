# caption_generator.py
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
#from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

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
            # Fallback if font not found 
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

#load_dotenv()

class WatsonXEnhancer:
    def __init__(self, model_id="meta-llama/llama-3-2-11b-vision-instruct"):
        """
        Initializes the IBM watsonx.ai text generation model.
        Args:
            model_id (str): The ID of the foundation model to use (e.g., "ibm/granite-13b-instruct-v2").
        """
        self.api_key = os.getenv("IBM_CLOUD_API_KEY")
        self.url = "https://us-south.ml.cloud.ibm.com"
        self.project_id = os.getenv("WATSONX_AI_PROJECT_ID")
        self.model_id = model_id

         # --- ADD THESE PRINT STATEMENTS ---
        print("\n--- WatsonXEnhancer Initialization Debug ---")
        print(f"Attempting to load .env from: {os.getcwd()}")
        print(f"IBM_CLOUD_API_KEY (first 5 chars): {self.api_key[:5] if self.api_key else 'None/Empty'}")
        print(f"WATSONX_AI_URL: {self.url if self.url else 'None/Empty'}")
        print(f"WATSONX_AI_URL (HARDCODED): '{self.url}' (Length: {len(self.url) if self.url else 'None'})")
        print(f"WATSONX_AI_PROJECT_ID: {self.project_id if self.project_id else 'None/Empty'}")
        print("-------------------------------------------\n")
        # --- END DEBUG PRINTS ---

        if not all([self.api_key, self.url, self.project_id]):
            print("Warning: IBM Watsonx.ai credentials (API key, URL, Project ID) are not fully configured in .env. Watsonx refinement will not work.")
            self.model = None
            return

        try:
            # Authenticate with IBM Cloud IAM
            authenticator = IAMAuthenticator(self.api_key)

            # Initialize the Model class for text generation
            self.model = ModelInference(
                model_id=self.model_id,
                credentials={
                    "url": self.url,
                    "apikey": self.api_key # The SDK handles token refreshing internally
                },
                project_id=self.project_id,
                # You might need to specify the space ID if your model is deployed in a space
                # space_id="YOUR_WATSONX_AI_SPACE_ID"
            )
            print(f"IBM Watsonx.ai Enhancer initialized with model: {self.model_id}")

        except Exception as e:
            self.model = None
            print(f"Error initializing IBM Watsonx.ai Enhancer: {e}")
            print("Please check your API key, URL, Project ID, and model ID.")


    def refine_caption(self, caption, purpose="general description"):
        """
        Refines the given caption using an IBM Watsonx.ai foundation model.
        Args:
            caption (str): The caption generated by BLIP.
            purpose (str): The intended use or style for the caption.
        Returns:
            str: The refined caption from the LLM.
        """
        if not self.model:
            return "IBM Watsonx.ai Enhancer not configured due to initialization errors."

        # Construct the prompt for the LLM
        # Adjust the prompt based on the chosen model's ideal input format (e.g., instruct vs. chat)
        # For instruct models (like Granite-Instruct, Llama-2-Chat, Mistral-Instruct):
        # A simple instruction-based prompt usually works well.
        prompt = (
            f"You are an AI assistant. Your task is to enhance and refine an image caption. "
            f"Make it more suitable for a '{purpose}'.\n\n"
            f"Initial Caption: '{caption}'\n\n"
            f"Refined Caption:"
        )

        # You can define generation parameters
        # These are common parameters; adjust as needed
        generate_params = {
            GenParams.MAX_NEW_TOKENS: 100,
            GenParams.TEMPERATURE: 0.7,
            GenParams.TOP_P: 1.0,
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE # Or DecodingMethods.GREEDY
        }

        try:
            # Use the generate method for text generation
            # For chat-optimized models, you might use model.chat() with a messages array
            # Refer to watsonx.ai documentation for specific model capabilities (chat vs. generation endpoint)
            generated_response = self.model.generate_text(
                prompt=prompt,
                params=generate_params
            )

            refined_caption = generated_response.strip()
            return refined_caption
        except Exception as e:
            return f"Error during IBM Watsonx.ai refinement: {e}"