import os
from huggingface_hub import hf_hub_download
from transformers import BlipProcessor, BlipForConditionalGeneration

def download_tiny_llama_gguf(destination_folder="."):
    """
    Downloads TinyLlama-1.1B-Chat-v0.4-Q4_K_M.gguf from TheBloke repo on Hugging Face.
    """
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v0.4-GGUF"
    filename = "TinyLlama-1.1B-Chat-v0.4-Q4_K_M.gguf"

    print("Downloading TinyLlama GGUF file...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=destination_folder,
        local_dir_use_symlinks=False
    )
    print(f"TinyLlama GGUF downloaded to {local_path}")

def download_blip_base(destination_folder="."):
    """
    Downloads BLIP base model and processor weights to cache directory.
    """
    model_name = "Salesforce/blip-image-captioning-base"

    print("Downloading BLIP base model...")
    processor = BlipProcessor.from_pretrained(model_name, cache_dir=destination_folder)
    model = BlipForConditionalGeneration.from_pretrained(model_name, cache_dir=destination_folder)
    print("BLIP base model downloaded and cached.")

if __name__ == "__main__":
    # You can change this folder path if you want to save elsewhere
    working_dir = os.getcwd()

    download_tiny_llama_gguf(destination_folder=working_dir)
    download_blip_base(destination_folder=working_dir)

    print("🎉 All models downloaded successfully.")
