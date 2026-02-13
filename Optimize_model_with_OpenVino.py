# -*- coding: utf-8 -*-
"""
A Multi-Modal AI Assistant Web Application
------------------------------------------
This Flask application serves a web interface for an AI assistant that can
understand and respond to text, images, and voice. It uses a local Llama GGUF
model for language tasks and an OpenVINO-optimized BLIP model for fast
image analysis. The application also includes a feedback mechanism and saves
conversation history.

Author: [N Narayan]
Date: 25 June 2025
"""

# --- Core Imports ---
import os
import json
import datetime
from PIL import Image

# --- Web Framework ---
from flask import Flask, render_template, request, jsonify

# --- AI & Machine Learning Imports ---
from llama_cpp import Llama
import numpy as np
import torch # Maintained for compatibility, though OpenVINO handles core inference
from transformers import BlipProcessor
from openvino.runtime import Core

# ==============================================================================
# --- APPLICATION CONFIGURATION ---
# ==============================================================================
# Folder to store user-uploaded images
UPLOAD_FOLDER = 'uploads'

# JSON files for persistent storage
SAVED_RESULTS_FILE = 'saved_results.json'
FEEDBACK_FILE = 'feedback_data.json'

# --- Model Paths (IMPORTANT: Update these paths to match your system) ---

# 1. Path to your local LLM (e.g., TinyLlama, Mistral, etc. in GGUF format)
LLAMA_MODEL_PATH = r"C:\Users\91910\TinyLlama-1.1B-Chat-v0.4-GGUF\TinyLlama-1.1B-Chat-v0.4-Q4_K_M.gguf"

# 2. Directory containing the OpenVINO-optimized BLIP model files
#    This directory should have 'blip_caption.xml' and 'blip_caption.bin'
BLIP_OV_MODEL_DIR = "blip_ov"

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==============================================================================
# --- MODEL INITIALIZATION ---
# ==============================================================================
# We'll initialize our models as global variables to load them only once.
llm = None
blip_processor, compiled_blip_model = None, None
device = "cpu" # Set device context

print("--- Initializing AI Models ---")

# --- 1. Load the Language Model (Llama GGUF) ---
try:
    if os.path.isfile(LLAMA_MODEL_PATH):
        print(f"🧠 Loading Llama model from: {LLAMA_MODEL_PATH}")
        # n_gpu_layers=0 means we're running this on the CPU.
        # Adjust n_threads based on your CPU cores for better performance.
        llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048, n_threads=8, n_gpu_layers=0)
        print("✅ Llama model loaded successfully.")
    else:
        print(f"❌ FATAL ERROR: Llama model file not found at '{LLAMA_MODEL_PATH}'")
except Exception as e:
    print(f"❌ FATAL ERROR loading Llama model: {e}")

# --- 2. Load the Image Analysis Model (BLIP) ---
# We use a two-step process:
#   a) Load the pre-processor from Hugging Face Transformers.
#   b) Load the compute-heavy model optimized with OpenVINO for CPU performance.
BLIP_MODEL_XML = os.path.join(BLIP_OV_MODEL_DIR, "blip_caption.xml")
BLIP_MODEL_BIN = os.path.join(BLIP_OV_MODEL_DIR, "blip_caption.bin")

try:
    print("\n👁️ Loading BLIP image analysis model...")
    # The processor handles text tokenization and image transformations.
    processor_name = "Salesforce/blip-image-captioning-base"
    blip_processor = BlipProcessor.from_pretrained(processor_name)
    print(f"✅ BLIP processor '{processor_name}' loaded successfully.")

    # Now, load and compile the main model using Intel's OpenVINO for a speed boost.
    if os.path.isfile(BLIP_MODEL_XML) and os.path.isfile(BLIP_MODEL_BIN):
        print(f"   -> Loading OpenVINO BLIP model from: {BLIP_MODEL_XML}")
        ie = Core()
        model = ie.read_model(model=BLIP_MODEL_XML, weights=BLIP_MODEL_BIN)
        compiled_blip_model = ie.compile_model(model=model, device_name="CPU")
        print("✅ OpenVINO BLIP model loaded and compiled for CPU.")
    else:
        print(f"❌ FATAL ERROR: OpenVINO model files not found.")
        print(f"   -> Searched for XML at: '{BLIP_MODEL_XML}'")
        print(f"   -> Searched for BIN at: '{BLIP_MODEL_BIN}'")
except Exception as e:
    print(f"❌ FATAL ERROR loading BLIP components: {e}")

print("\n--- Model initialization complete. ---\n")


# ==============================================================================
# --- FLASK APPLICATION SETUP ---
# ==============================================================================
# Configure the Flask app to serve static files (like CSS, JS, and uploaded images)
app = Flask(__name__, static_folder='.', static_url_path='/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "a_super_secret_key_for_your_flask_app"


# ==============================================================================
# --- HELPER FUNCTIONS (DATA & AI) ---
# ==============================================================================

# --- Data Persistence Functions ---

def load_json_data(file_path):
    """A robust function to load data from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json_data(data, file_path):
    """Saves Python data (list/dict) to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def save_result_to_file(query, solution, timestamp):
    """Appends a new query-solution pair to the history file."""
    results = load_json_data(SAVED_RESULTS_FILE)
    results.append({"query": query, "solution": solution, "timestamp": timestamp})
    save_json_data(results, SAVED_RESULTS_FILE)

def save_feedback_to_file(query, solution, feedback_type, timestamp):
    """Saves user feedback to its dedicated file."""
    feedback = load_json_data(FEEDBACK_FILE)
    feedback.append({
        "query": query,
        "solution": solution,
        "feedback_type": feedback_type,
        "timestamp": timestamp
    })
    save_json_data(feedback, FEEDBACK_FILE)

# --- AI Inference Functions ---

def get_image_caption(image_path):
    """
    Generates a descriptive caption for an image using the optimized BLIP model.
    This function mirrors the logic of a standard Hugging Face pipeline but uses
    the faster OpenVINO backend for inference.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: A text description of the image.
    """
    if not compiled_blip_model or not blip_processor:
        return "[Image analysis model is not available]"

    print(f"   -> Generating caption for '{image_path}'...")
    try:
        # Open the image using PIL
        image = Image.open(image_path).convert("RGB")

        # 1. Pre-process the image and a starting text prompt.
        #    The prompt "a photo of" guides the model to generate a caption.
        #    We specify `return_tensors="np"` because OpenVINO works with NumPy arrays.
        inputs = blip_processor(images=image, text=["a photo of"], return_tensors="np")

        # 2. Dynamically map the processed inputs to the model's expected input names.
        #    This makes the code more robust if the model's input names change.
        input_keys = list(compiled_blip_model.inputs)
        input_dict = {
            input_keys[0].get_any_name(): inputs["input_ids"],
            input_keys[1].get_any_name(): inputs["attention_mask"],
            input_keys[2].get_any_name(): inputs["pixel_values"],
        }

        # 3. Run inference through the compiled OpenVINO model.
        outputs = compiled_blip_model(input_dict)

        # 4. Decode the output. OpenVINO returns raw logits. We perform greedy
        #    decoding by finding the token with the highest probability at each step.
        output_ids = np.argmax(list(outputs.values())[0], axis=-1)

        # 5. Convert the token IDs back into a human-readable string.
        caption = blip_processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"   -> Generated Caption: '{caption.strip()}'")
        return caption.strip()

    except Exception as e:
        print(f"Error during image captioning: {e}")
        return "[Error analyzing image with OpenVINO]"

def get_llm_response(prompt_text, user_query):
    """
    Queries the local Llama model for a response, using a refined chat template
    and considering past user feedback.

    Args:
        prompt_text (str): The full prompt to send to the LLM.
        user_query (str): The original user query, used to check for feedback.

    Returns:
        str: The generated text from the LLM.
    """
    if not llm:
        return "Error: The Language Model is not loaded. Please check the server console."

    # Smart Feature: Check if we have received "good" feedback for this exact query before.
    # If so, we can just return the previously approved answer instantly.
    feedback_data = load_json_data(FEEDBACK_FILE)
    for entry in feedback_data:
        if entry["query"] == user_query and entry["feedback_type"] == "good":
            print(f"✅ Found 'good' feedback for query. Returning saved solution.")
            return entry["solution"]

    try:
        # Use a chat-optimized template for more reliable and structured responses.
        # This helps the model understand the roles of "user" and "assistant".
        formatted_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        print(f"   -> Sending to LLM: '{formatted_prompt[:150]}...'") # Log a snippet

        response = llm(
            formatted_prompt,
            max_tokens=500,       # Max length of the generated response
            temperature=0.7,      # Controls randomness: lower is more deterministic
            top_p=0.9,            # Nucleus sampling for diverse responses
            stop=["<|im_end|>"],  # Tell the model when to stop talking
            echo=False            # Don't repeat the prompt in the output
        )
        generated_text = response["choices"][0]["text"].strip()
        return generated_text
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        return "Sorry, I encountered an error while generating a response."


# ==============================================================================
# --- FLASK WEB ROUTES ---
# ==============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the main page logic, processing user inputs and generating AI responses.
    """
    # Initialize variables for rendering the page
    context = {
        "solution": None,
        "feedback_message": None,
        "selected_input_type": request.form.get('input_type', 'text'),
        "user_query": "",
        "image_filename_display": None,
        "original_query_for_feedback_hidden": "",
        "original_solution_hidden": "",
        "original_input_type_hidden": "text"
    }

    if request.method == 'POST':
        # --- CASE 1: The user is submitting feedback on a previous answer ---
        if 'feedback' in request.form:
            query = request.form.get('original_query_for_feedback_hidden')
            solution = request.form.get('original_solution_hidden')
            feedback_type = request.form.get('feedback')
            timestamp = datetime.datetime.now().isoformat()

            save_feedback_to_file(query, solution, feedback_type, timestamp)

            # Prepare context to re-display the page with a thank you message
            context["feedback_message"] = "Thank you for your feedback!"
            context["solution"] = solution
            context["user_query"] = query
            context["selected_input_type"] = request.form.get('original_input_type_hidden')

        # --- CASE 2: The user is submitting a new query ---
        else:
            llm_prompt = ""
            user_query = ""
            context["selected_input_type"] = request.form.get('input_type')

            # Process based on the selected input tab (Text, Image, or Voice)
            if context["selected_input_type"] == 'text':
                user_query = request.form.get('text_input', '').strip()
                if user_query:
                    llm_prompt = user_query
                else:
                    context["feedback_message"] = "Please enter your question in the text box."

            elif context["selected_input_type"] == 'image':
                user_query_for_image = request.form.get('text_input_for_image', '').strip()
                image_file = request.files.get('image_file')

                if image_file and image_file.filename != '':
                    filename = f"uploaded_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{image_file.filename}"
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image_file.save(image_path)
                    context["image_filename_display"] = filename

                    # Get the image caption first
                    image_context = get_image_caption(image_path)
                    
                    # Create a detailed prompt for the LLM
                    if user_query_for_image:
                        # If there's a specific question about the image
                        llm_prompt = f"Based on the image which shows '{image_context}', answer the following question: {user_query_for_image}"
                        user_query = user_query_for_image # The core query for history/feedback
                    else:
                        # If there's no question, just ask for a description
                        llm_prompt = f"Describe the following image in detail. The image shows: '{image_context}'"
                        user_query = f"Analysis of image (auto-detected: {image_context})"
                else:
                    context["feedback_message"] = "Please upload an image to continue."

            elif context["selected_input_type"] == 'voice':
                # The client-side Javascript will use Speech Recognition and put the result in 'text_input'
                user_query = request.form.get('text_input', '').strip()
                if user_query:
                    llm_prompt = user_query
                else:
                    context["feedback_message"] = "Voice not detected. Please try speaking again."

            # If a valid prompt was created, run the AI models
            if llm_prompt:
                print(f"▶️  Processing new query. Input Type: '{context['selected_input_type']}'")
                solution = get_llm_response(llm_prompt, user_query)
                context["solution"] = solution
                context["user_query"] = user_query
                context["original_query_for_feedback_hidden"] = user_query
                context["original_solution_hidden"] = solution
                context["original_input_type_hidden"] = context["selected_input_type"]
            elif not context["feedback_message"]:
                context["feedback_message"] = "Please provide some input to get started."

    # Load previous results to display in the history panel
    context["saved_results"] = load_json_data(SAVED_RESULTS_FILE)
    
    # Determine if the feedback form should be visible
    context["show_feedback_form"] = context["solution"] and "Error" not in context["solution"]

    return render_template('index.html', **context)


@app.route('/save_result', methods=['POST'])
def save_result_endpoint():
    """
    An API endpoint called by JavaScript to save a result to the history file.
    """
    data = request.get_json()
    query = data.get('query')
    solution = data.get('solution')
    if query and solution:
        timestamp = datetime.datetime.now().isoformat()
        save_result_to_file(query, solution, timestamp)
        return jsonify({"status": "success", "message": "Result saved!"})
    return jsonify({"status": "error", "message": "Invalid data received"}), 400


# ==============================================================================
# --- APPLICATION ENTRY POINT ---
# ==============================================================================
if __name__ == '__main__':
    # Final check: Make sure our critical models have loaded before starting the server.
    if not llm or not compiled_blip_model:
        print("\n" + "="*60)
        print("    CRITICAL ERROR: One or more AI models failed to load.    ")
        print("    The application cannot start. Please review the logs    ")
        print("    above, check your model paths, and try again.           ")
        print("="*60)
    else:
        print("\n🚀 Starting Flask server...")
        # 'host="0.0.0.0"' makes the server accessible on your local network.
        # 'debug=True' is great for development but should be turned off for production.
        app.run(debug=True, host='0.0.0.0', port=5000)
