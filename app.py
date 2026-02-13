from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
from PIL import Image
from llama_cpp import Llama
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch # type: ignore
import datetime

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
SAVED_RESULTS_FILE = 'saved_results.json'
FEEDBACK_FILE = 'feedback_data.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Initialize Models ---

# 1. LLM (Llama GGUF)
# !!! IMPORTANT: Make sure this path to your GGUF model file is correct !!!
LLAMA_MODEL_PATH = r"C:\Users\91910\TinyLlama-1.1B-Chat-v0.4-GGUF\TinyLlama-1.1B-Chat-v0.4-Q4_K_M.gguf"

# 2. Image Analysis Model (BLIP)
# !!! IMPORTANT: Paste the absolute path from Step 3 of the instructions here !!!
BLIP_MODEL_PATH = r"C:/AI_Models/blip-base" # <-- CHANGE THIS

llm = None
blip_processor, blip_model = None, None
device = "cpu"

# Load Llama model
try:
    if os.path.isfile(LLAMA_MODEL_PATH):
        print(f"Loading Llama model from: {LLAMA_MODEL_PATH}")
        llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048, n_threads=8, n_gpu_layers=0)
        print("✅ Llama model loaded successfully.")
    else:
        print(f"❌ ERROR: Llama model file not found at '{LLAMA_MODEL_PATH}'")
except Exception as e:
    print(f"❌ ERROR loading Llama model: {e}")

# Load BLIP model from the absolute path
try:
    if os.path.isdir(BLIP_MODEL_PATH):
        print(f"Loading BLIP model from: {BLIP_MODEL_PATH}")
        blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH)
        blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_PATH).to(device)
        blip_model.eval()
        print("✅ BLIP image model loaded successfully.")
    else:
        print(f"❌ ERROR: BLIP model directory not found at '{BLIP_MODEL_PATH}'. Please run the download script.")
except Exception as e:
    print(f"❌ ERROR loading BLIP model: {e}")

app = Flask(__name__, static_folder='.', static_url_path='/') # Configure static files to be served from root
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "supersecretkey_for_flask"

# --- Data Storage Functions (JSON files for simplicity) ---

def load_saved_results():
    if os.path.exists(SAVED_RESULTS_FILE):
        with open(SAVED_RESULTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_result_to_file(query, solution, timestamp):
    results = load_saved_results()
    results.append({"query": query, "solution": solution, "timestamp": timestamp})
    with open(SAVED_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)

def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return []

def save_feedback_to_file(query, solution, feedback_type, timestamp):
    feedback = load_feedback_data()
    feedback.append({
        "query": query,
        "solution": solution,
        "feedback_type": feedback_type,
        "timestamp": timestamp
    })
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback, f, indent=4)

# --- Model Inference Functions ---

@torch.no_grad()
def get_image_description(image_path):
    """Uses the BLIP model to generate a description of the image."""
    if not blip_model or not blip_processor:
        return "[Image analysis model is not loaded]"
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = blip_processor(raw_image, return_tensors="pt").to(device)
        output_ids = blip_model.generate(**inputs, max_new_tokens=50)
        description = blip_processor.decode(output_ids[0], skip_special_tokens=True)
        return description.strip()
    except Exception as e:
        print(f"Error during image description: {e}")
        return "[Error analyzing image]"

def query_llm(prompt_text, user_query):
    """
    Runs inference with your Llama CPP model using a robust chat template.
    Considers past feedback for exact query matches.
    """
    if not llm:
        return "Error: LLM not loaded. Please check the server console."

    # 1. Check for exact match in "good" feedback
    feedback_data = load_feedback_data()
    for entry in feedback_data:
        if entry["query"] == user_query and entry["feedback_type"] == "good":
            print(f"✅ Found exact 'good' feedback for query: '{user_query}'. Returning saved solution.")
            return entry["solution"]

    # 2. Check for exact match in "bad" feedback (optional: could trigger re-generation or specific handling)
    # For now, if "bad" feedback exists, we still try to generate a new response.
    # A more sophisticated system might block, or apply different prompt strategies.

    try:
        # Using a more robust chat template for better and more stable responses
        formatted_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        print(f"Sending to LLM: '{formatted_prompt}'")

        response = llm(
            formatted_prompt,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            stop=["<|im_end|>", "user:", "assistant:"], # Stop token for this template
            echo=False
        )
        generated_text = response["choices"][0]["text"].strip()
        return generated_text
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        return "Error generating solution from LLM."

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    solution = None
    feedback_message = None
    input_type = request.form.get('input_type', 'text')
    user_query = ""
    image_filename_display = None
    original_query_for_feedback = ""
    original_solution_for_feedback = "" # New variable for feedback

    if request.method == 'POST':
        if 'feedback' in request.form:
            query = request.form.get('original_query_for_feedback_hidden')
            sol = request.form.get('original_solution_hidden')
            feedback_type = request.form.get('feedback')
            timestamp = datetime.datetime.now().isoformat()
            save_feedback_to_file(query, sol, feedback_type, timestamp)
            feedback_message = "Thank you for your feedback!"
            # No new solution is generated after feedback, so render with existing context
            solution = sol
            user_query = query
            input_type = request.form.get('original_input_type_hidden')

        else: # Regular submission for a new query
            input_type = request.form.get('input_type')
            llm_prompt = ""

            if input_type == 'text':
                user_query = request.form.get('text_input', '').strip()
                if user_query:
                    llm_prompt = user_query
                else:
                    feedback_message = "Please enter your text question."

            elif input_type == 'image':
                user_query_for_image = request.form.get('text_input_for_image', '').strip()
                image_file = request.files.get('image_file')
                if image_file and image_file.filename != '':
                    filename = "uploaded_" + image_file.filename
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image_file.save(image_path)
                    image_filename_display = filename
                    
                    print("Analyzing image with BLIP...")
                    image_context = get_image_description(image_path)
                    print(f"Image Context: {image_context}")

                    if user_query_for_image:
                        llm_prompt = f"Based on the image which shows '{image_context}', answer this question: {user_query_for_image}"
                        user_query = user_query_for_image
                    else:
                        llm_prompt = f"Describe the following image in detail. The image shows: '{image_context}'"
                        user_query = f"Analysis of uploaded image (detected: {image_context})"
                else:
                    feedback_message = "Please upload an image for this input type."
            
            elif input_type == 'voice':
                # For voice, we expect the text_input to be populated by client-side SR
                user_query = request.form.get('text_input', '').strip()
                if user_query:
                    llm_prompt = user_query
                else:
                    feedback_message = "No voice input detected or transcribed."

            original_query_for_feedback = user_query or llm_prompt

            if llm_prompt and not feedback_message:
                solution = query_llm(llm_prompt, user_query) # Pass user_query for feedback lookup
                original_solution_for_feedback = solution
            elif not feedback_message and request.method == 'POST':
                feedback_message = "Please provide valid input to get a solution."

    saved_results = load_saved_results()

    return render_template('index.html',
                           solution=solution,
                           feedback_message=feedback_message,
                           selected_input_type=input_type,
                           user_query=user_query,
                           image_filename_display=image_filename_display,
                           show_feedback_form=(solution and "Error:" not in solution),
                           original_query_for_feedback_hidden=original_query_for_feedback,
                           original_solution_hidden=original_solution_for_feedback,
                           original_input_type_hidden=input_type,
                           saved_results=saved_results)

@app.route('/save_result', methods=['POST'])
def save_result():
    data = request.get_json()
    query = data.get('query')
    solution = data.get('solution')
    if query and solution:
        timestamp = datetime.datetime.now().isoformat()
        save_result_to_file(query, solution, timestamp)
        return jsonify({"status": "success", "message": "Result saved!"})
    return jsonify({"status": "error", "message": "Invalid data"}), 400

if __name__ == '__main__':
    if not llm or not blip_model:
        print("CRITICAL: One or more models failed to load. Please check paths and error messages.")
    # Add use_reloader=False to prevent crashes caused by the reloader
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)