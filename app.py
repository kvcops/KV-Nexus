from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
import logging
from langdetect import detect

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("API_KEY")
openweathermap_api_key = os.getenv('OPENWEATHERMAP_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)

if not api_key:
    logging.error("API_KEY environment variable not set")
    raise EnvironmentError("API_KEY environment variable not set")

genai.configure(api_key=api_key)
model_text = genai.GenerativeModel('gemini-pro')
model_vision = genai.GenerativeModel('gemini-pro-vision')
user_data = {}

# Define generation configurations
generation_configs = {
    "chat": {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048},
    "chef": {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048},
    "story": {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048},
    "psychology": {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 1024},
    "code": {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048},
    "algorithm": {"temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 1024},
}

models = {key: genai.GenerativeModel("gemini-pro", generation_config=config) for key, config in generation_configs.items()}

def format_response(response_text):
    """Formats the response text for display, handling potential language issues."""
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    try:
        language = detect(response_text)
        formatted_text = '<br>'.join(lines)
        logging.info(f"Detected language: {language}")
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        formatted_text = '<br>'.join(lines)
    return formatted_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/weather')
def get_weather():
    if not openweathermap_api_key:
        logging.error("OPENWEATHERMAP_API_KEY environment variable not set")
        return jsonify({'error': "Weather API key not set"}), 500

    ip = request.headers.get('x-real-ip')  # Get client's IP address
    # Your weather API logic here using the openweathermap_api_key and ip
    # Return the weather information as JSON
    return jsonify({'weather': 'sunny', 'temperature': '25'})

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_message = request.json['message']
        chat_history.append({"role": "user", "message": user_message})
        context = "\n".join([f"{msg['role']}: {msg['message']}" for msg in chat_history])
        prompt = f"{context}\nbot:"
        response = models["chat"].generate_content(prompt)
        reply = response.text.strip()
        chat_history.append({"role": "bot", "message": reply})
        return jsonify({"reply": reply, "chat_history": chat_history})
    return render_template('chat.html')

@app.route('/chef', methods=['GET', 'POST'])
def chef():
    if request.method == 'POST':
        user_ingredients = request.form['user_ingredients']
        prompt = f"Generate a recipe based on the following ingredients {user_ingredients} and explain the steps to cook it in a stepwise manner and formatted manner. Also explain who can eat and who shouldn't eat."
        response = models["chef"].generate_content([prompt])
        response_text = format_response(response.text)
        return jsonify({'response': response_text})
    return render_template('chef.html')

@app.route('/story_generator', methods=['GET', 'POST'])
def story_generator():
    if request.method == 'POST':
        user_input_words = request.form['keywords']
        genre = request.form['genre']
        prompt = f"Generate a story based on the following words {user_input_words} with genre {genre}..."
        try:
            response = models["story"].generate_content([prompt])
            if response.candidates and response.candidates[0].content.parts:
                response_text = format_response(response.candidates[0].content.parts[0].text)
            else:
                logging.error(f"No valid response found for prompt: {prompt}")
                response_text = "Sorry, I couldn't generate a story with the provided input."
            return jsonify({'response': response_text})
        except Exception as e:
            logging.error(f"Error generating story: {e}")
            return jsonify({'error': "Internal Server Error"}), 500
    return render_template('story_generator.html')

@app.route('/psychology_prediction', methods=['GET', 'POST'])
def psychology_prediction():
    if request.method == 'POST':
        # Get user input from the form
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        occupation = request.form['occupation']
        keywords = request.form['keywords']
        # Construct the prompt based on user input
        prompt = f"Predict psychological behavior for {name}, a {age}-year-old {gender} working as a {occupation}. Keywords: {keywords}"
        # Generate prediction using the psychology model
        response = models["psychology"].generate_content([prompt])
        response_text = format_response(response.text)
        return jsonify({'response': response_text})
    return render_template('psychology_prediction.html')

@app.route('/code_generation', methods=['GET', 'POST'])
def code_generation():
    if request.method == 'POST':
        code_type = request.form['codeType']
        language = request.form['language']
        prompt = f"Write a {language} code to implement {code_type}."
        response = models["code"].generate_content([prompt])
        if response.candidates and response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text
        else:
            response_text = "No valid response found."
        return jsonify({'response': response_text})
    return render_template('code_generation.html')

@app.route('/algorithm_generation', methods=['GET', 'POST'])
def algorithm_generation():
    if request.method == 'POST':
        algo = request.form['algorithm']
        prompt = f"Write a Python function to implement the {algo} algorithm. Ensure the function is well-structured and follows best practices for readability and efficiency."
        response = models["algorithm"].generate_content([prompt])
        if response.candidates and response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text
        else:
            response_text = "No valid response found."
        return jsonify({'response': response_text})
    return render_template('algorithm_generation.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        try:
            gender = request.form.get('gender')
            symptoms = request.form.get('symptoms')
            body_part = request.form.get('body-part')
            layer = request.form.get('layer')
            image = request.files.get('image')
            additional_symptoms = request.form.get('additionalSymptoms')
            final_symptoms = request.form.get('finalSymptoms')
            stage = 'initial'
            if additional_symptoms:
                stage = 'intermediate'
                symptoms += f", {additional_symptoms}"
            if final_symptoms:
                stage = 'final'
                symptoms += f", {final_symptoms}"
            user_data['gender'] = gender
            user_data['symptoms'] = symptoms
            user_data['body_part'] = body_part
            user_data['layer'] = layer
            if image:
                filename = secure_filename(image.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(filepath)
                user_data['image_path'] = filepath
                image_analysis_prompt = f"Analyze the image: {filepath}"
                image_response = model_vision.generate_content([image_analysis_prompt])
                user_data['image_analysis'] = image_response.text
            analysis_prompt = f"Gender: {gender}\nSymptoms: {symptoms}\nBody Part: {body_part}\nLayer: {layer}"
            analysis_response = model_text.generate_content([analysis_prompt])
            user_data['analysis'] = analysis_response.text
            response_text = format_response(analysis_response.text)
            return jsonify({'response': response_text, 'stage': stage})
        except Exception as e:
            logging.error(f"Error during analysis: {e}")
            return jsonify({'error': "Internal Server Error"}), 500
    return render_template('analyze.html')

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
