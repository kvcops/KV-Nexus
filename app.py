from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
from PIL import Image
import PIL
from io import BytesIO
import logging
from langdetect import detect
import requests
from requests import get
app = Flask(__name__)

# Load environment variables
load_dotenv()
api_key = os.environ.get("API_KEY")
unsplash_api_key = os.getenv('UNSPLASH_API_KEY')
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
# Set up logging
logging.basicConfig(level=logging.INFO)

# Configure the Google Generative AI API
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')
chat_history = []  # Define chat_history here

model_text = genai.GenerativeModel('gemini-pro')
model_vision = genai.GenerativeModel('gemini-pro-vision')
user_data = {}

# Generation configurations for different models
chat_generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
chef_generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
story_generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
psychology_generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1024,
}
code_generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
algorithm_generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1024,
}

# Create model instances with configurations
chat_model = genai.GenerativeModel("gemini-pro", generation_config=chat_generation_config)
chef_model = genai.GenerativeModel("gemini-pro", generation_config=chef_generation_config)
story_model = genai.GenerativeModel("gemini-pro", generation_config=story_generation_config)
psychology_model = genai.GenerativeModel("gemini-pro", generation_config=psychology_generation_config)
code_model = genai.GenerativeModel("gemini-pro", generation_config=code_generation_config)
algorithm_model = genai.GenerativeModel("gemini-pro", generation_config=algorithm_generation_config)

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
    """
    Gets weather details from OpenWeatherMap API using user's IP address.
    """

    # Get user's IP address
    user_ip = requests.get('https://api.ipify.org').text

    # Use IP-API to get the city name based on IP
    ip_api_url = f"http://ip-api.com/json/{user_ip}"
    ip_api_response = requests.get(ip_api_url)
    if ip_api_response.status_code == 200:
        ip_api_data = ip_api_response.json()
        city = ip_api_data.get('city')
        if not city:
            return jsonify({'error': 'City not found based on IP'}), 404
    else:
        return jsonify({'error': 'Failed to get location from IP'}), 404

    # Build OpenWeatherMap API URL
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    # Make API request
    response = requests.get(url)

    # Check for successful response
    if response.status_code == 200:
        data = response.json()

        # Extract relevant weather information
        weather = {
            'city': data['name'],  # Include the city name
            'temperature': data['main']['temp'],
            'description': data['weather'][0]['description'] if 'weather' in data and len(data['weather']) > 0 else 'N/A',
            'icon': f"http://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png" if 'weather' in data and len(data['weather']) > 0 else 'N/A'
        }

        return jsonify(weather)
    else:
        return jsonify({'error': 'City not found or API request failed'}), 404

@app.route('/fetch_image')
def fetch_image():
    genre = request.args.get('genre', 'recipe')  # Default to 'recipe' if no genre is provided
    url = f"https://api.unsplash.com/photos/random?query={genre}&client_id={unsplash_api_key}&w=1920&h=1080"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        image_url = data['urls']['regular']
        return jsonify({'image_url': image_url})
    else:
        return jsonify({'error': 'Failed to fetch image'}), 500
    
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_message = request.json['message']
        chat_history.append({"role": "user", "message": user_message})
        context = "\n".join([f"{msg['role']}: {msg['message']}" for msg in chat_history])
        prompt = f"{context}\nbot:"
        response = model.generate_content(prompt)
        reply = response.text.strip()
        chat_history.append({"role": "bot", "message": reply})
        return jsonify({"reply": reply, "chat_history": chat_history})
    return render_template('chat.html')

@app.route('/chef', methods=['GET', 'POST'])
def chef():
    if request.method == 'POST':
        user_ingredients = request.form['user_ingredients']
        prompt = f"Generate a recipe based on the following ingredients {user_ingredients} and explain the steps to cook it in a stepwise manner and formatted manner ..also explain who can eat and who shouldn't eat."
        response = chef_model.generate_content([prompt])
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
            response = story_model.generate_content([prompt])
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
        response = psychology_model.generate_content([prompt])
        response_text = format_response(response.text)
        return jsonify({'response': response_text})
    return render_template('psychology_prediction.html')

@app.route('/code_generation', methods=['GET', 'POST'])
def code_generation():
    if request.method == 'POST':
        code_type = request.form['codeType']
        language = request.form['language']
        prompt = f"Write a {language} code to implement {code_type}."
        response = code_model.generate_content([prompt])
        
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
        response = algorithm_model.generate_content([prompt])
        
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
                try:
                    img = Image.open(BytesIO(image.read()))
                    prompt = [f"**In English**, analyze the image and user description for a person experiencing {symptoms}...", img]
                    response = model_vision.generate_content(prompt)
                except PIL.UnidentifiedImageError as e:
                    logging.error(f"Error processing image: {e}")
                    return jsonify({'error': "Image format not recognized"}), 400
                except Exception as e:
                    logging.error(f"Error processing image: {e}")
                    return jsonify({'error': "Image processing failed"}), 500
            else:
                try:
                    prompt = f"In English, Analyze symptoms for {gender} with symptoms: {symptoms}..."
                    response = model_text.generate_content([prompt])
                except Exception as e:
                    logging.error(f"Error generating text response: {e}")
                    return jsonify({'error': "Text generation failed"}), 500
            analysis_text = response.candidates[0].content.parts[0].text if response.candidates and response.candidates[0].content.parts else "No valid response found."
            analysis = analysis_text.split('\n')
            analysis_html = '<ul>'
            for line in analysis:
                analysis_html += f'<li>{line}</li>'
            analysis_html += '</ul>'
            return jsonify({'analysis': analysis_html, 'stage': stage})
        except Exception as e:
            logging.error(f"Error in /analyze route: {e}")
            return jsonify({'error': "Internal Server Error"}), 500
    return render_template('analyze.html')



if __name__ == '__main__':
    app.run(debug=True)
