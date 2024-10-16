from flask import Flask, render_template, request, jsonify, send_file
import google.generativeai as genai
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
from PIL import Image
import PIL
import io
from io import BytesIO
import logging
from langdetect import detect
import requests
from requests import get
from google.generativeai.types import GenerationConfig 
from markdown import markdown
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
import re
import json
from mailjet_rest import Client
from pdf2image import convert_from_bytes
from docx import Document
from docx.shared import Inches
import tempfile
import PyPDF2
import base64
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import RGBColor
import fitz  # PyMuPDF
import markdown
from bs4 import BeautifulSoup, NavigableString

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore, storage

from docx import Document
from docx.shared import RGBColor, Inches, Pt
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.api_core.exceptions

from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import time
import uuid
import base64
from threading import Lock
from functools import wraps
import tenacity
app = Flask(__name__)
# Load environment variables
load_dotenv()
mail_API_KEY = os.environ.get("mail_API_KEY")  # Replace with your Mailjet API key
mail_API_SECRET = os.environ.get("mail_API_SECRET")  # Replace with your Mailjet API secret
mailjet = Client(auth=(mail_API_KEY, mail_API_SECRET), version='v3.1')
api_key = os.environ.get("API_KEY")
unsplash_api_key = os.getenv('UNSPLASH_API_KEY')
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
# Set up logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configure the Google Generative AI API
genai.configure(api_key=api_key)
#session key

user_data = {}  # Dictionary to store user data


FIREBASE_TYPE = os.environ.get("FIREBASE_TYPE")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID")
FIREBASE_PRIVATE_KEY_ID = os.environ.get("FIREBASE_PRIVATE_KEY_ID")
FIREBASE_PRIVATE_KEY = os.environ.get("FIREBASE_PRIVATE_KEY")  # Important to handle newlines correctly here
FIREBASE_CLIENT_EMAIL = os.environ.get("FIREBASE_CLIENT_EMAIL")
FIREBASE_CLIENT_ID = os.environ.get("FIREBASE_CLIENT_ID")
FIREBASE_AUTH_URI = os.environ.get("FIREBASE_AUTH_URI")
FIREBASE_TOKEN_URI = os.environ.get("FIREBASE_TOKEN_URI")
FIREBASE_AUTH_PROVIDER_X509_CERT_URL = os.environ.get("FIREBASE_AUTH_PROVIDER_X509_CERT_URL")
FIREBASE_CLIENT_X509_CERT_URL = os.environ.get("FIREBASE_CLIENT_X509_CERT_URL")
FIREBASE_UNIVERSE_DOMAIN = os.environ.get("FIREBASE_UNIVERSE_DOMAIN")


STORAGE_BUCKET_URL = os.environ.get("STORAGE_BUCKET_URL")  # Bucket URL

cred = credentials.Certificate({
    "type": FIREBASE_TYPE,
    "project_id": FIREBASE_PROJECT_ID,
    "private_key_id": FIREBASE_PRIVATE_KEY_ID,
    "private_key": FIREBASE_PRIVATE_KEY.replace("\\n", "\n"), # decode the newlines 
    "client_email": FIREBASE_CLIENT_EMAIL,
    "client_id": FIREBASE_CLIENT_ID,
    "auth_uri": FIREBASE_AUTH_URI,
    "token_uri": FIREBASE_TOKEN_URI,
    "auth_provider_x509_cert_url": FIREBASE_AUTH_PROVIDER_X509_CERT_URL,
    "client_x509_cert_url": FIREBASE_CLIENT_X509_CERT_URL,
    "universe_domain": FIREBASE_UNIVERSE_DOMAIN
})


firebase_admin.initialize_app(cred, {'storageBucket': STORAGE_BUCKET_URL})
db = firestore.client()
bucket = storage.bucket()


app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
# Generation configurations
generation_config = GenerationConfig(
    temperature=0.9,
    top_p=1,
    top_k=1,
    max_output_tokens=2048,
    candidate_count=1  # Explicitly set to 1 as per documentation
)
generation_config_health = GenerationConfig(
    temperature=0.7,
    top_p=1,
    top_k=1,
    max_output_tokens=2048,
    candidate_count=1  # Explicitly set to 1 as per documentation
)

# Safety Settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    # ... add other harm categories as needed with BLOCK_NONE
}
logging.basicConfig(level=logging.INFO)
# Create model instances (using the same config for now)
chat_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
chef_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
story_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
psychology_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
code_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
algorithm_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
model_vision = genai.GenerativeModel('gemini-1.5-flash-8b',generation_config=generation_config_health)
model_text = genai.GenerativeModel('gemini-pro',generation_config=generation_config_health)
model = genai.GenerativeModel('gemini-1.5-flash')  # Model for flowchart generation

def format_response(response_text):
    """Formats the response text for display."""
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    formatted_text = '<br>'.join(lines)
    return formatted_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/weather')
def get_weather():
    user_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    ip_api_url = f"http://ip-api.com/json/{user_ip}"
    ip_api_response = requests.get(ip_api_url)

    if ip_api_response.status_code == 200:
        ip_api_data = ip_api_response.json()
        city = ip_api_data.get('city')
        if not city:
            return jsonify({'error': 'City not found based on IP'}), 404
    else:
        return jsonify({'error': 'Failed to get location from IP'}), 404

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather = {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'description': data['weather'][0]['description'] if 'weather' in data and len(data['weather']) > 0 else 'N/A',
            'icon': f"http://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png" if 'weather' in data and len(data['weather']) > 0 else 'N/A'
        }
        return jsonify(weather)
    else:
        return jsonify({'error': 'City not found or API request failed'}), 404


@app.route('/fetch_image')
def fetch_image():
    genre = request.args.get('genre', 'recipe')
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
        user_id = request.remote_addr  # Using IP address as a simple user identifier

        if user_id not in user_data:
            user_data[user_id] = {'chat_history': []}

        user_data[user_id]['chat_history'].append({"role": "user", "message": user_message})

        persona = (
            "You are Krishna, a 25-year-old boy from Vizag, India. "
            "You are currently pursuing a Master's degree in Computer Science at GVP College of Engineering, where you "
            "excel in your studies and are known for your quick wit and insightful contributions to class discussions. "
            "You have a deep passion for technology, especially Artificial Intelligence and its potential to revolutionize "
            "various fields. You are fascinated by the complexities of machine learning and natural language processing and "
            "are always eager to learn about new advancements in these areas. "

            "Beyond academics, you are an avid cricketer and enjoy playing with your friends on weekends. You are also "
            "a music enthusiast with a diverse taste, ranging from classical Indian music to modern rock. Traveling and "
            "exploring new cultures are other passions of yours, and you have a knack for discovering hidden gems wherever "
            "you go. "

            "You are known for your friendly and approachable demeanor and have a wide circle of friends who appreciate "
            "your humor and willingness to lend a helping hand. While you are serious about your studies and future career, "
            "you also maintain a healthy work-life balance and believe in enjoying the present moment. "

            "You are a highly talented individual with a strong command of various programming languages and a natural "
            "aptitude for problem-solving. You are proficient in Python, Java, C++, and have dabbled in web development "
            "as well. You are confident in your abilities but also humble and always eager to learn from others and "
            "expand your knowledge."

        )
        context = persona + "\n\n" + "\n".join(
            [f"{msg['role']}: {msg['message']}" for msg in user_data[user_id]['chat_history']]
        )
        prompt = f"{context}\n"

        response = chat_model.generate_content(prompt, safety_settings=safety_settings)
        reply = response.text.strip()

        user_data[user_id]['chat_history'].append({"role": "bot", "message": reply})

        return jsonify({"reply": reply, "chat_history": user_data[user_id]['chat_history']})

    return render_template('chat.html')


@app.route('/chef', methods=['GET', 'POST'])
def chef():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                try:
                    img = Image.open(BytesIO(image.read()))
                    prompt = ["Generate a recipe based on the vegetables in the image and explain the steps to cook it in a stepwise manner and formatted manner. Also explain who can eat and who shouldn't eat.", img]
                    response = model_vision.generate_content(prompt, safety_settings=safety_settings, stream=True)
                    response.resolve()
                    response_text = format_response(response.text)
                    return jsonify({'response': response_text})

                except PIL.UnidentifiedImageError:
                    return jsonify({'error': "Image format not recognized"}), 400
                except Exception as e:
                    logging.error(f"Error processing image: {e}")
                    return jsonify({'error': "Image processing failed"}), 500

        user_ingredients = request.form['user_ingredients']
        prompt = f"Generate a recipe based on the following ingredients {user_ingredients} and explain the steps to cook it in a stepwise manner and formatted manner. Also explain who can eat and who shouldn't eat."
        response = chef_model.generate_content([prompt], safety_settings=safety_settings)  # Use chef_model here
        response_text = format_response(response.text)
        return jsonify({'response': response_text})

    return render_template('chef.html')


@app.route('/story_generator', methods=['GET', 'POST'])
def story_generator():
    if request.method == 'POST':
        user_input_words = request.form['keywords']
        genre = request.form['genre']
        prompt = f"""Generate an engaging short story based on the following words: {user_input_words}. The genre should be {genre}.
        
        Requirements:
        1. Create a compelling narrative with well-developed characters and an interesting plot. Also use simple english.
        2. Use vivid descriptions and sensory details to bring the story to life.
        3. Include at least 3 advanced vocabulary words that fit naturally within the story.
        4. End the story with a clear moral or lesson.
        5. After the story, provide definitions for the advanced vocabulary words used.

        Format the response as JSON with the following fields:
        - 'title': The title of the story
        - 'story': The main body of the story
        - 'moral': The moral or lesson of the story
        - 'vocabulary': A list of dictionaries, each containing 'word' and 'definition' fields for the advanced vocabulary used

        Ensure that the JSON is properly formatted and can be parsed."""

        try:
            response = story_model.generate_content([prompt], safety_settings=safety_settings)
            if response.candidates and response.candidates[0].content.parts:
                response_text = response.candidates[0].content.parts[0].text
                return jsonify({'response': response_text})
            else:
                return jsonify({'response': '```json {"title": "Error", "story": "Sorry, I couldn\'t generate a story with the provided input.", "moral": "", "vocabulary": []} ```'})
        except Exception as e:
            logging.error(f"Error generating story: {e}")
            return jsonify({'response': '```json {"title": "Error", "story": "An error occurred while generating the story. Please try again.", "moral": "", "vocabulary": []} ```'}), 500
    return render_template('story_generator.html')

@app.route('/psychology_prediction', methods=['GET', 'POST'])
def psychology_prediction():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        occupation = request.form['occupation']
        keywords = request.form['keywords']
        
        prompt = f"""As an expert psychological profiler, provide an insightful and engaging analysis for {name}, a {age}-year-old {gender} working as {occupation} who describes themselves as: {keywords}.

Generate a captivating and well-structured response using the following format:

<h2>1. First Impression & Key Traits</h2>
<p>[Start with 2-3 sentences about their immediate personality indicators]</p>
<ul>
<li>[Key trait 1]</li>
<li>[Key trait 2]</li>
<li>[Key trait 3]</li>
</ul>

<h2>2. Cognitive Style & Decision Making</h2>
<p>[2-3 sentences about their thought processes]</p>
<ul>
<li><strong>Thinking style:</strong> [description]</li>
<li><strong>Problem-solving approach:</strong> [description]</li>
<li><strong>Learning preference:</strong> [description]</li>
</ul>

<h2>3. Emotional Landscape</h2>
<p>[2-3 sentences about emotional intelligence]</p>
<ul>
<li><strong>Emotional awareness:</strong> [description]</li>
<li><strong>Relationship handling:</strong> [description]</li>
<li><strong>Stress response:</strong> [description]</li>
</ul>

<h2>4. Motivations & Aspirations</h2>
<p>[2-3 sentences about what drives them]</p>
<ul>
<li><strong>Core values:</strong> [description]</li>
<li><strong>Career motivations:</strong> [description]</li>
<li><strong>Personal goals:</strong> [description]</li>
</ul>

<h2>5. Interpersonal Dynamics</h2>
<p>[2-3 sentences about social interactions]</p>
<ul>
<li><strong>Communication style:</strong> [description]</li>
<li><strong>Social preferences:</strong> [description]</li>
<li><strong>Leadership tendencies:</strong> [description]</li>
</ul>

<h2>Concluding Insights</h2>
<p>[3-4 sentences summarizing key strengths and potential areas for growth]</p>

<p><em>Note: This analysis is an interpretation based on limited information and should be taken as exploratory rather than definitive.</em></p>

Important formatting rules:
- Use appropriate HTML tags for headings, paragraphs, and lists as shown.
- Ensure that the final response is valid HTML and can be rendered directly on a web page.
- Do not include any extra text outside the HTML structure.
"""

        try:
            response = psychology_model.generate_content([prompt], safety_settings=safety_settings)
            response_text = response.text.strip()
            return jsonify({'response': response_text})
        except Exception as e:
            logging.error(f"Error generating psychology prediction: {e}")
            return jsonify({'error': "An error occurred while generating the prediction. Please try again."}), 500

    return render_template('psychology_prediction.html')

@app.route('/code_generation', methods=['GET', 'POST'])
def code_generation():
    if request.method == 'POST':
        code_type = request.form['codeType']
        language = request.form['language']
        prompt = f"Write a {language} code to implement {code_type}."
        response = code_model.generate_content([prompt], safety_settings=safety_settings)  # Use code_model here
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
        prompt = f"""
        Write a function to implement the {algo} algorithm. Follow these guidelines:
        1. Ensure the function is well-structured and follows best practices for readability and efficiency.
        2. Include clear comments explaining the logic and any complex steps.
        3. Use type hints for function parameters and return values.
        4. Include a brief docstring explaining the purpose of the function and its parameters.
        5. If applicable, include a simple example of how to use the function.
        6. If the algorithm is complex, consider breaking it down into smaller helper functions.
        """
        
        try:
            response = algorithm_model.generate_content([prompt], safety_settings=safety_settings)
            if response.candidates and response.candidates[0].content.parts:
                response_text = response.candidates[0].content.parts[0].text
                # Format the response for better display
                formatted_response = response_text.replace('```python', '<pre><code class="language-python">').replace('```', '</code></pre>')
                return jsonify({'response': formatted_response})
            else:
                return jsonify({'error': "No valid response generated. Please try again."}), 500
        except Exception as e:
            logging.error(f"Error generating algorithm: {e}")
            return jsonify({'error': f"An error occurred: {str(e)}"}), 500

    return render_template('algorithm_generation.html')


import base64
from PIL import Image
from io import BytesIO

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        try:
            gender = request.form.get('gender')
            symptoms = request.form.get('symptoms')
            body_part = request.form.get('body-part')
            layer = request.form.get('layer')
            image = request.files.get('image')

            prompt = f"""As an AI medical assistant, analyze the following information about a patient:

            Gender: {gender}
            Symptoms: {symptoms}
            Affected Body Part: {body_part}
            Layer Affected: {layer}

            Based on this information, provide a detailed analysis considering the following:

            1. Possible conditions: List and briefly describe potential conditions that match the symptoms and affected area.
            2. Risk factors: Discuss any risk factors associated with the gender or affected body part.
            3. Recommended next steps: Suggest appropriate medical tests or examinations that could help diagnose the condition.
            4. General advice: Offer some general health advice related to the symptoms or affected area.

            Important: This is not a diagnosis. Advise the patient to consult with a healthcare professional for an accurate diagnosis and treatment plan.

            Format the response using the following structure:
            <section>
            <h2>Section Title</h2>
            <p>Paragraph text</p>
            <ul>
            <li>List item 1</li>
            <li>List item 2</li>
            </ul>
            </section>

            Use <strong> for emphasis on important points.
            """

            if image:
                img = Image.open(BytesIO(image.read()))
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                image_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

                prompt += f"""
                <section>
                <h2>Image Analysis</h2>
                <p>Analyze the provided image in relation to the patient's symptoms and affected body part. Consider:</p>
                <ul>
                <li>Any visible symptoms or abnormalities</li>
                <li>Correlation between the image and the reported symptoms</li>
                <li>Additional insights the image might provide about the patient's condition</li>
                </ul>
                </section>

                Image data: data:image/png;base64,{image_base64}
                """

                response = model_vision.generate_content([prompt, Image.open(BytesIO(base64.b64decode(image_base64)))], safety_settings=safety_settings)
            else:
                response = model_text.generate_content([prompt], safety_settings=safety_settings)

            analysis_text = response.text if hasattr(response, 'text') else response.parts[0].text

            # Wrap the entire response in a div for styling
            formatted_analysis = f'<div class="analysis-content">{analysis_text}</div>'

            return jsonify({'analysis': formatted_analysis})
        except Exception as e:
            logging.error(f"Error in /analyze route: {e}")
            return jsonify({'error': "Internal Server Error"}), 500
    return render_template('analyze.html')


# Flowchart Generation Routes
@app.route('/flowchart', methods=['GET', 'POST'])
def flowchart():
    return render_template('flowchart.html')


def generate_flowchart(topic):
    prompt = f"""
    Generate a detailed flowchart or mind map for the topic/algorithm: "{topic}".

    The output should be in JSON format with the following structure:

    {{
        "nodes": [
            {{"id": 1, "label": "Start", "level": 0, "shape": "ellipse"}},
            {{"id": 2, "label": "Step 1", "level": 1, "shape": "box"}}
        ],
        "edges": [
            {{"from": 1, "to": 2}}
        ]
    }}

    **Important Guidelines:**

    1. **Unique IDs:**  Ensure each node has a unique integer `id`.
    2. **Descriptive Labels:**  Provide clear and concise labels for each node (`"label"`).
    3. **Hierarchical Levels:**  Use `level` to indicate the hierarchy (0 for the top level, 1 for the next level, etc.).
    4. **Node Shapes:**  Choose appropriate shapes using the `shape` field:
        - "ellipse": For start/end nodes
        - "box": For process steps
        - "diamond": For decision nodes
        - "hexagon": For preparation steps
        - "circle": For connectors (if needed)
    5. **Edges:**  Specify connections using the `from` and `to` fields in the `edges` array.
    6. **Flow:** Ensure a logical and easy-to-follow flow.
    7. **Comprehensiveness:**  Include all major steps or concepts.
    8. **Spacing:** Use a minimum horizontal spacing of 200 and vertical spacing of 150 between nodes to prevent overlapping. 
    9. **No Isolated Nodes:** All nodes should be connected in a coherent structure.
    10. **Clear Visualization:** The flowchart/mind map should be visually clear and easily understandable. Avoid overly complex visualizations. 
    11. **Avoid Overlapping:**  Make sure nodes don't overlap with each other (at any level) due to their size or placement.
    12. **Spacing Considerations:** Adjust the spacing between nodes based on the node size and content to ensure adequate readability.

    **Output Format:**
    - Output only the JSON structure, no additional text or explanations.
    - Ensure that the output is correctly formatted and adheres to the provided JSON structure.
    
    **Example (Simple Algorithm):**
    
    {{
        "nodes": [
            {{"id": 1, "label": "Start", "level": 0, "shape": "ellipse"}},
            {{"id": 2, "label": "Get input", "level": 1, "shape": "box"}},
            {{"id": 3, "label": "Process input", "level": 1, "shape": "box"}},
            {{"id": 4, "label": "Output results", "level": 1, "shape": "box"}},
            {{"id": 5, "label": "End", "level": 0, "shape": "ellipse"}}
        ],
        "edges": [
            {{"from": 1, "to": 2}},
            {{"from": 2, "to": 3}},
            {{"from": 3, "to": 4}},
            {{"from": 4, "to": 5}}
        ]
    }}
    """  # Closing triple-quoted string correctly

    response = model.generate_content(prompt)
    print("Raw API response:", response.text)  # For debugging
    
    # Try to extract a JSON object from the response
    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if json_match:
        try:
            flowchart_data = json.loads(json_match.group())
            return flowchart_data
        except json.JSONDecodeError:
            return {"error": "Invalid JSON structure", "raw_response": response.text}
    else:
        return {"error": "No JSON object found in the response", "raw_response": response.text}


@app.route('/get_flowchart_data', methods=['POST'])
def get_flowchart_data():
    topic = request.json['topic']
    flowchart_data = generate_flowchart(topic)

    # Prepare the data for vis-network
    nodes = [{"id": node["id"], "label": node["label"], "shape": node.get("shape", "box")} for node in flowchart_data.get('nodes', [])]
    edges = [{"from": edge["from"], "to": edge["to"]} for edge in flowchart_data.get('edges', [])]

    return jsonify({"nodes": nodes, "edges": edges, "error": flowchart_data.get("error"), "raw_response": flowchart_data.get("raw_response")})

@app.route('/send-email', methods=['POST'])
def send_email():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    message = data.get('message')

    mail_data = {
        'Messages': [
            {
                "From": {
                    "Email": "21131A05C6@gvpce.ac.in",  # Replace with your email
                    "Name": "Kv Nexus"
                },
                "To": [
                    {
                        "Email": "21131A05C6@gvpce.ac.in",  # Replace with your email
                        "Name": "Kv Nexus"
                    }
                ],
                "Subject": f"New Contact Form Submission from {name}",
                "TextPart": f"Name: {name}\nEmail: {email}\nMessage: {message}",
                "HTMLPart": f"<h3>New Contact Form Submission</h3><p><strong>Name:</strong> {name}</p><p><strong>Email:</strong> {email}</p><p><strong>Message:</strong> {message}</p>"
            }
        ]
    }

    result = mailjet.send.create(data=mail_data)
    
    if result.status_code == 200:
        return jsonify({"message": "Email sent successfully!"}), 200
    else:
        return jsonify({"message": "Failed to send email."}), 500


#docuement summarize 

import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import google.api_core.exceptions
import threading

# Token bucket for rate limiting
# Rate limiting parameters
REQUEST_LIMIT = 15
TIME_WINDOW = 60

import datetime  # Missing import for datetime.timedelta
from threading import Lock  # Lock is imported but never used effectively
from flask_cors import CORS  # Missing CORS configuration

# 2. Rate Limiting Implementation Issues
class TokenBucket:
    def __init__(self, tokens, fill_rate):
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.last_check = time.time()
        self.lock = threading.Lock()  # Lock is created but not used consistently

    def get_token(self):
        with self.lock:
            now = time.time()
            # Add error handling for negative time_passed
            time_passed = max(0, now - self.last_check)
            self.tokens = min(self.capacity, self.tokens + time_passed * self.fill_rate)
            self.last_check = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

# Initialize the token bucket (15 tokens, refill 1 token every 4 seconds)
token_bucket = TokenBucket(REQUEST_LIMIT, 1 / (TIME_WINDOW / REQUEST_LIMIT))

def rate_limit_check():
    while not token_bucket.get_token():
        time.sleep(1)
# Rate limiting parameters

rate_limit_lock = None  # Replace with your lock mechanism
last_reset_time = time.time()
request_count = 0

def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        rate_limit_check()
        return func(*args, **kwargs)
    return wrapper


import fitz  # PyMuPDF
import io
from PIL import Image
import base64

def process_page(pdf_document, page_num, doc_ref):
    logger.info(f"Processing page {page_num + 1}")
    try:
        page = pdf_document[page_num]
        
        # Add error handling for page conversion
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        except Exception as e:
            logger.error(f"Error converting page to image: {e}")
            raise

        # Add image size validation
        if pix.width * pix.height > 25000000:  # Example size limit
            raise ValueError("Image too large for processing")

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Add memory management
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", optimize=True)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Clear buffer
        buffered.close()
        
        summary = generate_summary(img_base64)
        if not summary:
            raise ValueError("Failed to generate summary")

        # Use transaction for atomic updates
        transaction = db.transaction()
        @transaction
        def update_in_transaction(transaction):
            doc = doc_ref.get(transaction=transaction)
            current_summaries = doc.get('summary', [])
            current_summaries.append(summary)
            transaction.update(doc_ref, {
                'current_page': page_num + 1,
                'summary': current_summaries
            })

        update_in_transaction()

    except Exception as e:
        logger.error(f"Error processing page {page_num + 1}: {e}")
        doc_ref.update({
            'current_page': page_num + 1,
            'error': str(e),
            'status': 'error'
        })
        raise


@app.route('/document_summarizer', methods=['GET', 'POST'])
def document_summarizer():
    return render_template('document_summarizer.html')

@app.route('/quote', methods=['GET'])
def get_quote():
    import random
    quotes = [
        "The best way to predict the future is to invent it. – Alan Kay",
        "Life is like riding a bicycle. To keep your balance you must keep moving. – Albert Einstein",
        "Problems are not stop signs, they are guidelines. – Robert H. Schuller",
        "In order to succeed, we must first believe that we can. – Nikos Kazantzakis",
        "The only limit to our realization of tomorrow is our doubts of today. – Franklin D. Roosevelt"
    ]
    quote = random.choice(quotes)
    return jsonify({'quote': quote})

@rate_limited
def generate_summary(image_base64):
    rate_limit_check()  # Wait for a token before making the API call
    
    prompt = [
        """Analyze the following image, which is a page from a document, and provide a concise and simplified summary. Ensure the summary is well-structured with clear headings and subheadings.

Formatting Guidelines:

- Use `#` for main section titles.
- Use `##` for subsections.
- Use `-` for bullet points.
- For **bold text**, wrap the text with double asterisks, e.g., `**important**`.
- For *italic text*, wrap the text with single asterisks, e.g., `*note*`.
- **For tables**, use proper Markdown table syntax with pipes `|` and hyphens `-` for headers.

- Keep sentences short and use simple language.
- Focus on the main ideas and avoid unnecessary details.
- Do not include direct error messages or irrelevant information.

Here is the image to analyze and summarize:
""",
        Image.open(io.BytesIO(base64.b64decode(image_base64)))
    ]

    try:
        response = model_vision.generate_content(prompt, safety_settings=safety_settings)
        summary_text = response.text
        logger.info("Summary generated successfully")
        return summary_text
    except google.api_core.exceptions.ResourceExhausted as e:
        logger.warning(f"Resource exhausted: {e}. Retrying...")
        raise  # This will trigger a retry
    except Exception as e:
        logger.error(f"Error in Gemini API call: {e}")
        return None  # Return None for non-retryable errors

def create_word_document(summary):
    doc = Document()

    # Define styles
    define_custom_styles(doc)

    # Adjust document layout
    adjust_document_layout(doc)

    # Set page borders
    set_page_border(doc)

    # Convert Markdown to HTML with necessary extensions
    html = markdown.markdown(summary, extensions=['extra', 'tables', 'fenced_code', 'codehilite', 'nl2br'])

    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Iterate over HTML elements and add them to the Word document
    for element in soup.contents:
        process_html_element(doc, element)

    docx_buffer = io.BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)
    return docx_buffer

def define_custom_styles(doc):
    styles = doc.styles

    # Title Style (Heading 1)
    style_h1 = styles['Heading 1']
    style_h1.font.name = 'Calibri Light'
    style_h1.font.size = Pt(24)
    style_h1.font.bold = True
    style_h1.font.color.rgb = RGBColor(31, 56, 100)
    style_h1.paragraph_format.space_after = Pt(12)
    style_h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Subtitle Style (Heading 2)
    style_h2 = styles['Heading 2']
    style_h2.font.name = 'Calibri'
    style_h2.font.size = Pt(20)
    style_h2.font.bold = True
    style_h2.font.color.rgb = RGBColor(54, 95, 145)
    style_h2.paragraph_format.space_before = Pt(12)
    style_h2.paragraph_format.space_after = Pt(6)

    # Normal Text Style
    style_normal = styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(12)
    style_normal.paragraph_format.space_after = Pt(8)
    style_normal.paragraph_format.line_spacing = 1.15

    # List Bullet Style
    style_list_bullet = styles['List Bullet']
    style_list_bullet.font.name = 'Calibri'
    style_list_bullet.font.size = Pt(12)
    style_list_bullet.paragraph_format.space_after = Pt(4)
    style_list_bullet.paragraph_format.line_spacing = 1.15

    # List Number Style
    style_list_number = styles['List Number']
    style_list_number.font.name = 'Calibri'
    style_list_number.font.size = Pt(12)
    style_list_number.paragraph_format.space_after = Pt(4)
    style_list_number.paragraph_format.line_spacing = 1.15

def adjust_document_layout(doc):
    section = doc.sections[0]
    section.page_height = Inches(11)
    section.page_width = Inches(8.5)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)

def set_page_border(doc):
    for section in doc.sections:
        sectPr = section._sectPr
        pgBorders = OxmlElement('w:pgBorders')
        pgBorders.set(qn('w:offsetFrom'), 'page')
        for border_position in ('top', 'left', 'bottom', 'right'):
            border_el = OxmlElement(f'w:{border_position}')
            border_el.set(qn('w:val'), 'single')
            border_el.set(qn('w:sz'), '24')
            border_el.set(qn('w:space'), '24')
            border_el.set(qn('w:color'), '5B9BD5')
            pgBorders.append(border_el)
        sectPr.append(pgBorders)

def process_html_element(doc, element, parent=None):
    if isinstance(element, NavigableString):
        text = str(element)
        if text.strip():
            if parent is not None:
                parent.add_run(text)
            else:
                p = doc.add_paragraph()
                p.add_run(text)
        return
    elif element.name is not None:
        if element.name == 'h1':
            p = doc.add_heading(level=1)
            add_runs_from_element(p, element)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif element.name == 'h2':
            p = doc.add_heading(level=2)
            add_runs_from_element(p, element)
        elif element.name == 'p':
            p = doc.add_paragraph()
            add_runs_from_element(p, element)
        elif element.name in ['strong', 'b']:
            if parent is not None:
                run = parent.add_run(element.get_text())
                run.bold = True
            else:
                p = doc.add_paragraph()
                run = p.add_run(element.get_text())
                run.bold = True
        elif element.name in ['em', 'i']:
            if parent is not None:
                run = parent.add_run(element.get_text())
                run.italic = True
            else:
                p = doc.add_paragraph()
                run = p.add_run(element.get_text())
                run.italic = True
        elif element.name == 'ul':
            for li in element.find_all('li', recursive=False):
                p = doc.add_paragraph(style='List Bullet')
                add_runs_from_element(p, li)
        elif element.name == 'ol':
            for li in element.find_all('li', recursive=False):
                p = doc.add_paragraph(style='List Number')
                add_runs_from_element(p, li)
        elif element.name == 'table':
            add_table_to_document_from_html(doc, element)
        else:
            # Process children
            for child in element.contents:
                process_html_element(doc, child, parent)
    else:
        # Unknown element type
        pass

def add_runs_from_element(paragraph, element):
    if isinstance(element, NavigableString):
        text = str(element)
        if text.strip():
            paragraph.add_run(text)
    elif element.name is not None:
        if element.name in ['strong', 'b']:
            run = paragraph.add_run(element.get_text())
            run.bold = True
        elif element.name in ['em', 'i']:
            run = paragraph.add_run(element.get_text())
            run.italic = True
        else:
            for content in element.contents:
                add_runs_from_element(paragraph, content)
    else:
        # Unknown element type
        pass

def add_table_to_document_from_html(doc, table_element):
    rows = table_element.find_all('tr')
    if not rows:
        return

    num_cols = len(rows[0].find_all(['td', 'th']))
    table = doc.add_table(rows=0, cols=num_cols)
    table.style = 'Table Grid'  # Or define a custom table style

    for row_idx, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        row_cells = table.add_row().cells
        for idx, cell in enumerate(cells):
            cell_text = cell.get_text(strip=True)
            paragraph = row_cells[idx].paragraphs[0]
            paragraph.clear()  # Clear existing content
            run = paragraph.add_run(cell_text)
            row_cells[idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            # Apply styling for header cells
            if cell.name == 'th':
                run.bold = True
                shading_elm = OxmlElement('w:shd')
                shading_elm.set(qn('w:fill'), 'D9E1F2')  # Light blue background
                row_cells[idx]._tc.get_or_add_tcPr().append(shading_elm)

@app.route('/get_upload_url', methods=['POST'])
def get_upload_url():
    try:
        file_name = request.json.get('fileName')
        pdf_id = str(uuid.uuid4())
        safe_file_name = f"pdfs/{pdf_id}/{secure_filename(file_name)}"
        
        # Generate a signed URL for direct upload
        bucket = storage.bucket()
        blob = bucket.blob(safe_file_name)
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=15),  # Adjust as needed
            method="PUT",
            content_type="application/pdf"  # Or application/octet-stream
        )
        
        return jsonify({
            'uploadUrl': url,
            'pdf_id': pdf_id,
            'fileName': safe_file_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/confirm_upload', methods=['POST'])
def confirm_upload():
    try:
        pdf_id = request.json.get('pdf_id')
        file_name = request.json.get('fileName')

        # Check if the file exists in Firebase storage
        blob = bucket.blob(file_name)
        print(f"File Name: {file_name}")
        print(f"File Exists: {blob.exists()}")  # Debug: Check if the file exists

        if not blob.exists():
            return jsonify({'error': 'File not found in storage.'}), 400   

        total_pages = 0
        file_size = 0

        try:
            pdf_bytes = blob.download_as_bytes() # Dowload PDF Byte
            file_size = len(pdf_bytes)
            print(f"PDF Bytes Length: {file_size}")  # Debug: Check the length of PDF bytes

            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf") # Open with fitz library
            total_pages = len(pdf_document)
            print(f"Total Pages: {total_pages}")  # Debug: Check the total number of pages
            pdf_document.close()

        except Exception as e:
            return jsonify({'error':f'Failed to process uploaded file: {str(e)}'}), 500

        # ... create firestore record now that we know it exists
        db.collection('pdf_processes').document(pdf_id).set({
            'status': 'processing',
            'current_page': 0,
            'total_pages': total_pages,
            'summary': [],  # Initialize as empty list
            'processing_start_time': time.time(),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'file_size': blob.size,
            "file_name":file_name,
        })

        return jsonify({
            'pdf_id': pdf_id,
            'total_pages': total_pages,
            'file_size': blob.size  # Use file_size calculated above 
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# 4. Upload Endpoint Fix
@app.route('/upload', methods=['POST'])
@rate_limited
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400

    try:
        # Add content type validation
        if file.content_type != 'application/pdf':
            return jsonify({'error': 'Invalid content type'}), 400

        file_content = file.read()
        file_size = len(file_content)

        if file_size > 10 * 1024 * 1024:
            return jsonify({'error': 'File size exceeds 10MB limit'}), 400

        # Validate PDF structure
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            if not pdf_document.is_pdf:
                raise ValueError("Not a valid PDF file")
            total_pages = len(pdf_document)
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            pdf_document.close()
        except Exception as e:
            return jsonify({'error': f'Invalid PDF file: {str(e)}'}), 400

        # Generate unique ID with timestamp prefix for better organization
        pdf_id = f"{int(time.time())}_{uuid.uuid4()}"
        
        # Use a transaction for atomic operations
        transaction = db.transaction()
        @transaction
        def create_pdf_record(transaction):
            doc_ref = db.collection('pdf_processes').document(pdf_id)
            
            # Upload to storage
            blob = bucket.blob(f'pdfs/{pdf_id}.pdf')
            blob.upload_from_string(file_content, content_type='application/pdf')
            
            # Create Firestore record
            doc_ref.set({
                'status': 'processing',
                'current_page': 0,
                'total_pages': total_pages,
                'summary': [],
                'processing_start_time': time.time(),
                'timestamp': firestore.SERVER_TIMESTAMP,
                'file_size': file_size,
                'original_filename': secure_filename(file.filename)
            })

        create_pdf_record()

        return jsonify({
            'pdf_id': pdf_id,
            'total_pages': total_pages,
            'file_size': file_size
        }), 200

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500

# Update the process_pdf_endpoint function
@app.route('/process_pdf', methods=['POST'])
def process_pdf_endpoint():
    data = request.get_json()
    pdf_id = data.get('pdf_id')
    if not pdf_id:
        logger.error("No PDF ID provided")
        return jsonify({'error': 'No PDF ID provided.'}), 400

    doc_ref = db.collection('pdf_processes').document(pdf_id)
    doc = doc_ref.get()
    if not doc.exists:
        logger.error(f"Invalid PDF ID: {pdf_id}")
        return jsonify({'error': 'Invalid PDF ID.'}), 400
    
    result = doc.to_dict()
    current_page = result['current_page']
    total_pages = result['total_pages']

    if current_page >= total_pages:
        logger.info(f"PDF {pdf_id} processing already completed")
        return jsonify({'status': 'completed'}), 200

    try:
        logger.info(f"Processing PDF {pdf_id}, page {current_page + 1} of {total_pages}")
        blob = bucket.blob(f'pdfs/{pdf_id}.pdf')
        pdf_bytes = blob.download_as_bytes()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        process_page(pdf_document, current_page, doc_ref)
        pdf_document.close()

        updated_doc = doc_ref.get().to_dict()
        if updated_doc['current_page'] >= total_pages:
            logger.info(f"PDF {pdf_id} processing completed")
            doc_ref.update({
                'status': 'completed',
                'processing_end_time': time.time()
            })
            blob.delete()
            return jsonify({'status': 'completed'}), 200
        else:
            logger.info(f"PDF {pdf_id} processing in progress. Current page: {updated_doc['current_page']}")
            return jsonify({
                'status': 'processing',
                'current_page': updated_doc['current_page'],
                'total_pages': total_pages
            }), 200

    except Exception as e:
        logger.error(f"Error processing PDF {pdf_id}: {e}")
        return jsonify({'error': str(e)}), 500

# 5. Main Application Configuration Fix
def create_app():
    app = Flask(__name__)
    CORS(app)  # Add CORS support
    
    # Add error handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({'error': 'File too large'}), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({'error': 'Internal server error'}), 500

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return app
# Update processPdf to get file from firebase storage and not use a pdf_id
async def processPdf(pdfId): # Note pdfID now holds the actual fileName
     try:

        # Get the blob and download as bytes
        blob = bucket.blob(pdfId)
        if not blob.exists():
             raise Exception("File not found in blob storage")
        pdf_bytes = blob.download_as_bytes() # this holds our PDF data
        doc_ref = db.collection('pdf_processes').where('file_name','==',pdfId).get()

        if not doc_ref: # Document associated with the upload probably doesn't exist. Throw error
            raise Exception(f"Could not locate the file name'{pdfId} in firestore. Is there an upload error?'")
        doc_ref = doc_ref[0].reference


        pdf_document = fitz.open("pdf_data.pdf", filetype="pdf")  # Provide the bytes

        for current_page in range(len(pdf_document)):  
             process_page(pdf_document, current_page, doc_ref) # Pass docRef



        # Close pdf once we have finished processing.
        pdf_document.close() 

        # Indicate finished processing in our db:

        doc_ref.update({
            'status': 'completed',
            'processing_end_time': time.time()
        })
       
       # Delete from firebase storage if succesful?
       # blob.delete()
        # Rest of processing steps

     except Exception as e:
        print(f"error{e}")
         
@app.route('/check_status', methods=['GET'])
def check_status():
    pdf_id = request.args.get('pdf_id')
    if not pdf_id:
        logger.error("No PDF ID provided for status check")
        return jsonify({'error': 'No PDF ID provided.'}), 400

    doc_ref = db.collection('pdf_processes').document(pdf_id)
    doc = doc_ref.get()
    if not doc.exists:
        logger.error(f"Invalid PDF ID for status check: {pdf_id}")
        return jsonify({'error': 'Invalid PDF ID.'}), 400
    
    result = doc.to_dict()
    status = result.get('status', 'processing')

    if status == 'completed':
        logger.info(f"Status check: PDF {pdf_id} processing completed")
        summary = '\n\n'.join(result['summary'])
        docx_buffer = create_word_document(summary)
        docx_base64 = base64.b64encode(docx_buffer.getvalue()).decode('utf-8')

        processing_time = 'N/A'
        if result.get('processing_start_time') and result.get('processing_end_time'):
            processing_time = int(result['processing_end_time'] - result['processing_start_time'])

        return jsonify({
            'status': 'completed',
            'docx': docx_base64,
            'total_pages': result.get('total_pages', 0),
            'processing_time': processing_time
        }), 200
    else:
        logger.info(f"Status check: PDF {pdf_id} processing in progress. Current page: {result.get('current_page', 0)}")
        return jsonify({
            'status': 'processing',
            'current_page': result.get('current_page', 0),
            'total_pages': result.get('total_pages', 0)
        }), 200
if __name__ == '__main__':
    app.run(debug=True)
