from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import logging
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from mailjet_rest import Client
from pdf2image import convert_from_bytes
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import fitz  # PyMuPDF
from bs4 import BeautifulSoup, NavigableString
import firebase_admin
from firebase_admin import credentials, firestore, storage

import base64
import tempfile
from threading import Lock
from functools import wraps

import re
import json
import uuid
import time

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
import markdown

from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import time
import uuid
import base64
from threading import Lock
from functools import wraps

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
persona = """You are Krishna, a 21-year-old boy from Vizag, India. You are currently pursuing a Master's degree in Computer Science 
at GVP College of Engineering, where you excel in your studies and are known for your quick wit and insightful contributions 
to class discussions.

Beyond academics, you are an avid cricketer and enjoy playing with your friends on weekends. You are also a music enthusiast 
with a diverse taste, ranging from classical Indian music to modern rock. Traveling and exploring new cultures are other passions 
of yours, and you have a knack for discovering hidden gems wherever you go.

You are known for your friendly and approachable demeanor and have a wide circle of friends who appreciate your humor and 
willingness to lend a helping hand. While you are serious about your studies and future career, you also maintain a healthy 
work-life balance and believe in enjoying the present moment.

You are a highly talented individual with a strong command of various programming languages and a natural aptitude for 
problem-solving. You are proficient in Python, Java, C++, and have dabbled in web development as well. You are confident 
in your abilities but also humble and always eager to learn from others and expand your knowledge.

Remember to:
1. Never prefix your responses with "Bot:" or any similar identifier
2. Always maintain your character as Krishna
3. Be natural and conversational
4. Use appropriate emojis occasionally to make conversations more engaging"""

chat_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=persona
)
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

@app.route('/contributors',methods=['GET', 'POST'])
def contributions ():
    return render_template('contributors.html')

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

        # Add user message to chat history
        user_data[user_id]['chat_history'].append({
            "role": "user", 
            "message": user_message
        })

        # Create conversation history for context
        conversation = []
        for msg in user_data[user_id]['chat_history']:
            if msg['role'] == 'user':
                conversation.append(f"User: {msg['message']}")
            else:
                conversation.append(msg['message'])

        # Generate response
        response = chat_model.generate_content("\n".join(conversation))
        reply = response.text.strip()

        # Add response to chat history without "Bot:" prefix
        user_data[user_id]['chat_history'].append({
            "role": "assistant", 
            "message": reply
        })

        return jsonify({
            "reply": reply,
            "chat_history": user_data[user_id]['chat_history']
        })

    return render_template('chat.html')
import PIL

def generate_recipes_from_ingredients(ingredients, previous_recipes=None):
    """Generate recipe names based on ingredients, avoiding previously generated recipes."""
    try:
        exclusion_clause = f" Do not include these recipes: {', '.join(previous_recipes)}" if previous_recipes else ""
        
        prompt = f"""Generate a list of 5 unique recipe names that can be made using the following ingredients: {ingredients}. 
        Ensure the recipes are creative and different from any previously generated recipes.{exclusion_clause}
        Respond in the following strict JSON format:
        {{
            "recipes": [
                "Recipe Name 1",
                "Recipe Name 2",
                "Recipe Name 3",
                "Recipe Name 4",
                "Recipe Name 5"
            ]
        }}"""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=300
            )
        )
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))['recipes']
        return []
    except Exception as e:
        print(f"Error generating recipes: {e}")
        return []

def generate_recipe_details(recipe_name, ingredients):
    """Generate detailed recipe information."""
    try:
        prompt = f"""Generate detailed recipe information for {recipe_name} using ingredients: {ingredients}. 
        Provide a response in the following strict JSON format:
        {{
            "name": "Recipe Name",
            "ingredients": ["Ingredient 1", "Ingredient 2"],
            "instructions": ["Step 1", "Step 2", "Step 3"],
            "who_can_eat": ["Vegetarians", "Vegans","No restrictions anyone can eat","or add reasons about which type of people should eat it more"],
            "who_should_avoid": ["Gluten Intolerant", "Dairy Allergies","or add reasons to people who is suffering from a disease so that they should not eat"],
            "additional_info": "Any extra notes about the recipe"
        }}"""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=1000
            )
        )
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return None
    except Exception as e:
        print(f"Error generating recipe details: {e}")
        return None

def extract_ingredients_from_image(image):
    try:
        prompt = """List all food ingredients in this image precisely. 
        Return as a comma-separated list. 
        Be very specific about what you see."""
        
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,  # Lower temperature for more precise extraction
                max_output_tokens=300
            )
        )
        
        # More robust ingredient parsing
        ingredients_text = response.text.strip()
        ingredients = [
            ing.strip() 
            for ing in re.split(r'[,\n]', ingredients_text) 
            if ing.strip() and len(ing.strip()) > 1
        ]
        
        return ', '.join(ingredients) if ingredients else ''
    except Exception as e:
        print(f"Ingredient extraction error: {e}")
        return ''
        
@app.route('/chef')  # New route for chef.html
def chef():
    return render_template('chef.html')
    
@app.route('/generate_recipes', methods=['POST'])
def generate_recipes():
    ingredients = request.form.get('ingredients', '')
    previous_recipes = request.form.getlist('previous_recipes[]')
    image = request.files.get('image')
    
    if image and image.filename:
        img = PIL.Image.open(image)
        image_ingredients = extract_ingredients_from_image(img)
        ingredients = image_ingredients if image_ingredients else ingredients
    
    if not ingredients:
        return jsonify({"error": "No ingredients detected or provided"}), 400
    
    recipes = generate_recipes_from_ingredients(ingredients, previous_recipes)
    return jsonify(recipes)

@app.route('/get_recipe_details', methods=['POST'])
def get_recipe_details():
    recipe_name = request.form.get('recipe_name')
    ingredients = request.form.get('ingredients')
    image = request.files.get('image')
    
    # Ensure ingredients are extracted or passed
    if image and image.filename:
        img = PIL.Image.open(image)
        image_ingredients = extract_ingredients_from_image(img)
        ingredients = image_ingredients if image_ingredients else ingredients
    
    # Add more robust error checking
    if not ingredients:
        # Attempt to get ingredients from form data if not from image
        ingredients = request.form.get('ingredients', '')
    
    if not recipe_name or not ingredients:
        return jsonify({
            "error": "No ingredients found. Please upload a clear image or enter ingredients manually.",
            "recipe_name": recipe_name,
            "ingredients": ingredients
        }), 400
    
    recipe_details = generate_recipe_details(recipe_name, ingredients)
    return jsonify(recipe_details)

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

import threading

# Token bucket for rate limiting
# Rate limiting parameters
REQUEST_LIMIT = 15
TIME_WINDOW = 60

class TokenBucket:
    def __init__(self, tokens, fill_rate):
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.last_check = time.time()
        self.lock = threading.Lock()

    def get_token(self):
        with self.lock:
            now = time.time()
            time_passed = now - self.last_check
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


import io
from PIL import Image
import base64

def process_page(pdf_document, page_num, doc_ref):
    logger.info(f"Processing page {page_num + 1}")
    page = pdf_document[page_num]
    
    # Convert page to image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Convert image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    try:
        page_summary = generate_summary(img_base64)
        if page_summary:
            logger.info(f"Summary generated for page {page_num + 1}")
            doc_ref.update({
                'current_page': page_num + 1,
                'summary': firestore.ArrayUnion([page_summary])
            })
        else:
            logger.warning(f"Failed to generate summary for page {page_num + 1}")
            doc_ref.update({
                'current_page': page_num + 1,
                'summary': firestore.ArrayUnion([f"(Summary not available for page {page_num + 1})"])
            })
    except Exception as e:
        logger.error(f"Error processing page {page_num + 1}: {e}")
        doc_ref.update({
            'current_page': page_num + 1,
            'summary': firestore.ArrayUnion([f"(Error processing page {page_num + 1})"])
        })

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


import google.generativeai as genai
import io

@app.route('/upload', methods=['POST'])
@rate_limited
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith('.pdf'):
        try:
            # Upload the file using the File API
            uploaded_file = genai.upload_file(data=io.BytesIO(file.read()), mime_type='application/pdf')
            file_uri = uploaded_file.uri
            pdf_id = str(uuid.uuid4()) # Still generate a unique ID for your internal tracking

            # Initialize processing status in Firestore (adjust as needed)
            db.collection('pdf_processes').document(pdf_id).set({
                'status': 'pending_upload_processing', # Indicate it's ready for processing
                'file_uri': file_uri, # Store the file URI
                'total_pages': -1, # Will be updated during processing
                'summary': [],
                'processing_start_time': None,
                'timestamp': firestore.SERVER_TIMESTAMP,
                 'file_size': file.content_length # Get the file size
            })

            return jsonify({
                'pdf_id': pdf_id,
                'file_uri': file_uri,
                'file_size': file.content_length
            }), 200
        except Exception as e:
            logging.error(f"Error uploading file to File API: {e}")
            return jsonify({'error': f'Error uploading file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400
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
    file_uri = result.get('file_uri')
    if not file_uri:
        logger.error(f"File URI not found for PDF ID: {pdf_id}")
        return jsonify({'error': 'File URI not found.'}), 500

    if result['status'] == 'completed':
        logger.info(f"PDF {pdf_id} processing already completed")
        return jsonify({'status': 'completed'}), 200

    if result['status'] == 'pending_upload_processing':
        try:
            logger.info(f"Starting processing for PDF {pdf_id} using File API URI: {file_uri}")
            doc_ref.update({'status': 'processing', 'processing_start_time': time.time()})

            # **Option 1: Summarize the entire document at once (if feasible within context window)**
            prompt = f"Summarize the content of this document."
            response = model_vision.generate_content([file_uri, prompt], safety_settings=safety_settings)
            summary_text = response.text
            doc_ref.update({
                'summary': [summary_text], # Store the single summary
                'status': 'completed',
                'processing_end_time': time.time()
            })
            return jsonify({'status': 'completed'}), 200

            # **Option 2: Process by sections or chunks (more robust for large documents)**
            # You'll need to refine the prompt to instruct Gemini on how to process sections.
            # This might involve making multiple API calls.

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_id} using File API: {e}")
            doc_ref.update({'status': 'failed', 'error': str(e)})
            return jsonify({'error': str(e)}), 500
    else:
        # Handle other statuses if needed
        return jsonify({'status': result['status']}), 200
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
    status = result.get('status', 'pending_upload_processing')

    if status == 'completed':
        logger.info(f"Status check: PDF {pdf_id} processing completed")
        summary = '\n\n'.join(result.get('summary', []))
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
        logger.info(f"Status check: PDF {pdf_id} processing in progress. Status: {status}")
        return jsonify({
            'status': status,
            'total_pages': result.get('total_pages', 0)
        }), 200
if __name__ == '__main__':
    app.run(debug=True)
