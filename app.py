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
logging.basicConfig(level=logging.INFO)
# Configure the Google Generative AI API
genai.configure(api_key=api_key)
#session key

user_data = {}  # Dictionary to store user data

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
    max_output_tokens=512,
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

# Create model instances (using the same config for now)
chat_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
chef_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
story_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
psychology_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
code_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
algorithm_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
model_vision = genai.GenerativeModel('gemini-1.5-flash',generation_config=generation_config_health)
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
        prompt = f"""Analyze the likely psychological traits of {name}, 
            a {age}-year-old {gender} {occupation}, who describes 
            themself as {keywords}. Based on this limited information, 
            consider their potential personality characteristics, 
            motivations, and social tendencies. Keep in mind that this 
            analysis is speculative and may not be a complete or 
            accurate representation of {name}'s psychology."""
        response = psychology_model.generate_content([prompt], safety_settings=safety_settings)  # Use psychology_model here
        response_text = format_response(response.text)
        return jsonify({'response': response_text})
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
        prompt = f"Write a Python function to implement the {algo} algorithm. Ensure the function is well-structured and follows best practices for readability and efficiency."
        response = algorithm_model.generate_content([prompt], safety_settings=safety_settings)  # Use algorithm_model here
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

            if image:
                try:
                    img = Image.open(BytesIO(image.read()))
                    prompt = [
                        f"""Acting as a doctor, but without providing any 
                        diagnosis or medical advice, what are some possible 
                        explanations for a {gender} patient experiencing 
                        {symptoms} on their {body_part}? Consider the provided 
                        image, but remember this is a speculative exercise 
                        and should not be taken as actual medical guidance. 
                        Use Simple and easy english""",
                        img
                    ]
                    response = model_vision.generate_content(prompt, safety_settings=safety_settings)

                except Exception as e:
                    logging.error(f"Error processing image: {e}")
                    return jsonify({'error': "Image processing failed"}), 500
            else:
                try:
                    prompt = f"""If a {gender} patient presented with {symptoms} 
                                on their {body_part}, what are some potential 
                                conditions that a doctor might consider? Please 
                                note that this is a hypothetical exercise and 
                                should not be interpreted as a real diagnosis 
                                or medical advice. Use Simple and easy english"""
                    response = model_text.generate_content([prompt], safety_settings=safety_settings)
                except Exception as e:
                    logging.error(f"Error generating text response: {e}")
                    return jsonify({'error': "Text generation failed"}), 500

            analysis_text = response.candidates[0].content.parts[0].text if response.candidates and response.candidates[0].content.parts else "No valid response found."

            # --- Format the response (final version) ---
            formatted_analysis = analysis_text.replace("**", "<b>")  # Replace start of heading
            formatted_analysis = formatted_analysis.replace("**", "</b><br>")  # Replace end of heading and add a line break
            formatted_analysis = formatted_analysis.replace("* ", "\n* ")
            formatted_analysis = formatted_analysis.replace("\n\n", "\n")
            formatted_analysis = formatted_analysis.replace("* \n", "* ")
            return jsonify({'analysis': formatted_analysis})
        except Exception as e:
            logging.error(f"Error in /analyze route: {e}")
            return jsonify({'error': "Internal Server Error"}), 500
    return render_template('analyze.html')  # Make sure you have analyze.html



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

# Rate limiting setup
import tempfile
import time
from threading import Lock
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
REQUEST_LIMIT = 15  # 15 requests per minute
TIME_WINDOW = 60  # 60 seconds (1 minute)
request_count = 0
last_request_time = 0
rate_limit_lock = Lock()

def process_pdf_in_batches(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    total_pages = len(pdf_document)
    all_summaries = []
    batch_size = 15
    batches = []
    
    # Prepare batches of pages
    for i in range(0, total_pages, batch_size):
        batch = list(range(i, min(i + batch_size, total_pages)))
        batches.append(batch)
    
    for batch_index, batch in enumerate(batches):
        batch_texts = []
        batch_images = []
        
        for page_num in batch:
            page = pdf_document[page_num]
            
            # Extract text
            text = page.get_text()
            batch_texts.append(f"Page {page_num + 1}: {text}")
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                batch_images.append((page_num + 1, image_base64))
        
        # Generate summary for this batch without batch numbering
        batch_summary = generate_summary(batch_texts, [img[1] for img in batch_images])
        all_summaries.append(batch_summary)
        
        # If not the last batch, wait to avoid rate limiting
        if batch_index < len(batches) - 1:
            time.sleep(60)
    
    # Combine all summaries without batch numbering
    final_summary = "\n\n".join(all_summaries)
    return final_summary, total_pages

def rate_limited(func):
    def wrapper(*args, **kwargs):
        global request_count, last_request_time
        with rate_limit_lock:
            current_time = time.time()
            if current_time - last_request_time > TIME_WINDOW:
                request_count = 0
                last_request_time = current_time
            
            request_count += 1
            if request_count > REQUEST_LIMIT:
                remaining_time = TIME_WINDOW - (current_time - last_request_time)
                return jsonify({'error': f'Rate limit exceeded. Please try again in {int(remaining_time)} seconds.'}), 429

            last_request_time = current_time  # Reset for each request if less than per minute request
        return func(*args, **kwargs)

    return wrapper


@app.route('/document_summarizer', methods=['GET', 'POST'])
def document_summarizer():
    return render_template('document_summarizer.html')


def process_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    images = []
    texts = []

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Extract text
        text = page.get_text()
        texts.append(text)
        
        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            images.append(image_base64)

    return texts, images

def generate_summary(texts, images):
    prompt = [
        "Summarize the following text and describe any images present itself in that page. "
        "Format the summary with clear section titles and subtitles. "
        "Use 'TITLE:' for main sections and 'SUBTITLE:' for subsections. "
        "Do not use any special characters or symbols for formatting. "
        "Separate paragraphs with a blank line. "
        "If there are any tables in the text, format them using Markdown syntax.",
        "Simplify using simple and easy english.",
        "Dont Make it more lengthy.",
        "\n".join(texts)
    ]
    
    # Add images to prompt if available
    for image in images:
        img = Image.open(io.BytesIO(base64.b64decode(image)))
        prompt.append(img)
    
    response = model.generate_content(prompt)
    return response.text

def create_word_document(summary):
    doc = Document()
    
    # Define styles
    styles = doc.styles
    title_style = styles.add_style('CustomTitle', WD_STYLE_TYPE.PARAGRAPH)
    title_style.font.size = Pt(18)
    title_style.font.bold = True
    title_style.font.color.rgb = RGBColor(0, 0, 128)
    title_style.paragraph_format.space_after = Pt(12)  # Add some space after titles
    
    subtitle_style = styles.add_style('CustomSubtitle', WD_STYLE_TYPE.PARAGRAPH)
    subtitle_style.font.size = Pt(14)
    subtitle_style.font.bold = True
    subtitle_style.font.color.rgb = RGBColor(0, 128, 0)
    subtitle_style.paragraph_format.space_before = Pt(12)  # Add space before subtitles
    subtitle_style.paragraph_format.space_after = Pt(6)   # Add space after subtitles
    
    normal_style = styles.add_style('CustomNormal', WD_STYLE_TYPE.PARAGRAPH)
    normal_style.font.size = Pt(11)
    normal_style.paragraph_format.space_after = Pt(6)  # Add some space between paragraphs
    
    # Clean up the summary text to remove double asterisks
    summary = re.sub(r'\*\*([^*]+)\*\*', r'\1', summary)
    
    # Split the summary into sections based on TITLE markers
    sections = re.split(r'\n\s*TITLE:', summary)
    
    for i, section in enumerate(sections):
        if section.strip():
            if i > 0:  # Add "TITLE:" back for all sections except the first one
                section = "TITLE:" + section
            
            # Add extra space before new sections (except the first one)
            if i > 0:
                doc.add_paragraph().style = doc.styles['Normal']
            
            paragraphs = section.split('\n')
            for para in paragraphs:
                para = para.strip()
                if para.startswith('TITLE:'):
                    p = doc.add_paragraph(para[6:].strip(), style='CustomTitle')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif para.startswith('SUBTITLE:'):
                    p = doc.add_paragraph(para[9:].strip(), style='CustomSubtitle')
                elif para.startswith('|'):  # Table detection
                    table = doc.add_table(rows=1, cols=len(para.split('|')) - 2)
                    table.style = 'Table Grid'
                    cells = para.split('|')[1:-1]
                    for i, cell in enumerate(cells):
                        table.rows[0].cells[i].text = cell.strip()
                    # Add some space after the table
                    doc.add_paragraph().style = doc.styles['Normal']
                elif para:
                    cleaned_para = para.strip('* ').strip()
                    if cleaned_para:
                        p = doc.add_paragraph(cleaned_para, style='CustomNormal')
                        if para.lstrip().startswith('*'):
                            p.style = 'List Bullet'
    
    # Add double borders to every page
    set_page_border(doc)
    
    # Set up the document for optimal layout
    section = doc.sections[0]
    section.page_height = Inches(11)  # Standard letter page height
    section.page_width = Inches(8.5)  # Standard letter page width
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    
    docx_buffer = io.BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)
    return docx_buffer

def set_page_border(doc):
    """
    Adds a double border to every page of the document.
    """
    for section in doc.sections:
        sectPr = section._sectPr
        pgBorders = OxmlElement('w:pgBorders')
        pgBorders.set(qn('w:offsetFrom'), 'page')
        for border_position in ('top', 'left', 'bottom', 'right'):
            border_el = OxmlElement(f'w:{border_position}')
            border_el.set(qn('w:val'), 'double')
            border_el.set(qn('w:sz'), '18') #increase the border width
            border_el.set(qn('w:space'), '24')
            border_el.set(qn('w:color'), 'auto')
            pgBorders.append(border_el)
        sectPr.append(pgBorders)

@app.route('/upload', methods=['POST'])
@rate_limited
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Increase max content length if needed
    # app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    
    if file and file.filename.endswith('.pdf'):
        try:
            start_time = time.time()
            summary, total_pages = process_pdf_in_batches(file)
            processing_time = time.time() - start_time
            
            docx_buffer = create_word_document(summary)
            
            return jsonify({
                'docx': base64.b64encode(docx_buffer.getvalue()).decode('utf-8'),
                'total_pages': total_pages,
                'processing_time': round(processing_time, 2)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400
    
@app.route('/quote', methods=['GET'])
def get_quote():
    try:
        prompt = "Generate a random inspirational quote."
        response = model.generate_content(prompt)
        quote = response.text.strip().strip('"')  # Remove quotes and extra whitespace
        return jsonify({'quote': quote}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to generate quote.'}), 500



if __name__ == '__main__':
    app.run(debug=True)
