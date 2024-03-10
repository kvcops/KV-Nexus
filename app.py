from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
app = Flask(__name__)

api_key = os.environ.get("API_KEY")

genai.configure(api_key=api_key)

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

chat_model = genai.GenerativeModel("gemini-pro", generation_config=chat_generation_config)
chef_model = genai.GenerativeModel("gemini-pro", generation_config=chef_generation_config)
story_model = genai.GenerativeModel("gemini-pro", generation_config=story_generation_config)
psychology_model = genai.GenerativeModel("gemini-pro", generation_config=psychology_generation_config)
code_model = genai.GenerativeModel("gemini-pro", generation_config=code_generation_config)
algorithm_model = genai.GenerativeModel("gemini-pro", generation_config=algorithm_generation_config)

def format_response(response_text):
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    formatted_text = '<br>'.join(lines)
    return formatted_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_query = request.form['user_query']
        prompt = f"{user_query}"
        response = chat_model.generate_content([prompt])
        response_text = format_response(response.text)
        return jsonify({'response': response_text})
    return render_template('chat.html')

@app.route('/chef', methods=['GET', 'POST'])
def chef():
    if request.method == 'POST':
        user_ingredients = request.form['user_ingredients']
        prompt = f"Generate a recipe based on the following ingredients {user_ingredients} and explain the steps to cook it in a stepwise manner and formatted manner ..also explain who can eat and who shouldnt eat."
        response = chef_model.generate_content([prompt])
        response_text = format_response(response.text)
        return jsonify({'response': response_text})
    return render_template('chef.html')

@app.route('/story_generator', methods=['GET', 'POST'])
def story_generator():
    if request.method == 'POST':
        user_input_words = request.form['keywords']
        prompt = f"Generate a story based on the following words {user_input_words}...the story should be entertaining."
        response = story_model.generate_content([prompt])
        response_text = format_response(response.text)
        return jsonify({'response': response_text})
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


if __name__ == '__main__':
    app.run(debug=True)