from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Google API
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Initialize the phidata agent
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

# Create the agent instance
multimodal_Agent = initialize_agent()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Get the query from form data
        user_query = request.form.get('query', '')
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400

        # Save temporary file and get its path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            file.save(temp_video.name)
            video_path = temp_video.name

        # Read the file content
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()

        # Create generation config for direct Gemini API
        generation_config = {
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 2048,
        }

        # Initialize the Gemini model for video processing
        model = genai.GenerativeModel('gemini-2.0-flash-exp',
                                    generation_config=generation_config)

        # Create the video analysis prompt
        video_prompt = f"""
        Analyze the uploaded video for content and context.
        Extract the main topics, events, and key elements from the video.
        Provide a detailed summary of the video content.
        """

        # Create the content for video analysis
        video_contents = {
            "parts": [
                {"text": video_prompt},
                {"inline_data": {
                    "mime_type": "video/mp4",
                    "data": base64.b64encode(video_data).decode('utf-8')
                }}
            ]
        }

        # Get video analysis
        video_analysis = model.generate_content(video_contents)
        
        # Create comprehensive analysis prompt for the agent
        agent_prompt = f"""
        Based on this video analysis: {video_analysis.text}
        
        Research and respond to the following query:
        {user_query}
        
        Use the search tool to gather additional context and information.
        Provide a comprehensive, well-researched response that combines 
        both the video insights and supplementary information.
        """

        # Get agent response with search capability
        agent_response = multimodal_Agent.run(agent_prompt)
        
        # Clean up temporary file
        Path(video_path).unlink(missing_ok=True)

        return jsonify({
            'success': True,
            'analysis': agent_response.content
        })

    except Exception as e:
        # Clean up on error
        if 'video_path' in locals():
            Path(video_path).unlink(missing_ok=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)