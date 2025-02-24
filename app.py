from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware 
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import os
import certifi
import logging
from logging_config import configure_logging

# Configure logging based on environment
environment = os.getenv('ENVIRONMENT', 'production')
configure_logging(environment)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Set SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
gemini = genai.GenerativeModel('gemini-pro')

# Configure OpenAI with the new client format
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, you might want to restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")



templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat/gemini")
async def chat_gemini(message: str = Form(...)):
    try:
        logger.debug(f"Processing Gemini request")
        response = gemini.generate_content(message)
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Error in Gemini processing: {str(e)}")
        return {"response": f"Error: {str(e)}"}

@app.post("/chat/openai")
async def chat_openai(message: str = Form(...)):
    try:
        logger.debug(f"Processing OpenAI request")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in OpenAI processing: {str(e)}")
        return {"response": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    from threading import Timer
    
    def open_browser():
        webbrowser.open("http://localhost:8000")
    
    # Open browser after a 1.5 second delay to ensure server is running
    Timer(1.5, open_browser).start()
    
    # Configure uvicorn logging
    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",  # Change to "debug" for development
        access_log=False  # Set to True for development
    )
    
    server = uvicorn.Server(uvicorn_config)
    server.run()