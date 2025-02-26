from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import os
import certifi
import logging
from logging_config import configure_logging
from pydantic import BaseModel, Field
from typing import List, Optional

# Configure logging based on environment
environment = os.getenv('ENVIRONMENT', 'production')
configure_logging(environment)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()

# Configure OpenAI with the new client format
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define available models
GEMINI_MODELS = [
    {"id": "gemini-2.0-flash", "name": "2.0-Flash", "description": "*"},
    {"id": "gemini-2.0-flash-lite", "name": "2.0-Lite", "description": "*"},
    {"id": "gemini-1.5-flash", "name": "1.5-Flash", "description": "*"},
    {"id": "gemini-1.5-flash-8b", "name": "1.5-Flash-8B", "description": "*"}
]

OPENAI_MODELS = [
    {"id": "gpt-4o", "name": "4o", "description": "*"},
    {"id": "gpt-4o-mini", "name": "4o-Mini", "description": "*"},
    {"id": "o1", "name": "O1", "description": "*"},
    {"id": "o1-mini", "name": "O1-Mini", "description": "*"}
]

# Define the request models
class ChatRequest(BaseModel):
    message: str = Field(..., description="The message to send to the AI model")
    model_id: Optional[str] = Field(None, description="The specific model ID to use")

class ModelSelectionRequest(BaseModel):
    model_id: str = Field(..., description="The model ID to select")

# Create FastAPI app
app = FastAPI(
    title="AI Model Comparison API",
    description="API for comparing responses from Gemini and OpenAI models with model selection",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Default models
default_gemini_model = "gemini-2.0-flash"
default_openai_model = "gpt-4o"

# In-memory storage for selected models (in a production app, this could be in a database or redis)
class ModelSettings:
    def __init__(self):
        self.current_gemini_model = default_gemini_model
        self.current_openai_model = default_openai_model

model_settings = ModelSettings()

def get_gemini_model(model_id=None):
    """Try to get a specific Gemini model or fall back to alternatives"""
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    
    # Use specified model ID or current selected model
    model_id_to_use = model_id or model_settings.current_gemini_model
    
    try:
        logger.info(f"Attempting to use Gemini model: {model_id_to_use}")
        model = genai.GenerativeModel(model_name=model_id_to_use)
        # Test the model with a simple prompt
        test_response = model.generate_content("Test")
        logger.info(f"Successfully connected to model: {model_id_to_use}")
        return model
    except Exception as e:
        logger.warning(f"Failed to use specified model {model_id_to_use}: {str(e)}")
        
        # Try to fall back to other models if the specified one fails
        fallback_models = [m["id"] for m in GEMINI_MODELS if m["id"] != model_id_to_use]
        
        for fallback_id in fallback_models:
            try:
                logger.info(f"Trying fallback model: {fallback_id}")
                model = genai.GenerativeModel(model_name=fallback_id)
                test_response = model.generate_content("Test")
                logger.info(f"Successfully connected to fallback model: {fallback_id}")
                return model
            except Exception as fallback_error:
                logger.warning(f"Failed to use fallback model {fallback_id}: {str(fallback_error)}")
                continue
        
        # If we get here, we couldn't find any working model
        raise Exception("Could not find any working Gemini model")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat/gemini", summary="Get a response from Gemini model")
async def chat_gemini(message: str = Form(...), model_id: Optional[str] = Form(None)):
    """
    Send a message to the Gemini AI model and get a response.
    
    - **message**: The text message to send to the AI
    - **model_id**: (Optional) The specific Gemini model ID to use
    
    Returns a JSON object with the model's response.
    """
    try:
        logger.debug(f"Processing Gemini request with model: {model_id or model_settings.current_gemini_model}")
        gemini = get_gemini_model(model_id)
        response = gemini.generate_content(message)
        model_used = getattr(gemini, 'model_name', model_settings.current_gemini_model)
        return {
            "response": response.text,
            "model_used": model_used
        }
    except Exception as e:
        logger.error(f"Error in Gemini processing: {str(e)}")
        return {"response": f"Error: {str(e)}", "model_used": "error"}

@app.post("/chat/openai", summary="Get a response from OpenAI model")
async def chat_openai(message: str = Form(...), model_id: Optional[str] = Form(None)):
    """
    Send a message to the OpenAI model and get a response.
    
    - **message**: The text message to send to the AI
    - **model_id**: (Optional) The specific OpenAI model ID to use
    
    Returns a JSON object with the model's response.
    """
    model_to_use = model_id or model_settings.current_openai_model
    
    try:
        logger.debug(f"Processing OpenAI request with model: {model_to_use}")
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": message}]
        )
        return {
            "response": response.choices[0].message.content,
            "model_used": model_to_use
        }
    except Exception as e:
        logger.error(f"Error in OpenAI processing: {str(e)}")
        return {"response": f"Error: {str(e)}", "model_used": "error"}

@app.post("/set/gemini-model", summary="Set the current Gemini model")
async def set_gemini_model(model_id: str = Form(...)):
    """
    Set the current Gemini model to use for requests.
    
    - **model_id**: The model ID to set as current
    """
    try:
        # Validate the model exists in our list
        valid_model = any(m["id"] == model_id for m in GEMINI_MODELS)
        if not valid_model:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Invalid model ID: {model_id}"}
            )
        
        # Test the model to ensure it works
        try:
            test_model = get_gemini_model(model_id)
            # If successful, update the current model
            model_settings.current_gemini_model = model_id
            logger.info(f"Gemini model set to: {model_id}")
            return {"status": "success", "message": f"Gemini model set to: {model_id}", "model": model_id}
        except Exception as model_error:
            logger.error(f"Error testing Gemini model {model_id}: {str(model_error)}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Model {model_id} is not available: {str(model_error)}"}
            )
    except Exception as e:
        logger.error(f"Error setting Gemini model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/set/openai-model", summary="Set the current OpenAI model")
async def set_openai_model(model_id: str = Form(...)):
    """
    Set the current OpenAI model to use for requests.
    
    - **model_id**: The model ID to set as current
    """
    try:
        # Validate the model exists in our list
        valid_model = any(m["id"] == model_id for m in OPENAI_MODELS)
        if not valid_model:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Invalid model ID: {model_id}"}
            )
        
        # Update the current model (we'll assume it works for now, as testing might incur costs)
        model_settings.current_openai_model = model_id
        logger.info(f"OpenAI model set to: {model_id}")
        return {"status": "success", "message": f"OpenAI model set to: {model_id}", "model": model_id}
    except Exception as e:
        logger.error(f"Error setting OpenAI model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/models/gemini", summary="Get available Gemini models")
async def get_gemini_models():
    """Get the list of available Gemini models."""
    return {
        "models": GEMINI_MODELS,
        "current_model": model_settings.current_gemini_model
    }

@app.get("/models/openai", summary="Get available OpenAI models")
async def get_openai_models():
    """Get the list of available OpenAI models."""
    return {
        "models": OPENAI_MODELS,
        "current_model": model_settings.current_openai_model
    }

# API version endpoint for testing
@app.get("/api/version", summary="Get API version information")
async def get_version():
    """
    Get the current API version information.
    
    Returns version details including API version, supported models, and server status.
    """
    return {
        "version": "1.0.0",
        "gemini_model": model_settings.current_gemini_model,
        "openai_model": model_settings.current_openai_model,
        "status": "operational"
    }

# Diagnostic endpoint to check API status
@app.get("/diagnose/gemini", summary="Check Gemini API status")
async def diagnose_gemini():
    """
    Check the status of the Gemini API connection.
    
    Returns diagnostic information about available models and API status.
    """
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        try:
            # Try to list models
            models = genai.list_models()
            available_models = [m.name for m in models if 'gemini' in m.name.lower()]
            
            # Try to create a working model
            try:
                gemini = get_gemini_model()
                model_name = getattr(gemini, 'model_name', 'unknown')
                
                return {
                    "status": "success",
                    "working_model": model_name,
                    "available_models": available_models,
                    "current_model": model_settings.current_gemini_model,
                    "api_key_configured": bool(os.getenv('GOOGLE_API_KEY')),
                    "message": "Gemini API is working correctly"
                }
            except Exception as model_error:
                return {
                    "status": "warning",
                    "error": str(model_error),
                    "available_models": available_models,
                    "current_model": model_settings.current_gemini_model,
                    "api_key_configured": bool(os.getenv('GOOGLE_API_KEY')),
                    "message": "Could not initialize Gemini model, but API is accessible"
                }
        except Exception as list_error:
            return {
                "status": "error",
                "error": str(list_error),
                "current_model": model_settings.current_gemini_model,
                "api_key_configured": bool(os.getenv('GOOGLE_API_KEY')),
                "message": "Could not list available models"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "current_model": model_settings.current_gemini_model,
            "api_key_configured": bool(os.getenv('GOOGLE_API_KEY')),
            "message": "Error connecting to Gemini API"
        }

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