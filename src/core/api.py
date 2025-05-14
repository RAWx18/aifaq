import uvicorn
from utils import load_yaml_file
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes.main import router as main_router
from routes.test import router as test_router
from routes.main import guardrails_processor as main_guardrails_processor

from conversation import initialize_guardrails
from guardrails import GuardrailProcessor, GuardrailConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_data = load_yaml_file("config.yaml")

def create_app() -> FastAPI:
    # Create the FastAPI application
    app = FastAPI()

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Add an endpoint to update guardrails configuration
    @app.post("/update_guardrails")
    async def update_guardrails(request: Request):
        """Update the guardrails configuration."""
        data = await request.json()
        logger.info(f"Updating guardrails with: {data}")
        
        # Create a new guardrails config
        config = GuardrailConfig()
        
        # Update from request data
        if 'blocked_topics' in data:
            config.blocked_topics = data['blocked_topics']
            logger.info(f"Set blocked_topics to: {data['blocked_topics']}")
        if 'filtered_patterns' in data:
            config.filtered_patterns = data['filtered_patterns']
        if 'max_response_length' in data:
            config.max_response_length = data['max_response_length']
            logger.info(f"Set max_response_length to: {data['max_response_length']}")
        if 'enable_safety_checks' in data:
            config.enable_safety_checks = data['enable_safety_checks']
        if 'custom_responses' in data:
            config.custom_responses = data['custom_responses']
        if 'disclaimers' in data:
            config.disclaimers = data['disclaimers']
        
        # Update the global guardrails processor
        from routes.main import guardrails_processor
        new_processor = GuardrailProcessor(config)
        
        # Update in routes/main.py
        globals()['main_guardrails_processor'] = new_processor
        
        # For debugging
        logger.info(f"Successfully updated guardrails. Blocked topics: {config.blocked_topics}")
        
        return {"status": "success", "message": "Guardrails updated successfully"}

    app.include_router(main_router)
    app.include_router(test_router)

    @app.get("/health")
    async def health_check():
        """Simple health check endpoint"""
        return {"status": "ok", "message": "API is running"}
    
    return app


# For direct running
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host=config_data["host"], port=config_data["port"])
else:
    # For import by uvicorn
    app = create_app()