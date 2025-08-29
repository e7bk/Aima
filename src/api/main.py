from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.endpoints import stocks, predictions, analytics
from src.api.model_service import model_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Predictor API",
    description="AI-powered stock prediction system with multiple ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "frontend" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Include API routers
app.include_router(stocks.router)
app.include_router(predictions.router)
app.include_router(analytics.router)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("Starting Stock Predictor API...")
        await model_service.initialize_models()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't raise to allow API to start even without models

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Stock Predictor API...")

@app.get("/", response_class=HTMLResponse, summary="Main dashboard")
async def dashboard(request: Request):
    """Main dashboard page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Predictor Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link rel="stylesheet" href="/static/css/dashboard.css">
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Stock Predictor Dashboard</h1>
                <p>AI-powered stock predictions with multiple ML models</p>
            </div>

            <div class="section">
                <h2>Stock Information</h2>
                <div class="controls">
                    <div class="control-group">
                        <label for="stockSelect">Select Stock:</label>
                        <select id="stockSelect">
                            <option value="AAPL">Apple (AAPL)</option>
                            <option value="GOOGL">Google (GOOGL)</option>
                            <option value="MSFT">Microsoft (MSFT)</option>
                            <option value="TSLA">Tesla (TSLA)</option>
                            <option value="AMZN">Amazon (AMZN)</option>
                        </select>
                    </div>
                    <button data-action="refresh-data">Refresh Data</button>
                    <button data-action="train-models" class="secondary">Train Models</button>
                </div>
                <div id="stockInfo"></div>
                <div id="chartInfo"></div>
            </div>

            <div class="section">
                <h2>Price Chart & Analysis</h2>
                <div class="controls">
                    <button data-action="load-chart">Load Chart</button>
                    <button data-action="load-predictions">Get Predictions</button>
                    <button data-action="load-comparison">Compare Models</button>
                </div>
                <div id="priceChart" class="chart-container"></div>
            </div>

            <div class="section">
                <h2>Model Predictions</h2>
                <div id="predictionsContainer" class="predictions-grid">
                    <p class="loading">Click "Get Predictions" to load forecasts</p>
                </div>
            </div>

            <div class="section">
                <h2>Model Comparison</h2>
                <div id="comparisonChart" class="chart-container"></div>
            </div>

            <div class="section">
                <h2>Model Status</h2>
                <div id="modelStatus">
                    <p class="loading">Loading model status...</p>
                </div>
            </div>
        </div>

        <script src="/static/js/dashboard.js"></script>
    </body>
    </html>
    """)

@app.get("/health", summary="Health check")
async def health_check():
    """API health check endpoint"""
    try:
        status = await model_service.get_model_status()
        return {
            "status": "healthy",
            "models_initialized": status["initialized"],
            "api_version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "partial",
            "error": str(e),
            "api_version": "1.0.0"
        }

@app.get("/api/info", summary="API information")
async def api_info():
    """Get API information and available endpoints"""
    return {
        "name": "Stock Predictor API",
        "version": "1.0.0",
        "description": "AI-powered stock prediction system",
        "endpoints": {
            "stocks": "/stocks/ - Stock data and information",
            "predictions": "/predictions/ - ML model predictions",
            "analytics": "/analytics/ - Charts and analysis",
            "docs": "/docs - Interactive API documentation",
            "health": "/health - Health check"
        },
        "models": [
            "RandomForest - Tree-based ensemble model",
            "GradientBoosting - Sequential boosting algorithm", 
            "LSTM - Long Short-Term Memory neural network",
            "BiLSTM - Bidirectional LSTM",
            "Ensemble - Combined multiple models"
        ],
        "prediction_horizons": ["1d", "5d", "10d"],
        "features": [
            "Real-time predictions",
            "Interactive charts",
            "Multiple ML models",
            "Technical indicators",
            "Portfolio analysis"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)