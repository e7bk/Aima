from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.endpoints import stocks, analytics
from src.api.demo_endpoints import demo_predictions  # Use demo predictions
from src.api.demo_model_service import demo_model_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Predictor API - Demo Mode",
    description="AI-powered stock prediction system with mock predictions for demonstration",
    version="1.0.0-demo",
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

# Include API routers - using demo predictions
app.include_router(stocks.router)
app.include_router(demo_predictions)  # Mock predictions instead of real ones
app.include_router(analytics.router)

@app.on_event("startup")
async def startup_event():
    """Initialize demo services on startup - instant startup!"""
    try:
        logger.info("Starting Stock Predictor API in DEMO MODE...")
        logger.info("🚀 Using mock predictions - no ML model training required!")
        await demo_model_service.initialize_models()
        logger.info("✅ Demo API startup completed successfully in <1 second!")
    except Exception as e:
        logger.error(f"Demo startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Stock Predictor API (Demo Mode)...")

@app.get("/", response_class=HTMLResponse, summary="Main dashboard")
async def dashboard(request: Request):
    """Main dashboard page with demo mode indicator"""
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Predictor Dashboard - Demo Mode</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link rel="stylesheet" href="/static/css/dashboard.css">
        <style>
            .demo-banner {{
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                color: white;
                padding: 10px;
                text-align: center;
                font-weight: bold;
                border-radius: 5px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .demo-info {{
                background: #f8f9fa;
                border-left: 4px solid #4ecdc4;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="demo-banner">
                🚀 DEMO MODE - Using Mock Predictions (No ML Training Required)
            </div>
            
            <div class="header">
                <h1>Stock Predictor Dashboard</h1>
                <p>AI-powered stock predictions with multiple ML models</p>
            </div>

            <div class="demo-info">
                <h3>🎯 Demo Mode Features:</h3>
                <ul>
                    <li>✅ Instant startup - no model training delays</li>
                    <li>✅ Realistic mock predictions for 5 different models</li>
                    <li>✅ All API endpoints work identically to production</li>
                    <li>✅ Interactive charts and visualizations</li>
                    <li>✅ Perfect for demos, testing, and development</li>
                </ul>
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
                    <button data-action="train-models" class="secondary">Mock Training</button>
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
        status = await demo_model_service.get_model_status()
        return {
            "status": "healthy",
            "demo_mode": True,
            "models_initialized": status["initialized"],
            "api_version": "1.0.0-demo",
            "message": "Running in demo mode with mock predictions"
        }
    except Exception as e:
        return {
            "status": "partial",
            "demo_mode": True,
            "error": str(e),
            "api_version": "1.0.0-demo"
        }

@app.get("/api/info", summary="API information")
async def api_info():
    """Get API information and available endpoints"""
    return {
        "name": "Stock Predictor API",
        "version": "1.0.0-demo",
        "mode": "DEMO - Mock Predictions",
        "description": "AI-powered stock prediction system - demo mode with realistic mock data",
        "demo_features": [
            "Instant startup (no ML model loading)",
            "Realistic mock predictions", 
            "All API endpoints functional",
            "Perfect for demonstrations and testing"
        ],
        "endpoints": {
            "stocks": "/stocks/ - Stock data and information",
            "predictions": "/predictions/ - Mock ML model predictions", 
            "analytics": "/analytics/ - Charts and analysis",
            "docs": "/docs - Interactive API documentation",
            "health": "/health - Health check"
        },
        "mock_models": [
            "RandomForest - Mock tree-based ensemble model",
            "GradientBoosting - Mock sequential boosting algorithm", 
            "LSTM - Mock Long Short-Term Memory neural network",
            "BiLSTM - Mock Bidirectional LSTM",
            "Ensemble - Mock combined multiple models"
        ],
        "prediction_horizons": ["1d", "5d", "10d"],
        "features": [
            "Instant mock predictions",
            "Interactive charts",
            "Mock multiple ML models",
            "Mock technical indicators",
            "Demo portfolio analysis"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)