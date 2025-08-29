#!/usr/bin/env python3
"""
Stock Predictor Demo Launcher

This script launches the lightweight demo version of the Stock Predictor API
that uses mock predictions instead of training heavy ML models.

Perfect for:
- Quick demonstrations
- Development and testing
- Showcasing the UI without computational overhead
- CI/CD environments

Usage:
    python run_demo.py

The demo server will start on http://localhost:8000
"""

import uvicorn
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    print("🚀 Starting Stock Predictor in DEMO MODE")
    print("=" * 50)
    print("✅ No ML model training required")  
    print("✅ Instant startup")
    print("✅ Realistic mock predictions")
    print("✅ All API endpoints functional")
    print("=" * 50)
    print("🌐 Server will start at: http://localhost:8000")
    print("📖 API docs available at: http://localhost:8000/docs")
    print("=" * 50)
    
    # Import and run the demo FastAPI app
    from src.api.demo_main import app
    
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )