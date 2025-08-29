# 🚀 Stock Predictor Demo Mode

This demo version provides all the functionality of the full Stock Predictor API without the computational overhead of training ML models. Perfect for demonstrations, development, and testing!

## ✨ Demo Features

- **🚀 Instant Startup**: No ML model loading or training required
- **🎯 Realistic Mock Predictions**: Generates believable stock predictions using statistical models
- **🔄 All API Endpoints**: Complete API compatibility with the full version
- **📊 Interactive Dashboard**: Full web interface with charts and visualizations  
- **🎲 Consistent Results**: Same symbol always generates similar predictions for demo consistency

## 🚦 Quick Start

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Demo
```bash
# Option 1: Use the demo launcher
python run_demo.py

# Option 2: Direct uvicorn command  
uvicorn src.api.demo_main:app --host 0.0.0.0 --port 8000 --reload
```

The demo server will start at: **http://localhost:8000**

API documentation: **http://localhost:8000/docs**

## 🎯 Demo vs Production Comparison

| Feature | Production | Demo Mode |
|---------|------------|-----------|
| Startup Time | ~30-60 seconds | <1 second |
| Model Training | 5 complex ML models | Mock training (instant) |
| Memory Usage | High (ML models loaded) | Low (no models) |
| Predictions | Real ML predictions | Realistic mock data |
| Database Required | Yes (for training data) | Optional (falls back to mocks) |
| GPU/CPU Usage | High during training | Minimal |

## 🔧 Demo Architecture

### Core Demo Components

- **`demo_model_service.py`** - Mock ML service that generates realistic predictions
- **`demo_endpoints.py`** - API endpoints that use mock predictions  
- **`demo_main.py`** - FastAPI app configured for demo mode
- **`run_demo.py`** - Convenient launcher script

### Mock Prediction Logic

The demo generates realistic predictions by:

1. **Stock-specific volatility** - Different stocks have different prediction ranges
2. **Model-specific biases** - Each "model" has consistent characteristics
3. **Horizon scaling** - Longer-term predictions are less certain
4. **Realistic confidence scores** - Based on prediction magnitude and model type

### Supported Mock Models

- **RandomForest** - Mock tree-based ensemble (slightly pessimistic bias)
- **GradientBoosting** - Mock boosting algorithm (neutral bias) 
- **LSTM** - Mock neural network (optimistic bias)
- **BiLSTM** - Mock bidirectional LSTM (slight optimistic bias)
- **Ensemble** - Mock combined model (balanced, highest confidence)

## 🎮 Demo API Usage

### Get Mock Predictions
```bash
curl -X POST "http://localhost:8000/predictions/" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "horizons": ["1d", "5d", "10d"],
    "models": ["RandomForest", "LSTM", "Ensemble"]
  }'
```

### Quick Prediction
```bash
curl "http://localhost:8000/predictions/TSLA?model=Ensemble&horizon=1d"
```

### Mock Training (completes instantly)
```bash
curl -X POST "http://localhost:8000/predictions/train/AAPL"
```

### Model Status
```bash
curl "http://localhost:8000/predictions/models/status"
```

## 💡 Use Cases

### 🎪 Demonstrations
- Show the complete system without setup delays
- Present to stakeholders without computational requirements
- Demo at conferences or meetings

### 🧪 Development & Testing  
- Develop frontend without backend complexity
- Test API integration without ML dependencies
- CI/CD pipelines that need fast feedback

### 📚 Learning & Exploration
- Understand the API structure and responses
- Experiment with different request patterns
- Learn the system architecture

## 🔄 Converting Demo to Production

To switch from demo to production mode:

1. **Use the full server**: Run `src/api/main.py` instead of `demo_main.py`
2. **Setup database**: Ensure PostgreSQL and Redis are configured
3. **Load stock data**: Use the data collection services
4. **Train models**: Allow time for actual ML model training

## 🎛️ Demo Configuration

### Customizing Mock Predictions

Edit `src/api/demo_model_service.py` to adjust:

- **Stock volatilities**: Modify `_stock_volatility` dictionary
- **Model biases**: Adjust `model_bias` in `_generate_mock_prediction()`
- **Confidence calculations**: Update `_calculate_confidence()` method
- **Available stocks**: Add more symbols to volatility mapping

### Mock Performance Metrics

Pre-configured realistic metrics for each model:
- RMSE, MAE, R² scores
- Accuracy percentages  
- Feature importance rankings

## 🚨 Important Notes

- **Demo data only**: Predictions are mock data, not real ML predictions
- **No persistent training**: Mock "training" doesn't save any models
- **Database optional**: System works without database connections
- **Development only**: Not for production trading decisions!

## 🆘 Troubleshooting

### Common Issues

**Import errors**: Make sure all dependencies are installed via `pip install -r requirements.txt`

**Port conflicts**: Change the port in `run_demo.py` if 8000 is in use

**Database warnings**: Normal in demo mode - database is optional

### Getting Help

- Check the FastAPI docs at `/docs` for API details
- See the main project README for full system documentation
- All endpoints include demo mode indicators in responses

---

**⚡ Ready to demo? Run `python run_demo.py` and visit http://localhost:8000**