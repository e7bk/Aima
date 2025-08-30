📈 Aima

A comprehensive AI-powered stock prediction system that leverages multiple machine learning models to forecast stock price movements. Built with Python, FastAPI, and TensorFlow, this application provides real-time predictions, interactive visualisations, and a complete web-based dashboard.

---

## ✨ Features

* **Multiple ML Models**: LSTM, Random Forest, Gradient Boosting, Bidirectional LSTM, and Ensemble methods
* **Real-time Predictions**: Get instant stock forecasts for 1-day, 5-day, and 10-day horizons
* **Interactive Dashboard**: Web-based interface with live charts and analytics
* **Technical Indicators**: Comprehensive analysis including moving averages, RSI, MACD, and Bollinger Bands
* **Demo Mode**: Lightweight demonstration version with mock predictions for instant setup
* **RESTful API**: Complete API with interactive documentation
* **Database Integration**: PostgreSQL storage for historical data and predictions
* **Caching Layer**: Redis integration for improved performance

---

## 📚 Table of Contents

* [Installation](#installation)
* [Quick Start](#quick-start)
* [Demo Mode](#demo-mode)
* [API Documentation](#api-documentation)
* [Architecture](#architecture)
* [Models](#models)
* [Configuration](#configuration)
* [Development](#development)
* [Testing](#testing)
* [Technical Indicators](#technical-indicators)
* [Docker Support](#docker-support)
* [Disclaimer](#disclaimer)
* [Licence](#licence)
* [Contributing](#contributing)
* [Support](#support)

---

## ⚙️ Installation

### 🔧 Prerequisites

* Python 3.8 or higher
* PostgreSQL (for production mode)
* Redis (optional, for caching)

### 🚀 Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stock-predictor.git
   cd stock-predictor
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (optional for demo mode):

   ```bash
   cp .env.example .env
   # Edit .env with your database and API configurations
   ```

---

## ⚡ Quick Start

### 🧪 Demo Mode (Recommended for First Run)

```bash
python run_demo.py
```

Visit [http://localhost:8000](http://localhost:8000) to access the dashboard.

### 🏗️ Production Mode

1. Set up PostgreSQL database

2. Configure environment variables in `.env`

3. Run database migrations:

   ```bash
   python create_tables.py
   ```

4. Start the application:

   ```bash
   cd src && python -m uvicorn api.main:app --reload
   ```

Or using Docker Compose:

```bash
docker-compose up
```

---

## 🧩 Demo Mode

Demo mode provides all the functionality without computational overhead:

* **Instant startup** (< 1 second)
* **Realistic mock predictions** using statistical models
* **Full API compatibility** with production mode
* **Interactive dashboard** with all features
* **Perfect for demonstrations** and development

---

## 📡 API Documentation

Once running, interactive API documentation is available at:

* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 🛠️ Key Endpoints

* `POST /predictions/` — Get predictions for multiple models and horizons
* `GET /predictions/{symbol}` — Quick prediction for a specific stock
* `POST /predictions/train/{symbol}` — Train models for a specific stock
* `GET /predictions/models/status` — Check model training status
* `GET /stocks/{symbol}` — Get stock information and current price
* `GET /analytics/technical/{symbol}` — Technical indicator analysis

---

## 🧱 Architecture

The application follows a modular architecture with clear separation of concerns:

```
src/
├── api/               # FastAPI application and endpoints
│   ├── main.py        # Production API server
│   ├── demo_main.py   # Demo mode server
│   └── endpoints/     # API route definitions
├── data/              # Data collection and preprocessing
│   ├── collector.py   # Stock data fetching (yfinance)
│   └── indicators.py  # Technical indicators calculation
├── models/            # Machine learning models
│   ├── lstm_model.py  # LSTM neural network
│   ├── random_forest.py # Random Forest ensemble
│   └── ensemble.py    # Model combination strategies
├── database/          # Database models and connections
│   ├── models.py      # SQLAlchemy ORM models
│   └── connection.py  # Database configuration
└── frontend/          # Web dashboard
    └── static/        # HTML, CSS, JavaScript files
```

---

## 🧠 Models

### 🤖 Available Models

1. **LSTM (Long Short-Term Memory)**
2. **Random Forest**
3. **Gradient Boosting**
4. **Bidirectional LSTM**
5. **Ensemble Model**

### 📊 Model Performance Metrics

* **RMSE** (Root Mean Square Error)
* **MAE** (Mean Absolute Error)
* **R²** (Coefficient of Determination)
* **Directional Accuracy** (Up/Down prediction accuracy)

---

## 🛠️ Configuration

Configuration is managed through environment variables in `src/utils/config.py`:

```python
# Database Configuration
DATABASE_URL = "postgresql://user:password@localhost/stockdb"

# API Configuration
API_RATE_LIMIT = 100  # requests per minute
CORS_ORIGINS = ["*"]

# Data Configuration
DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
HISTORY_PERIOD = "2y"  # Data collection period

# Redis Configuration (optional)
REDIS_URL = "redis://localhost:6379/0"
```

---

## 🧪 Development

### 🧼 Code Quality

Format code:

```bash
black src/
```

Lint code:

```bash
flake8 src/
```

### 🧪 Running Tests

```bash
pytest tests/
```

### ➕ Adding New Models

1. Create model class inheriting from `BaseModel` in `src/models/`
2. Implement required methods: `train()`, `predict()`, `evaluate()`
3. Register model in `src/api/model_service.py`
4. Add model type to `src/api/models.py`

---

## ✅ Testing

The project includes comprehensive tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/test_models/        # Model tests
pytest tests/test_api/           # API tests
pytest tests/test_data/          # Data processing tests
```

---

## 📈 Technical Indicators

The system calculates various technical indicators:

* **Trend Indicators**: SMA, EMA, MACD
* **Momentum Indicators**: RSI, Stochastic Oscillator
* **Volatility Indicators**: Bollinger Bands, ATR
* **Volume Indicators**: OBV, Volume SMA

---

## 🐳 Docker Support

Run the complete stack with Docker Compose:

```bash
# Start all services (API, PostgreSQL, Redis)
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## ⚠️ Disclaimer

**Important**: This application is for educational and research purposes only. Stock predictions are inherently uncertain, and this system should not be used for actual financial decisions. Always consult with qualified financial advisors before making investment decisions.

---

## 📄 Licence

This project is licensed under the MIT Licence. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### 🛠️ Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Format your code (`black src/`)
7. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
8. Push to the branch (`git push origin feature/AmazingFeature`)
9. Open a Pull Request

---

## 💬 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Built with Python, FastAPI, TensorFlow, and modern web technologies for reliable stock market analysis.**
