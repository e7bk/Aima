from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class PredictionHorizon(str, Enum):
    """Available prediction horizons"""
    ONE_DAY = "1d"
    FIVE_DAY = "5d"
    TEN_DAY = "10d"

class ModelType(str, Enum):
    """Available model types"""
    RANDOM_FOREST = "RandomForest"
    GRADIENT_BOOSTING = "GradientBoosting"
    LSTM = "LSTM"
    BI_LSTM = "BiLSTM"
    ENSEMBLE = "Ensemble"

class StockRequest(BaseModel):
    """Request model for stock operations"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)", min_length=1, max_length=10)
    
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)", min_length=1, max_length=10)
    horizons: List[PredictionHorizon] = Field(default=[PredictionHorizon.ONE_DAY], description="Prediction horizons")
    models: List[ModelType] = Field(default=[ModelType.ENSEMBLE], description="Models to use for predictions")

class StockInfo(BaseModel):
    """Stock information model"""
    symbol: str
    name: Optional[str] = None
    last_updated: datetime
    total_records: int

class PredictionResult(BaseModel):
    """Single model prediction result"""
    model: ModelType
    horizon: PredictionHorizon
    predicted_return: float
    confidence_score: Optional[float] = None
    direction: str  # "up" or "down"
    probability: Optional[float] = None

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    rmse: float
    mae: float
    r2_score: float
    accuracy: Optional[float] = None
    directional_accuracy: Optional[float] = None

class PredictionResponse(BaseModel):
    """Complete prediction response"""
    symbol: str
    current_price: Optional[float]
    timestamp: datetime
    predictions: List[PredictionResult]
    model_metrics: Dict[str, Dict[str, ModelMetrics]]
    feature_importance: Optional[Dict[str, List[float]]] = None

class HistoricalDataPoint(BaseModel):
    """Historical stock data point"""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
class TechnicalIndicator(BaseModel):
    """Technical indicator values"""
    name: str
    value: float
    signal: Optional[str] = None  # "buy", "sell", "hold"

class ChartDataResponse(BaseModel):
    """Chart data response"""
    symbol: str
    historical_data: List[HistoricalDataPoint]
    predictions: List[PredictionResult]
    technical_indicators: List[TechnicalIndicator]
    chart_url: Optional[str] = None

class AvailableStock(BaseModel):
    """Available stock in database"""
    symbol: str
    name: Optional[str] = None
    last_price: Optional[float] = None
    last_updated: datetime
    total_records: int

class StocksListResponse(BaseModel):
    """List of available stocks"""
    stocks: List[AvailableStock]
    total_count: int

class ModelStatus(BaseModel):
    """Model training status"""
    model: ModelType
    is_trained: bool
    last_trained: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, ModelMetrics]] = None

class SystemStatus(BaseModel):
    """System status response"""
    status: str
    models: List[ModelStatus]
    database_connected: bool
    total_stocks: int
    last_data_update: Optional[datetime] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    
class FeatureImportance(BaseModel):
    """Feature importance data"""
    feature_name: str
    importance: float
    rank: int

class ModelAnalysis(BaseModel):
    """Detailed model analysis"""
    model: ModelType
    horizon: PredictionHorizon
    metrics: ModelMetrics
    feature_importance: List[FeatureImportance]
    prediction_history: List[Dict[str, Any]]

class CompareModelsRequest(BaseModel):
    """Request to compare multiple models"""
    symbol: str
    horizon: PredictionHorizon
    models: List[ModelType]
    include_metrics: bool = True
    include_feature_importance: bool = False

class CompareModelsResponse(BaseModel):
    """Model comparison response"""
    symbol: str
    horizon: PredictionHorizon
    model_analyses: List[ModelAnalysis]
    best_model: ModelType
    timestamp: datetime

class BacktestRequest(BaseModel):
    """Backtesting request"""
    symbol: str
    model: ModelType
    horizon: PredictionHorizon
    start_date: datetime
    end_date: datetime
    initial_investment: float = Field(default=10000.0, gt=0)

class BacktestResult(BaseModel):
    """Backtesting result"""
    symbol: str
    model: ModelType
    horizon: PredictionHorizon
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    performance_chart_url: Optional[str] = None