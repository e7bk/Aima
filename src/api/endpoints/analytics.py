from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
import json
import pandas as pd

from src.database.connection import get_db
from src.database.models import Stock, StockPrice
from src.api.models import ModelType, PredictionHorizon
from src.api.model_service import model_service

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/{symbol}/chart", summary="Get interactive price chart")
async def get_price_chart(
    symbol: str,
    days: int = 365,
    include_predictions: bool = True,
    db: Session = Depends(get_db)
):
    """Generate interactive price chart with optional predictions overlay."""
    try:
        # Find stock
        stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        price_data = db.query(StockPrice)\
            .filter(
                StockPrice.stock_id == stock.id,
                StockPrice.date >= start_date
            )\
            .order_by(StockPrice.date.asc())\
            .all()
        
        if not price_data:
            raise HTTPException(status_code=404, detail=f"No price data available for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': price.date,
            'open': price.open_price,
            'high': price.high,
            'low': price.low,
            'close': price.close,
            'volume': price.volume
        } for price in price_data])
        
        # Create candlestick chart
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=f"{symbol} Price"
        ))
        
        # Add predictions if requested
        if include_predictions:
            try:
                predictions = await model_service.predict(
                    symbol=symbol.upper(),
                    horizons=[PredictionHorizon.ONE_DAY, PredictionHorizon.FIVE_DAY],
                    models=[ModelType.ENSEMBLE]
                )
                
                if predictions:
                    latest_price = df['close'].iloc[-1]
                    latest_date = df['date'].iloc[-1]
                    
                    for pred in predictions:
                        future_price = latest_price * (1 + pred.predicted_return)
                        future_date = latest_date + timedelta(days=int(pred.horizon.value.replace('d', '')))
                        
                        fig.add_trace(go.Scatter(
                            x=[latest_date, future_date],
                            y=[latest_price, future_price],
                            mode='lines+markers',
                            name=f"Prediction {pred.horizon.value}",
                            line=dict(dash='dash', width=2)
                        ))
            except Exception:
                pass
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Stock Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=600,
            showlegend=True
        )
        
        # Convert to JSON
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return {
            "symbol": symbol,
            "chart_data": chart_json,
            "days": days,
            "data_points": len(df),
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")

@router.get("/{symbol}/predictions-comparison", summary="Compare model predictions")
async def get_predictions_comparison_chart(
    symbol: str,
    horizon: PredictionHorizon = PredictionHorizon.ONE_DAY,
    models: List[ModelType] = [ModelType.RANDOM_FOREST, ModelType.LSTM, ModelType.ENSEMBLE],
    db: Session = Depends(get_db)
):
    """Generate chart comparing predictions from different models."""
    try:
        # Get predictions from all models
        predictions = await model_service.predict(
            symbol=symbol.upper(),
            horizons=[horizon],
            models=models
        )
        
        if not predictions:
            raise HTTPException(status_code=404, detail="No predictions available")
        
        # Create comparison chart
        fig = go.Figure()
        
        model_names = [pred.model.value for pred in predictions]
        predicted_returns = [pred.predicted_return * 100 for pred in predictions]
        colors = ['green' if pred.direction == 'up' else 'red' for pred in predictions]
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=predicted_returns,
            marker_color=colors,
            text=[f"{r:.2f}%" for r in predicted_returns],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"{symbol} Model Predictions Comparison ({horizon.value})",
            xaxis_title="Models",
            yaxis_title="Predicted Return (%)",
            template="plotly_white",
            height=500
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return {
            "symbol": symbol,
            "chart_data": chart_json,
            "predictions_summary": [
                {
                    "model": pred.model.value,
                    "predicted_return": pred.predicted_return,
                    "direction": pred.direction
                }
                for pred in predictions
            ],
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predictions comparison failed: {str(e)}")