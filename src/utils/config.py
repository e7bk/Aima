import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Database settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./stock_predictor.db')
    DATABASE_HOST = os.getenv('DATABASE_HOST', 'localhost')
    DATABASE_PORT = int(os.getenv('DATABASE_PORT', 5432))
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'stock_predictor')
    DATABASE_USERNAME = os.getenv('DATABASE_USERNAME', 'postgres')
    DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', 'password')
    
    # API settings
    API_KEY = os.getenv('API_KEY', '')
    API_REQUESTS_PER_MINUTE = int(os.getenv('API_REQUESTS_PER_MINUTE', 60))
    API_TIMEOUT_SECONDS = int(os.getenv('API_TIMEOUT_SECONDS', 30))
    API_MAX_RETRIES = int(os.getenv('API_MAX_RETRIES', 3))
    API_RETRY_DELAY_SECONDS = int(os.getenv('API_RETRY_DELAY_SECONDS', 5))
    
    # Redis settings
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
    
    # App settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    DAYS_OF_HISTORY = int(os.getenv('DAYS_OF_HISTORY', 365))
    DEFAULT_STOCK_SYMBOLS = os.getenv('DEFAULT_STOCK_SYMBOLS', 'AAPL,GOOGL,MSFT,TSLA,AMZN').split(',')
    
    @property
    def database_url(self):
        if self.DATABASE_URL.startswith('sqlite'):
            return self.DATABASE_URL
        return f"postgresql://{self.DATABASE_USERNAME}:{self.DATABASE_PASSWORD}@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
