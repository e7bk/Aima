// Dashboard JavaScript
class StockDashboard {
    constructor() {
        this.apiBase = '';
        this.currentSymbol = 'AAPL';
        this.isLoading = false;
        this.charts = {};
        
        this.init();
    }
    
    init() {
        this.bindEventListeners();
        this.loadInitialData();
        this.setupAutoRefresh();
    }
    
    bindEventListeners() {
        // Stock selection
        const stockSelect = document.getElementById('stockSelect');
        if (stockSelect) {
            stockSelect.addEventListener('change', (e) => {
                this.currentSymbol = e.target.value;
                this.updateUrl();
            });
        }
        
        // Control buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-action]')) {
                const action = e.target.dataset.action;
                this.handleAction(action, e.target);
            }
        });
        
        // Handle URL parameters
        this.handleUrlParams();
    }
    
    handleUrlParams() {
        const params = new URLSearchParams(window.location.search);
        const symbol = params.get('symbol');
        if (symbol) {
            this.currentSymbol = symbol.toUpperCase();
            const stockSelect = document.getElementById('stockSelect');
            if (stockSelect) {
                stockSelect.value = this.currentSymbol;
            }
        }
    }
    
    updateUrl() {
        const url = new URL(window.location);
        url.searchParams.set('symbol', this.currentSymbol);
        window.history.pushState({}, '', url);
    }
    
    async handleAction(action, button) {
        if (this.isLoading) return;
        
        const originalText = button.textContent;
        this.setButtonLoading(button, true);
        
        try {
            switch (action) {
                case 'load-chart':
                    await this.loadChart();
                    break;
                case 'load-predictions':
                    await this.loadPredictions();
                    break;
                case 'load-comparison':
                    await this.loadComparison();
                    break;
                case 'refresh-data':
                    await this.refreshData();
                    break;
                case 'train-models':
                    await this.trainModels();
                    break;
                default:
                    console.warn('Unknown action:', action);
            }
        } catch (error) {
            this.showError(`Action failed: ${error.message}`);
        } finally {
            this.setButtonLoading(button, false, originalText);
        }
    }
    
    setButtonLoading(button, loading, originalText = '') {
        if (loading) {
            button.disabled = true;
            button.textContent = 'Loading...';
            this.isLoading = true;
        } else {
            button.disabled = false;
            button.textContent = originalText || button.textContent.replace('Loading...', '');
            this.isLoading = false;
        }
    }
    
    async loadInitialData() {
        try {
            await Promise.all([
                this.loadChart(),
                this.loadModelStatus(),
                this.loadStockInfo()
            ]);
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }
    
    async loadChart() {
        const chartContainer = document.getElementById('priceChart');
        if (!chartContainer) return;
        
        try {
            this.showLoading(chartContainer);
            
            const response = await fetch(`${this.apiBase}/analytics/${this.currentSymbol}/chart?days=365&include_predictions=true`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Failed to load chart');
            }
            
            if (data.chart_data) {
                const chartData = JSON.parse(data.chart_data);
                Plotly.newPlot('priceChart', chartData.data, chartData.layout, {responsive: true});
                this.charts.priceChart = true;
            }
            
            this.updateChartInfo(data);
            
        } catch (error) {
            this.showError('Error loading chart: ' + error.message, chartContainer);
        }
    }
    
    async loadPredictions() {
        const container = document.getElementById('predictionsContainer');
        if (!container) return;
        
        try {
            this.showLoading(container);
            
            // Load predictions for multiple models and horizons
            const models = ['RandomForest', 'LSTM', 'Ensemble'];
            const horizons = ['1d', '5d', '10d'];
            
            const predictions = [];
            
            for (const model of models) {
                for (const horizon of horizons) {
                    try {
                        const response = await fetch(`${this.apiBase}/predictions/${this.currentSymbol}?model=${model}&horizon=${horizon}`);
                        const data = await response.json();
                        
                        if (response.ok && data.prediction) {
                            predictions.push({
                                ...data.prediction,
                                current_price: data.current_price
                            });
                        }
                    } catch (error) {
                        console.warn(`Failed to load prediction for ${model} ${horizon}:`, error);
                    }
                }
            }
            
            this.renderPredictions(predictions, container);
            
            // Also load comparison chart
            this.loadComparison();
            
        } catch (error) {
            this.showError('Error loading predictions: ' + error.message, container);
        }
    }
    
    renderPredictions(predictions, container) {
        if (!predictions.length) {
            container.innerHTML = '<p class="error">No predictions available. Try training the models first.</p>';
            return;
        }
        
        const currentPrice = predictions[0].current_price;
        
        const html = predictions.map(pred => `
            <div class="prediction-card">
                <h3>${pred.model} - ${pred.horizon}</h3>
                ${currentPrice ? `<p><strong>Current Price:</strong> <span>$${currentPrice.toFixed(2)}</span></p>` : ''}
                <p><strong>Predicted Return:</strong> <span class="${pred.direction === 'up' ? 'direction-up' : 'direction-down'}">${(pred.predicted_return * 100).toFixed(2)}%</span></p>
                <p><strong>Direction:</strong> <span class="${pred.direction === 'up' ? 'direction-up' : 'direction-down'}">${pred.direction.toUpperCase()}</span></p>
                ${pred.probability ? `<p><strong>Confidence:</strong> <span>${(pred.probability * 100).toFixed(1)}%</span></p>` : ''}
                <p><strong>Target Price:</strong> <span>$${currentPrice ? (currentPrice * (1 + pred.predicted_return)).toFixed(2) : 'N/A'}</span></p>
            </div>
        `).join('');
        
        container.innerHTML = html;
    }
    
    async loadComparison() {
        const chartContainer = document.getElementById('comparisonChart');
        if (!chartContainer) return;
        
        try {
            this.showLoading(chartContainer);
            
            const response = await fetch(`${this.apiBase}/analytics/${this.currentSymbol}/predictions-comparison?horizon=1d`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Failed to load comparison');
            }
            
            if (data.chart_data) {
                const chartData = JSON.parse(data.chart_data);
                Plotly.newPlot('comparisonChart', chartData.data, chartData.layout, {responsive: true});
                this.charts.comparisonChart = true;
            }
            
        } catch (error) {
            this.showError('Error loading comparison: ' + error.message, chartContainer);
        }
    }
    
    async loadModelStatus() {
        try {
            const response = await fetch(`${this.apiBase}/predictions/models/status`);
            const data = await response.json();
            
            if (response.ok) {
                this.updateModelStatus(data);
            }
        } catch (error) {
            console.warn('Error loading model status:', error);
        }
    }
    
    async loadStockInfo() {
        try {
            const response = await fetch(`${this.apiBase}/stocks/${this.currentSymbol}/latest`);
            const data = await response.json();
            
            if (response.ok) {
                this.updateStockInfo(data);
            }
        } catch (error) {
            console.warn('Error loading stock info:', error);
        }
    }
    
    updateModelStatus(data) {
        const statusContainer = document.getElementById('modelStatus');
        if (!statusContainer) return;
        
        const models = data.models || {};
        const html = Object.entries(models).map(([name, status]) => `
            <div class="model-status">
                <div class="status-indicator ${status.is_trained ? 'trained' : ''}"></div>
                <span>${name}: ${status.is_trained ? 'Trained' : 'Not Trained'}</span>
            </div>
        `).join('');
        
        statusContainer.innerHTML = html;
    }
    
    updateStockInfo(data) {
        const infoContainer = document.getElementById('stockInfo');
        if (!infoContainer) return;
        
        infoContainer.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="value">$${data.current_price?.toFixed(2) || 'N/A'}</span>
                    <span class="label">Current Price</span>
                </div>
                <div class="stat-card">
                    <span class="value">$${data.high?.toFixed(2) || 'N/A'}</span>
                    <span class="label">Day High</span>
                </div>
                <div class="stat-card">
                    <span class="value">$${data.low?.toFixed(2) || 'N/A'}</span>
                    <span class="label">Day Low</span>
                </div>
                <div class="stat-card">
                    <span class="value">${data.volume ? (data.volume / 1000000).toFixed(1) + 'M' : 'N/A'}</span>
                    <span class="label">Volume</span>
                </div>
            </div>
        `;
    }
    
    updateChartInfo(data) {
        const infoContainer = document.getElementById('chartInfo');
        if (!infoContainer) return;
        
        infoContainer.innerHTML = `
            <p>📊 Showing ${data.data_points} data points over ${data.days} days</p>
            <p>🔄 Last updated: ${new Date().toLocaleTimeString()}</p>
        `;
    }
    
    async trainModels() {
        try {
            const response = await fetch(`${this.apiBase}/predictions/train/${this.currentSymbol}`, {
                method: 'POST'
            });
            const data = await response.json();
            
            if (response.ok) {
                this.showSuccess(`Model training started for ${this.currentSymbol}. This may take a few minutes.`);
                // Refresh model status periodically
                setTimeout(() => this.loadModelStatus(), 5000);
            } else {
                throw new Error(data.detail || 'Training failed');
            }
        } catch (error) {
            this.showError('Training failed: ' + error.message);
        }
    }
    
    async refreshData() {
        this.showSuccess('Refreshing data...');
        await this.loadInitialData();
        this.showSuccess('Data refreshed successfully!');
    }
    
    showLoading(container) {
        container.innerHTML = '<div class="loading">Loading</div>';
    }
    
    showError(message, container = null) {
        console.error(message);
        const html = `<div class="error">❌ ${message}</div>`;
        
        if (container) {
            container.innerHTML = html;
        } else {
            // Show in notifications area or create one
            this.showNotification(html, 'error');
        }
    }
    
    showSuccess(message) {
        const html = `<div class="success">✅ ${message}</div>`;
        this.showNotification(html, 'success');
    }
    
    showNotification(html, type) {
        let notificationsContainer = document.getElementById('notifications');
        if (!notificationsContainer) {
            notificationsContainer = document.createElement('div');
            notificationsContainer.id = 'notifications';
            notificationsContainer.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 1000; max-width: 400px;';
            document.body.appendChild(notificationsContainer);
        }
        
        const notification = document.createElement('div');
        notification.innerHTML = html;
        notification.style.marginBottom = '10px';
        
        notificationsContainer.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
    
    setupAutoRefresh() {
        // Auto-refresh every 5 minutes
        setInterval(() => {
            if (!this.isLoading) {
                this.loadStockInfo();
            }
        }, 5 * 60 * 1000);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new StockDashboard();
});