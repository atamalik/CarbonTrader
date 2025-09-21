# ACCMN - Autonomous Carbon Credit Management Network

A sophisticated AI-powered carbon trading desk system that provides real-time carbon credit pricing, portfolio management, and autonomous trading capabilities.

## 🚀 Features

### 📈 Real-Time Carbon Price Ticker
- **Live price updates** for EUA, CCA, RGGI, VCS, and Gold Standard credits
- **Dynamic simulation** with realistic market movements
- **Visual animations** with glowing effects for price changes
- **Auto-refresh controls** (5, 10, 30, 60 seconds)
- **Professional trading desk appearance**

### 🤖 AI Agent System
- **Multi-agent architecture** using CrewAI and LangGraph
- **Buyer Agent**: Carbon credit purchasing with risk-aware pricing
- **Seller Agent**: Credit sales with market demand analysis
- **Coordinator Agent**: Market coordination and fair pricing
- **Autonomous workflow** for data fetching, validation, and trading

### 💼 Portfolio Management
- **Carbon-adjusted exposure calculations**
- **PCAF-compliant financed emissions**
- **Risk scoring and compliance tracking**
- **Real-time portfolio metrics**
- **ESG analytics and benchmarking**

### 📊 Market Data Integration
- **Real API integration** with DOVU, Carbonmark, AlliedOffsets, and RFF
- **Fallback simulation** for continuous operation
- **Regulatory news tracking**
- **Market sentiment analysis**
- **Rate limiting and error handling**

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- SQLite3
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/atamalik/CarbonTrader.git
cd CarbonTrader

# Create virtual environment
python -m venv clean_venv
source clean_venv/bin/activate  # On Windows: clean_venv\Scripts\activate

# Install dependencies
pip install streamlit plotly pandas numpy sqlite3 requests
pip install streamlit-aggrid
pip install langgraph crewai
```

## 🚀 Quick Start

### Run the Application
```bash
# Start the Streamlit app
streamlit run app_ui.py

# Or run the terminal demo
python demo_realtime_prices.py
```

### Access the Application
- **Web Interface**: http://localhost:8501
- **Markets Tab**: Real-time carbon price ticker
- **Portfolio Tab**: Portfolio management and analytics
- **Trading Tab**: Trade execution and simulation
- **Risk & Compliance**: Risk metrics and compliance tracking
- **AI Agents**: Agent status and workflow controls

## 📁 Project Structure

```
CarbonTrader/
├── app_ui.py                 # Main Streamlit application
├── config.py                 # Configuration and constants
├── data_manager.py           # Data pipeline and caching
├── market_data.py            # External data fetching
├── ui_components.py          # Reusable UI components
├── agent_manager.py          # AI agent orchestration
├── utils.py                  # Utility functions
├── app_mcp.py               # Multi-agent carbon processing
├── simulate_portfolio_update.py  # Portfolio simulation
├── demo_realtime_prices.py  # Terminal demo
├── test_data_loading.py     # Data loading tests
├── test_api_keys.py         # API key testing
├── DataGenerationCodes/     # Data generation scripts
├── CSV Files/               # Data files
└── database/                # Database files
```

## 🔧 Configuration

### API Keys (Optional)
Set environment variables for real-time data:
```bash
export DOVU_API_KEY="your_dovu_key"
export CARBONMARK_API_KEY="your_carbonmark_key"
export ALLIEDOFFSETS_API_KEY="your_alliedoffsets_key"
export RFF_API_KEY="your_rff_key"
```

### Customization
Edit `config.py` to modify:
- UI settings and refresh intervals
- API endpoints and rate limits
- Default values and thresholds
- Compliance and risk configurations

## 📊 Data Sources

### Carbon Price APIs
- **DOVU**: Global carbon credit pricing
- **Carbonmark**: Voluntary carbon market credits
- **AlliedOffsets**: Voluntary market data
- **RFF**: World Carbon Pricing Database

### Fallback Simulation
- **Realistic market dynamics** with trends and momentum
- **Dynamic volatility** based on market conditions
- **Volume correlation** with price movements
- **Professional-grade simulation** for demos

## 🤖 AI Agent Workflow

1. **Data Fetching**: Retrieve company financial and emissions data
2. **Validation**: Validate emissions data quality and completeness
3. **Exposure Calculation**: Calculate carbon-adjusted exposure
4. **Trading**: Execute carbon credit trades based on risk assessment

## 📈 Features in Detail

### Real-Time Price Ticker
- **Visual animations** with color-coded price changes
- **Glowing effects** for price movements (green/red/white)
- **Live timestamps** showing exact update times
- **Professional trading desk appearance**

### Portfolio Analytics
- **Carbon intensity calculations**
- **Risk-adjusted pricing**
- **Compliance tracking**
- **ESG performance metrics**

### Market Intelligence
- **Regulatory news monitoring**
- **Market sentiment analysis**
- **Price trend analysis**
- **Volume correlation insights**

## 🧪 Testing

### Run Tests
```bash
# Test data loading
python test_data_loading.py

# Test API keys
python test_api_keys.py

# Test real-time updates
python demo_realtime_prices.py
```

## 📝 Usage Examples

### Basic Usage
```python
from market_data import get_carbon_prices
from data_manager import get_market_overview

# Get real-time carbon prices
prices = get_carbon_prices(force_refresh=True)

# Get comprehensive market data
market_data = get_market_overview(force_refresh=True)
```

### AI Agent Usage
```python
from agent_manager import start_ai_simulation, trigger_ai_workflow

# Start autonomous AI simulation
start_ai_simulation()

# Trigger AI workflow for specific companies
results = trigger_ai_workflow(max_companies=5)
```

## 🔒 Security & Compliance

- **Data encryption** for sensitive information
- **Rate limiting** for API protection
- **Error handling** with graceful degradation
- **Audit logging** for compliance tracking
- **PCAF compliance** for financed emissions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CrewAI** for multi-agent orchestration
- **LangGraph** for workflow management
- **Streamlit** for the web interface
- **Plotly** for interactive visualizations
- **Carbon market data providers** for real-time pricing

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the `docs/` folder
- Review the test files for usage examples

## 🚀 Roadmap

- [ ] Real-time API integration
- [ ] Advanced risk models
- [ ] Machine learning predictions
- [ ] Mobile app interface
- [ ] Multi-language support
- [ ] Advanced reporting features

---

**ACCMN** - Transforming carbon trading with AI-powered automation and real-time market intelligence.
