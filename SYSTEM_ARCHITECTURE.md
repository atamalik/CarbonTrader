# CarbonTrader - System Architecture Diagrams

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CarbonTrader System                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   User Interface │    │  Business Logic │    │   Data Layer    │            │
│  │   (Streamlit)   │    │     Layer       │    │   (SQLite)      │            │
│  │                 │    │                 │    │                 │            │
│  │ • app_ui.py     │◄──►│ • data_manager  │◄──►│ • sec_financials│            │
│  │ • ui_components │    │ • market_data   │    │ • Tables: 12+   │            │
│  │ • config.py     │    │ • agent_manager │    │ • Indexes       │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│           │                       │                       │                   │
│           │                       │                       │                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │  AI Agent Layer │    │ External APIs   │    │ Data Generation │            │
│  │                 │    │                 │    │                 │            │
│  │ • app_mcp.py    │    │ • DOVU API      │    │ • generate_*.py │            │
│  │ • LangGraph     │    │ • Carbonmark    │    │ • financial_*   │            │
│  │ • CrewAI        │    │ • AlliedOffsets │    │ • emissions_*   │            │
│  │ • Workflows     │    │ • RFF API       │    │ • lending_*     │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Flow Process                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  External APIs ──┐                                                              │
│                  │                                                              │
│  ┌─────────────┐ │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ Carbon Price│ │    │ Market Data │    │ Data Manager│    │ UI Components│   │
│  │ APIs        │─┼───►│ Fetcher     │───►│             │───►│             │   │
│  │             │ │    │             │    │             │    │             │   │
│  └─────────────┘ │    └─────────────┘    └─────────────┘    └─────────────┘   │
│                  │            │                   │                   │        │
│  Database ───────┼────────────┼───────────────────┼───────────────────┼───────┤
│                  │            │                   │                   │        │
│  ┌─────────────┐ │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ SQLite DB   │ │    │ Caching     │    │ Processing  │    │ Real-time   │   │
│  │             │◄┼────│ Layer       │◄───│ Pipeline    │◄───│ Updates     │   │
│  │ • 12 Tables │ │    │             │    │             │    │             │   │
│  └─────────────┘ │    └─────────────┘    └─────────────┘    └─────────────┘   │
│                  │                                                              │
│  AI Agents ──────┘                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 3. AI Agent Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            AI Agent Workflow (LangGraph)                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Input     │    │ Fetch Data  │    │ Validate    │    │ Calculate   │     │
│  │             │    │             │    │ Emissions   │    │ Exposure    │     │
│  │ • ADSH      │───►│ • Financial │───►│ • Quality   │───►│ • PCAF      │     │
│  │ • Company   │    │ • Activity  │    │ • Complete  │    │ • Risk      │     │
│  │ • Sector    │    │ • Emissions │    │ • Accuracy  │    │ • Pricing   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                              │                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Output    │    │ Trade       │    │ AI Agents   │    │ Coordinator │     │
│  │             │    │ Credits     │    │             │    │ Agent       │     │
│  │ • Portfolio │◄───│             │◄───│ • Buyer     │◄───│             │     │
│  │ • Metrics   │    │ • Execute   │    │ • Seller    │    │ • Market    │     │
│  │ • Trades    │    │ • Update    │    │ • Analysis  │    │ • Pricing   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 4. Database Schema Relationships

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Database Schema Overview                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐                                                               │
│  │ submissions │ (Master Table)                                                 │
│  │ • adsh (PK) │                                                               │
│  │ • name      │                                                               │
│  │ • sic       │                                                               │
│  │ • sector    │                                                               │
│  └─────────────┘                                                               │
│           │                                                                    │
│           │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│           │ │income_stmt  │  │balance_sheet│  │ cash_flow   │                │
│           │ │• revenue    │  │• assets     │  │• operating  │                │
│           │ │• expenses   │  │• liabilities│  │• investing  │                │
│           │ │• net_income │  │• equity     │  │• financing  │                │
│           │ └─────────────┘  └─────────────┘  └─────────────┘                │
│           │                                                                    │
│           │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│           │ │business_    │  │emissions_   │  │lending_data │                │
│           │ │activity     │  │estimates    │  │• exposure   │                │
│           │ │• tag        │  │• scope1/2/3 │  │• credit_lim │                │
│           │ │• unit       │  │• intensity  │  │• risk_score │                │
│           │ │• value      │  │• confidence │  │• rating     │                │
│           │ └─────────────┘  └─────────────┘  └─────────────┘                │
│           │                                                                    │
│           │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│           │ │financed_    │  │carbon_trades│  │ai_workflow_ │                │
│           │ │emissions    │  │• trade_type │  │results      │                │
│           │ │• attribution│  │• quantity   │  │• status     │                │
│           │ │• intensity  │  │• price_usd  │  │• output     │                │
│           │ │• methodology│  │• exchange   │  │• errors     │                │
│           │ └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                                 │
│  ┌─────────────┐  ┌─────────────┐                                             │
│  │emission_    │  │emission_    │                                             │
│  │factors      │  │factors_     │                                             │
│  │• static     │  │research     │                                             │
│  │• conversion │  │• dynamic    │                                             │
│  │• hierarchical│  │• industry   │                                             │
│  └─────────────┘  └─────────────┘                                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 5. User Interface Layout

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Streamlit UI Layout                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                            Header                                      │   │
│  │  • Alerts & Notifications  • System Status  • User Preferences        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────┐  ┌─────────────────────────────────────────────────────────┐   │
│  │   Sidebar   │  │                    Main Content                        │   │
│  │             │  │                                                         │   │
│  │ Navigation: │  │  ┌─────────────────────────────────────────────────┐   │   │
│  │ • Markets   │  │  │              Tab Content                       │   │   │
│  │ • Portfolio │  │  │                                                 │   │   │
│  │ • Trading   │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │   │   │
│  │ • Risk      │  │  │  │   Panel 1   │  │   Panel 2   │  │ Panel 3 │ │   │   │
│  │ • Research  │  │  │  │             │  │             │  │         │ │   │   │
│  │ • AI Agents │  │  │  │ • Metrics   │  │ • Charts    │  │ • Data  │ │   │   │
│  │             │  │  │  │ • Tables    │  │ • Analysis  │  │ • Forms │ │   │   │
│  │ Quick Stats:│  │  │  │ • Controls  │  │ • Insights  │  │ • Tools │ │   │   │
│  │ • Portfolio │  │  │  └─────────────┘  └─────────────┘  └─────────┘ │   │   │
│  │ • Exposure  │  │  │                                                 │   │   │
│  │ • Compliance│  │  │  ┌─────────────────────────────────────────┐   │   │   │
│  │ • Risk      │  │  │  │            Bottom Panel                │   │   │   │
│  │             │  │  │  │                                         │   │   │   │
│  │ Settings:   │  │  │  │ • Advanced Analytics  • AI Insights   │   │   │   │
│  │ • Refresh   │  │  │  │ • Benchmarking       • Reports        │   │   │   │
│  │ • Themes    │  │  │  └─────────────────────────────────────────┘   │   │   │
│  │ • Data      │  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────┘  └─────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 6. Real-time Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Real-time Data Updates                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Timer     │    │ Market Data │    │ Data Cache  │    │ UI Refresh  │     │
│  │             │    │ Fetcher     │    │             │    │             │     │
│  │ Every 10s   │───►│             │───►│ • Prices    │───►│ • Ticker    │     │
│  │ (Prices)    │    │ • API Calls │    │ • News      │    │ • Charts    │     │
│  │ Every 30s   │    │ • Simulation│    │ • Sentiment │    │ • Metrics   │     │
│  │ (Portfolio) │    │ • Fallback  │    │ • Cache     │    │ • Alerts    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│           │                   │                   │                   │        │
│           │                   │                   │                   │        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ AI Agent    │    │ Portfolio   │    │ Risk        │    │ Compliance  │     │
│  │ Workflows   │    │ Updates     │    │ Monitoring  │    │ Tracking    │     │
│  │             │    │             │    │             │    │             │     │
│  │ • Continuous│    │ • Dynamic   │    │ • VaR       │    │ • Thresholds│     │
│  │ • On-demand │    │ • Real-time │    │ • Stress    │    │ • Alerts    │     │
│  │ • Batch     │    │ • Cached    │    │ • Limits    │    │ • Reports   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 7. Security & Configuration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Security & Configuration                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Environment │    │ API Keys    │    │ Rate        │    │ Error       │     │
│  │ Variables   │    │ Management  │    │ Limiting    │    │ Handling    │     │
│  │             │    │             │    │             │    │             │     │
│  │ • .env file │    │ • No hard   │    │ • Per API   │    │ • Graceful  │     │
│  │ • .gitignore│    │   coding    │    │ • Throttling│    │   failure   │     │
│  │ • Examples  │    │ • Env vars  │    │ • Backoff   │    │ • Fallbacks │     │
│  │ • Templates │    │ • Rotation  │    │ • Queuing   │    │ • Logging   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│           │                   │                   │                   │        │
│           │                   │                   │                   │        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Data        │    │ Access      │    │ Audit       │    │ Monitoring  │     │
│  │ Encryption  │    │ Control     │    │ Logging     │    │ & Alerts    │     │
│  │             │    │             │    │             │    │             │     │
│  │ • Sensitive │    │ • User      │    │ • API calls │    │ • Health    │     │
│  │ • Database  │    │   roles     │    │ • Changes   │    │   checks    │     │
│  │ • Transit   │    │ • Perms     │    │ • Errors    │    │ • Metrics   │     │
│  │ • At rest   │    │ • Sessions  │    │ • Actions   │    │ • Alerts    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 8. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Deployment Options                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Development │    │ Staging     │    │ Production  │    │ Cloud       │     │
│  │             │    │             │    │             │    │ Deployment  │     │
│  │ • Local     │    │ • Test      │    │ • On-prem   │    │ • AWS       │     │
│  │ • SQLite    │    │ • Simulated │    │ • PostgreSQL│    │ • Azure     │     │
│  │ • Mock APIs │    │ • Real APIs │    │ • Redis     │    │ • GCP       │     │
│  │ • Debug     │    │ • Load test │    │ • Monitoring│    │ • Kubernetes│     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│           │                   │                   │                   │        │
│           │                   │                   │                   │        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ CI/CD       │    │ Monitoring  │    │ Backup      │    │ Scaling     │     │
│  │ Pipeline    │    │ & Logging   │    │ & Recovery  │    │ Strategy    │     │
│  │             │    │             │    │             │    │             │     │
│  │ • GitHub    │    │ • Prometheus│    │ • Automated │    │ • Horizontal│     │
│  │ • Tests     │    │ • Grafana   │    │ • Point-in- │    │ • Load      │     │
│  │ • Deploy    │    │ • Alerts    │    │   time      │    │   balancing │     │
│  │ • Rollback  │    │ • Dashboards│    │ • Disaster  │    │ • Auto      │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 9. Performance Metrics

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Performance Benchmarks                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Data        │    │ API         │    │ UI          │    │ AI Agent    │     │
│  │ Loading     │    │ Response    │    │ Rendering   │    │ Execution   │     │
│  │             │    │ Times       │    │             │    │             │     │
│  │ • Portfolio │    │ • DOVU:     │    │ • Page      │    │ • Single    │     │
│  │   2-5s      │    │   200-500ms │    │   load:     │    │   company:  │     │
│  │ • Market    │    │ • Carbonmark│    │   1-3s      │    │   5-15s     │     │
│  │   1-3s      │    │   300-800ms │    │ • Charts:   │    │ • Batch     │     │
│  │ • Trades    │    │ • Allied:   │    │   500ms-1s  │    │   (5):      │     │
│  │   1-2s      │    │   400-1s    │    │ • Updates:  │    │   30-60s    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│           │                   │                   │                   │        │
│           │                   │                   │                   │        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Database    │    │ Memory      │    │ CPU         │    │ Network     │     │
│  │ Operations  │    │ Usage       │    │ Usage       │    │ Bandwidth   │     │
│  │             │    │             │    │             │    │             │     │
│  │ • Queries:  │    │ • Base:     │    │ • Idle:     │    │ • API calls │     │
│  │   10-50ms   │    │   100-200MB │    │   5-15%     │    │   1-5MB/s   │     │
│  │ • Updates:  │    │ • Peak:     │    │ • Active:   │    │ • Data sync │     │
│  │   20-100ms  │    │   500MB-1GB │    │   20-50%    │    │   10-50MB/s │     │
│  │ • Indexes:  │    │ • Cache:    │    │ • AI:       │    │ • Streaming │     │
│  │   <5ms      │    │   50-100MB  │    │   50-80%    │    │   5-20MB/s  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 10. Integration Points

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            External Integrations                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Carbon      │    │ Financial   │    │ Regulatory  │    │ AI/ML       │     │
│  │ Markets     │    │ Data        │    │ Systems     │    │ Services    │     │
│  │             │    │             │    │             │    │             │     │
│  │ • ICE       │    │ • Bloomberg │    │ • SEC       │    │ • OpenAI    │     │
│  │ • ARB       │    │ • Reuters   │    │ • EPA       │    │ • Anthropic │     │
│  │ • RGGI      │    │ • Yahoo     │    │ • EU ETS    │    │ • LangChain │     │
│  │ • VCS       │    │ • Alpha     │    │ • CBAM      │    │ • Hugging   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│           │                   │                   │                   │        │
│           │                   │                   │                   │        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Trading     │    │ Risk        │    │ Compliance  │    │ Reporting   │     │
│  │ Systems     │    │ Management  │    │ Platforms   │    │ Tools       │     │
│  │             │    │             │    │             │    │             │     │
│  │ • FIX       │    │ • RiskMetrics│   │ • Workiva   │    │ • Tableau   │     │
│  │ • OMS       │    │ • MSCI      │    │ • Diligent  │    │ • Power BI  │     │
│  │ • EMS       │    │ • Refinitiv │    │ • MetricStream│  │ • Excel     │     │
│  │ • TMS       │    │ • Moody's   │    │ • GRC       │    │ • PDF       │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

These architecture diagrams provide a comprehensive visual representation of the CarbonTrader system, showing:

1. **System Architecture**: High-level component relationships
2. **Data Flow**: How data moves through the system
3. **AI Workflows**: LangGraph agent execution flow
4. **Database Schema**: Table relationships and structure
5. **UI Layout**: Streamlit interface organization
6. **Real-time Updates**: Live data refresh mechanisms
7. **Security**: Configuration and protection measures
8. **Deployment**: Various deployment options
9. **Performance**: Benchmarks and metrics
10. **Integrations**: External system connections

This visual documentation complements the technical documentation and provides a clear understanding of how all components work together to create a comprehensive carbon trading platform.
