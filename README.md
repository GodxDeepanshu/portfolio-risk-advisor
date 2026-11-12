# AI-Driven Quantitative Portfolio Risk Advisor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

An institutional-grade quantitative risk modeling and optimization platform. Features continuous Markowitz Modern Portfolio Theory (MPT) quadratic programming, 10,000+ iteration vectorized Monte Carlo simulations, parametric and historical tail-risk diagnostics (VaR 95%/99%, Conditional VaR / Expected Shortfall), and an NLP headline sentiment pipeline for dynamic portfolio risk exposure rebalancing.

---

## 📌 Key Architectural Pillars

1. **Markowitz Efficient Frontier Optimization**:
   - Solves non-linear constrained portfolio allocations using Sequential Least Squares Programming (`SLSQP`).
   - Identifies the **Global Minimum Variance (GMV)** and **Maximum Sharpe Ratio (Tangency)** portfolios.
   - Generates the continuous Capital Allocation Line (CAL).

2. **10,000+ Vectorized Monte Carlo Path Simulation**:
   - Simulates multi-asset correlated Geometric Brownian Motion (GBM) via Cholesky decomposition ($L L^T = \Sigma$).
   - Projects distribution fan charts across 252+ trading days to compute probability of capital loss, terminal wealth percentiles ($P_5, P_{50}, P_{95}$), and stress drawdowns.

3. **Institutional Tail Risk Diagnostics**:
   - **Parametric VaR (95% & 99%)**: Analytical normal variance-covariance framework.
   - **Historical VaR & CVaR (Expected Shortfall)**: Non-parametric empirical quantile analysis accounting for fat tails and leptokurtic return distributions.
   - **Downside Risk Ratios**: Sharpe Ratio, Sortino Ratio, Maximum Drawdown (MDD), and Calmar Ratio.

4. **NLP Sentiment Alpha Dynamic Tilting**:
   - Ingests financial headlines and market commentary.
   - Computes continuous polarity scores $\alpha_i \in [-1, 1]$ per ticker.
   - Dynamically tilts base portfolio weights while maintaining standard simplex constraints ($\sum w_i = 1$).

5. **Containerized Interactive Dashboard**:
   - Streamlit interface for live scenario testing, asset reweighting, and visual risk inspection.
   - Containerized with Docker and `docker-compose` for reproducible deployments.

---

## 📐 Mathematical Formulation

### 1. Portfolio Return and Variance
$$\mathbb{E}[R_p] = \mathbf{w}^T \boldsymbol{\mu}$$
$$\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$$

### 2. Maximum Sharpe Ratio Optimization
$$\max_{\mathbf{w}} \frac{\mathbf{w}^T \boldsymbol{\mu} - R_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}} \quad \text{s.t.} \quad \sum_{i=1}^n w_i = 1, \quad 0 \le w_i \le 1$$

### 3. Value at Risk (VaR) & Conditional VaR (CVaR)
$$\text{VaR}_\alpha = -\inf \{ l \in \mathbb{R} : P(L > l) \le 1 - \alpha \}$$
$$\text{CVaR}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha]$$

### 4. Correlated Monte Carlo Path Generation
$$\mathbf{S}_{t+\Delta t} = \mathbf{S}_t \exp \left( \left(\boldsymbol{\mu} - \frac{1}{2}\boldsymbol{\sigma}^2\right)\Delta t + \boldsymbol{\sigma}\sqrt{\Delta t} \, \mathbf{L} \mathbf{Z} \right)$$
where $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $\mathbf{L}$ is the Cholesky factor of correlation matrix $\mathbf{R}$.

---

## 🚀 Quickstart

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/GodxDeepanshu/portfolio-risk-advisor.git
cd portfolio-risk-advisor

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Unit & Integration Test Suite
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

To build and run via Docker:
```bash
docker build -t portfolio-risk-advisor .
docker run -p 8501:8501 portfolio-risk-advisor
```

Or with Docker Compose:
```bash
docker-compose up -d
```

---

## 📂 Project Structure

```
portfolio-risk-advisor/
├── app.py                     # Streamlit Interactive Web Application
├── Dockerfile                 # Multi-stage production container
├── docker-compose.yml         # Container orchestration spec
├── requirements.txt           # Core Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Correlated Geometric Brownian Motion & market data
│   ├── markowitz.py           # Efficient Frontier & SLSQP portfolio optimization
│   ├── monte_carlo.py         # 10,000+ vectorized Monte Carlo simulation engine
│   ├── risk_metrics.py        # VaR (95%/99%), CVaR, Sharpe, Sortino, MDD
│   ├── sentiment_alpha.py     # NLP headline sentiment analysis & dynamic tilt
│   └── portfolio_advisor.py   # Main pipeline orchestrator
└── tests/
    └── test_risk_engine.py    # Pytest test suite (100% passing)
```

---

## 👤 Author

**Deepanshu Kushwaha**  
- Dual-degree: B.S. Data Science (IIT Madras) | B.Tech Electrical Engineering (BIET Jhansi)
- Aspire Institute Seed Fund Grantee (Harvard Business School faculty initiative)
- Reliance Foundation Undergraduate STEM Scholar
- GitHub: [@GodxDeepanshu](https://github.com/GodxDeepanshu) | [@CodeWithDeepanshuk](https://github.com/CodeWithDeepanshuk)
- Email: deepanshukushwahaiit1@gmail.com
