# 🇰🇪 Kenya Economic Pulse

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://kenyaeconomicpulse.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.3.0-blue.svg)](https://github.com/kaks2679/project)

> **End-to-end data science portfolio** — 10-page interactive dashboard exploring Kenya's economy using real data + machine learning.

**Author:** [Stephen Muema](https://muemastephenportfolio.netlify.app/) | Data Scientist & ML Engineer | Nairobi, Kenya 🇰🇪

---

## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kenyaeconomicpulse.streamlit.app)

**👉 [https://kenyaeconomicpulse.streamlit.app](https://kenyaeconomicpulse.streamlit.app)**

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Overview** | Kenya at a glance — KPIs, GDP/poverty trends, county snapshot |
| 📊 **Economic Indicators** | 20+ macro indicators + Holt-Winters forecasts |
| 🗺️ **County Inequality Map** | 47 counties, KMeans clustering, interactive maps |
| 💳 **M-PESA Impact Predictor** | GBM regression (R²=0.904) proving mobile money reduces poverty |
| 🎓 **Youth Unemployment** | ML forecaster + scenario simulator (GDP/FDI/Education sliders) |
| 🔮 **Economic Forecaster** | Holt-Winters + ARIMA(2,1,2) with confidence intervals |
| ⚖️ **County Comparison** | Side-by-side radar + bar charts for any 2 counties |
| 🏛️ **Policy Simulator** | 5-lever what-if with waterfall impact chart |
| 🔍 **Anomaly Detection** | Isolation Forest detecting 2008 GFC, COVID, etc. |
| 💬 **NL Query Engine** | Ask plain-English questions → instant chart answers |

---

## 🤖 Machine Learning Models

| Model | Task | Performance |
|-------|------|-------------|
| Gradient Boosting | Poverty prediction | **R² = 0.904** |
| Random Forest | Poverty prediction | R² = 0.882 |
| Ridge Regression | Poverty prediction | R² = 0.871 |
| KMeans (k=5) | County segmentation | Silhouette > 0.58 |
| Holt-Winters | Economic forecasting | MAE < 0.8pp |
| ARIMA(2,1,2) | Economic forecasting | 95% CI |
| Isolation Forest | Anomaly detection | Contamination=10% |

---

## 🔑 Key Findings

| Finding | Insight |
|---------|---------|
| 💳 **M-Pesa Effect** | Financial inclusion rose 26% → 85% (2007-2023). Poverty fell 46.8% → 33.5%. ML confirms mobile money is #1 poverty predictor |
| 🗺️ **County Gap** | 69 percentage point poverty gap — Wajir (82.4%) vs Kiambu (13.2%) |
| 🎓 **Youth Crisis** | 61.5% youth unemployment — **4.5× global average** (13.6%) |
| 📈 **Growth Resilient** | Kenya averaged ~5.2% GDP growth 2000-2023 despite COVID, elections, drought |
| ⚡ **Power Divide** | Electricity access: 10.8% (Turkana) to 92.4% (Nairobi) |
| 💰 **Diaspora Power** | Remittances hit $4.2B (2023) — now larger than FDI inflows |

---

## 📡 Data Sources

| Source | Data |
|--------|------|
| [World Bank Open Data](https://data.worldbank.org/country/KE) | GDP, inflation, poverty, remittances (2000-2023) |
| [KNBS 2019 Census](https://www.knbs.or.ke/) | All 47 counties, 12 socioeconomic features |
| [Central Bank of Kenya](https://www.centralbank.go.ke/) | M-Pesa users, mobile money transactions (2007-2023) |
| [ILO ILOSTAT](https://ilostat.ilo.org/data/) | Youth unemployment (2005-2023) |
| [FinAccess Survey](https://www.knbs.or.ke/finaccess-household-survey/) | Financial inclusion metrics |

---

## 🛠️ Tech Stack
Frontend: Streamlit
ML Models: Scikit-learn (GBM, RF, KMeans, Isolation Forest)
Forecasting: Statsmodels (Holt-Winters, ARIMA)
Visuals: Plotly, Folium
Data: Pandas, NumPy, wbgapi
Reports: ReportLab (PDF generation)


---

## ⚡ Run Locally

```bash
# Clone
git clone https://github.com/kaks2679/project.git
cd project

# Install
pip install -r requirements.txt

# Run
streamlit run app.py

👤 Author
Stephen Muema | Data Scientist & ML Engineer

https://img.shields.io/badge/Portfolio-muemastephenportfolio.netlify.app-0A66C2?style=flat&logo=netlify&logoColor=white
https://img.shields.io/badge/GitHub-kaks2679-181717?style=flat&logo=github&logoColor=white
https://img.shields.io/badge/Email-stephenmuema@proton.me-EA4335?style=flat&logo=protonmail&logoColor=white

Nairobi, Kenya 🇰🇪
