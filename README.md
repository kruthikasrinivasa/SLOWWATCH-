# SLOWWATCH
### An Explainable and Uncertainty-Aware Temporal AI Framework for Early Detection of Gradual Vegetation Degradation from Multi-Temporal Satellite Observations

<p align="center">
  <strong>Temporal AI • Remote Sensing • Machine Learning • Environmental Intelligence • Explainable AI</strong>
</p>

---

## 1. Overview

**SLOWWATCH** is an AI-driven framework designed to identify early signs of gradual vegetation degradation by analyzing multi-temporal satellite observations and environmental conditions.

Conventional vegetation monitoring approaches often rely on individual satellite observations or threshold-based vegetation indices. However, gradual vegetation degradation is a temporal process that may not be evident from a single observation.

SLOWWATCH addresses this challenge by analyzing vegetation trajectories over time, identifying abnormal changes in vegetation patterns, incorporating environmental context, and estimating degradation risk with uncertainty awareness.

The framework combines:

- Multi-temporal satellite imagery and vegetation index analysis.
- Quality-filtered NDVI observations.
- Temporal feature engineering and anomaly detection.
- Environmental context from temperature and precipitation.
- Machine learning and sequence-based prediction models.
- Temporal risk estimation and uncertainty analysis.
- Explainable AI for interpreting model predictions.
- Disturbance-event evaluation and cross-region validation.

The primary objective is to move beyond detecting existing vegetation changes towards identifying potential degradation signals early enough to support environmental monitoring and decision-making.

## 2. Problem Statement

Vegetation degradation is often a gradual process influenced by multiple environmental and temporal factors.

Traditional monitoring methods face several challenges:

- Individual satellite observations may fail to capture long-term degradation patterns.
- Cloud cover and poor-quality satellite observations can affect vegetation measurements.
- Seasonal variations can be mistaken for actual vegetation degradation.
- Environmental conditions such as temperature and precipitation influence vegetation dynamics.
- Prediction models may produce risk estimates without communicating their uncertainty.
- Black-box predictions provide limited insight into the factors contributing to degradation warnings.

SLOWWATCH addresses these challenges through a temporal, multi-source and explainable AI approach.

## 3. Project Objectives

1. Monitor vegetation dynamics using multi-temporal satellite observations.
2. Identify abnormal NDVI trajectories and gradual vegetation decline.
3. Extract temporal and environmental features to improve degradation prediction.
4. Compare machine learning and sequence-based deep learning approaches.
5. Estimate temporal vegetation degradation risk.
6. Incorporate uncertainty analysis to communicate prediction reliability.
7. Provide feature-level explanations for model predictions.
8. Evaluate predictions against independently constructed disturbance-event references.
9. Examine model performance across multiple geographical regions.
10. Generate experimental future vegetation forecasts and potential degradation indicators.

## 4. Study Regions and Dataset

The framework was developed and evaluated using three geographically distinct regions in South India.

| Study region | State | HLS granules | Monthly observations |
|---|---|---:|---:|
| Kodagu | Karnataka | 487 | 77 |
| Wayanad | Kerala | 1,464 | 88 |
| Nilgiris | Tamil Nadu | 1,267 | 93 |

**Observation period:** 2018–2025

### Data Sources

**1. Harmonized Landsat Sentinel-2 (HLS)**

HLS observations provide the satellite-derived vegetation information used for temporal monitoring.

- HLS-VI L30 observations.
- Fmask v2 for quality filtering.
- NDVI computation and monthly aggregation.
- Multi-temporal vegetation trajectory construction.

**2. ERA5-Land**

ERA5-Land reanalysis data provides environmental context for understanding vegetation dynamics.

Environmental variables include:

- Temperature.
- Precipitation.

**3. NASA FIRMS**

Fire Information for Resource Management System (FIRMS) data is used to construct reference disturbance events for event-level evaluation.

These events provide an independent disturbance reference rather than fully field-validated vegetation degradation labels.

## 5. System Architecture and Workflow

The SLOWWATCH framework follows a multi-stage processing and prediction pipeline.

```text
        HLS Satellite Observations
                   |
                   v
        Fmask v2 Quality Filtering
                   |
                   v
          NDVI Computation
                   |
                   v
       Monthly Temporal Aggregation
                   |
                   v
       Temporal Feature Extraction
                   |
          +--------+--------+
          |                 |
          v                 v
    NDVI Trajectories   ERA5-Land Data
          |             Temperature &
          |             Precipitation
          +--------+--------+
                   |
                   v
       Multi-Source Feature Dataset
                   |
                   v
          Model Development
                   |
            +------+------+
            |             |
            v             v
       Random Forest     LSTM
            |             |
            +------+------+
                   |
                   v
         Temporal Risk Estimation
                   |
          +--------+--------+
          |                 |
          v                 v
    Uncertainty Analysis  Explainable AI
          |                 |
          +--------+--------+
                   |
                   v
          Early Warning Analysis
                   |
                   v
       Event-Level Evaluation
                   |
                   v
        Cross-Region Evaluation
                   |
                   v
       Future NDVI Forecasting
                   |
                   v
       Vegetation Monitoring Insights
```

## 6. Methodology

### 6.1 Satellite Data Preprocessing

HLS satellite observations are processed using Fmask v2 quality information to filter unreliable observations.

NDVI is calculated to represent vegetation greenness:

NDVI = (NIR - Red) / (NIR + Red)

The quality-filtered observations are aggregated into monthly NDVI values to construct consistent temporal vegetation records.

### 6.2 Temporal Feature Engineering

Instead of relying only on raw NDVI values, SLOWWATCH extracts temporal characteristics that capture changes in vegetation behaviour.

The framework uses 15 temporal and environmental features.

Important temporal features include:

- NDVI change: Measures changes between successive observations.
- Rolling statistics: Capture short-term vegetation trends and fluctuations.
- Six-observation slope: Represents the direction and rate of vegetation change over a temporal window.
- Seasonal expected NDVI: Represents expected vegetation behaviour based on seasonal patterns.
- Seasonal anomaly: Measures deviations from expected seasonal vegetation conditions.
- Z-score: Represents the relative magnitude of vegetation anomalies.

These features help distinguish persistent vegetation decline from normal seasonal variations.

### 6.3 Environmental Context Integration

Temperature and precipitation observations from ERA5-Land are incorporated into the temporal feature dataset.

This provides additional environmental information for interpreting vegetation changes and understanding conditions associated with potential degradation.

### 6.4 Model Development

Two prediction approaches are investigated.

**Random Forest**

Random Forest is used as the primary machine learning model.

It learns relationships between temporal vegetation characteristics, environmental conditions and disturbance-related reference labels.

**Long Short-Term Memory (LSTM)**

LSTM is implemented as a comparative deep learning approach to capture sequential dependencies in vegetation observations.

A six-observation temporal sequence is used for sequence-based modelling.

Both approaches are evaluated using chronological train, validation and test splits to preserve temporal ordering.

### 6.5 Temporal Risk Estimation

The framework uses model predictions to estimate vegetation degradation risk over time.

Temporal risk analysis helps identify potentially concerning vegetation patterns instead of treating every abnormal NDVI observation as an independent event.

The resulting warnings support early identification of regions showing possible vegetation decline.

### 6.6 Uncertainty Analysis

SLOWWATCH incorporates uncertainty analysis to communicate the reliability of model predictions.

The objective is to distinguish between predictions with relatively higher confidence and those that require additional observation or investigation.

Risk estimates and uncertainty information are considered together to support more responsible interpretation of early warnings.

### 6.7 Explainable AI

Explainability is incorporated to understand which input features contribute to model predictions.

Feature-level explanations help interpret the influence of:

- Vegetation changes.
- Temporal trends.
- Seasonal anomalies.
- Environmental variables.

This makes the framework more interpretable than relying exclusively on prediction scores.

### 6.8 Ablation Analysis

Ablation experiments examine the contribution of different feature groups to predictive performance.

The analysis investigates whether combining temporal and environmental information improves predictions compared with using a more limited NDVI-based feature representation.

### 6.9 Event-Level and Cross-Region Evaluation

**Event-level evaluation:** Model warnings are examined against reference disturbance events constructed using FIRMS data.

**Cross-region evaluation:** The framework is examined across Kodagu, Wayanad and Nilgiris to investigate its behaviour under different geographical and environmental conditions.

These evaluations provide additional evidence beyond conventional classification metrics.

## 7. Experimental Results

### Model Comparison

The following are the reported model evaluation results.

| Evaluation metric | Random Forest | LSTM |
|---|---:|---:|
| Accuracy | 84.21% | 68.42% |
| F1-score | 89.66% | 75.00% |
| ROC-AUC | 84.29% | — |
| PR-AUC | 92.94% | — |

Random Forest demonstrated stronger performance than LSTM in the reported evaluation.

The results indicate that the engineered temporal and environmental features provide useful predictive information without necessarily requiring a more complex sequential deep learning architecture.

*Note: The LSTM ROC-AUC and PR-AUC values are not reported here because corresponding verified values are unavailable.*

### Ablation Analysis

The reported ablation experiment showed an improvement in accuracy from 57.89% to 78.95% when the expanded feature configuration was used.

This supports the importance of incorporating additional temporal and environmental information rather than relying on a limited vegetation-index representation alone.

### Kodagu Risk Analysis

The reported Kodagu analysis included:

- 77 monthly observations.
- Mean estimated risk: 0.3584.
- Two generated warnings.
- Mean uncertainty: 0.5454.
- Mean confidence: 0.4546.

These values describe the reported experimental risk analysis and should not be interpreted as calibrated real-world degradation probabilities.

## 8. Future Vegetation Forecasting

SLOWWATCH also includes an experimental future NDVI forecasting component to examine potential vegetation trajectories beyond the historical observation period.

The forecasting outputs include:

- Predicted NDVI values.
- Approximate lower and upper uncertainty bounds.
- Historical 12-month mean NDVI.
- Difference between predicted NDVI and historical baseline.
- Potential NDVI-decline indicators.
- Forecast horizon and forecast origin.

Experimental forecasts were generated for January–June 2026 across Kodagu, Wayanad and Nilgiris.

The forecast outputs help identify periods in which predicted vegetation conditions deviate from historical behaviour.

**Important:** These future NDVI forecasts and associated risk indicators are experimental monitoring signals. They are not independently validated future degradation events or calibrated probabilities of vegetation loss.

## 9. Key Contributions

The main contributions of SLOWWATCH are:

1. **Temporal vegetation monitoring:** Focuses on vegetation trajectories rather than isolated satellite observations.

2. **Multi-source environmental analysis:** Combines satellite-derived vegetation information with temperature and precipitation context.

3. **Temporal anomaly characterization:** Uses seasonal anomalies, rolling statistics and trend-based features to identify abnormal vegetation behaviour.

4. **Comparative predictive modelling:** Evaluates Random Forest and LSTM approaches for vegetation degradation prediction.

5. **Uncertainty-aware risk interpretation:** Incorporates uncertainty information alongside temporal risk estimates.

6. **Explainable predictions:** Examines feature contributions to make model outputs more interpretable.

7. **Disturbance-based evaluation:** Uses FIRMS-derived reference events for additional event-level assessment.

8. **Cross-region investigation:** Examines the framework across three distinct South Indian geographical regions.

9. **Future vegetation forecasting:** Extends the monitoring workflow towards experimental future NDVI trajectory analysis.

## 10. Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| Satellite data | HLS-VI L30 |
| Quality assessment | Fmask v2 |
| Environmental data | ERA5-Land |
| Disturbance references | NASA FIRMS |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Model evaluation | Scikit-learn metrics |
| Development | Jupyter Notebook, VS Code, Git, GitHub |

## 11. Limitations

The current implementation has the following limitations:

- FIRMS-derived disturbance events are not equivalent to field-validated vegetation degradation ground truth.
- Missing satellite observations and unreadable quality-mask files can affect temporal data completeness.
- The Kodagu spatial study window is limited.
- Seasonal baselines are derived from the available 2018–2025 observation period and are not completely leakage-free.
- Evaluation is limited to three geographical regions.
- Future NDVI forecasts are experimental and require further independent validation.
- The reported risk indicators should not be interpreted as calibrated probabilities of actual vegetation degradation.

## 12. Future Scope

Potential directions for further development include:

- Extending the framework to additional geographical regions.
- Incorporating longer historical satellite observation records.
- Integrating additional environmental and land-cover variables.
- Improving uncertainty calibration and probabilistic risk estimation.
- Validating degradation warnings using field observations and independent ground-truth datasets.
- Exploring advanced temporal architectures and forecasting approaches.
- Developing operational monitoring dashboards with spatial risk visualization.
- Improving early-warning evaluation through longer-term independent event observations.

## 13. Research Significance

SLOWWATCH investigates how temporal vegetation behaviour, environmental context and interpretable machine learning can be combined to support early identification of potential vegetation degradation.

Rather than treating vegetation monitoring as a simple NDVI thresholding problem, the framework studies changes in vegetation trajectories over time and incorporates uncertainty and explainability into the prediction process.

The work contributes towards more informative and interpretable approaches for environmental monitoring using multi-temporal Earth observation data.

## 14. Repository

**GitHub:** [SLOWWATCH – Repository](https://github.com/kruthikasrinivasa/SLOWWATCH-)

**Author:** Kruthika S

**Project Type:** B.Tech Major Capstone / Research Project

**Domain:** Artificial Intelligence, Machine Learning, Remote Sensing and Environmental Monitoring

---

## 15. Disclaimer

SLOWWATCH is an academic research and experimental monitoring framework. Its predictions and warning indicators are intended to support further investigation and environmental assessment. They should not be treated as definitive evidence of vegetation degradation or as a substitute for field-based validation.

---

<p align="center">
  <strong>SLOWWATCH</strong><br>
  Temporal Intelligence for Early Vegetation Degradation Monitoring
</p>
