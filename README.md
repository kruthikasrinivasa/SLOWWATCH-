# SLOWWATCH
### An Explainable and Uncertainty-Aware Temporal AI Framework for Early Detection of Gradual Vegetation Degradation from Multi-Temporal Satellite Observations

<p align="center">
  <strong>Temporal AI | Remote Sensing | Machine Learning | Explainable AI | Environmental Monitoring</strong>
</p>

---

## Overview

**SLOWWATCH** is an explainable and uncertainty-aware temporal AI framework developed to identify potential early-warning signals of gradual vegetation degradation using multi-temporal satellite observations.

Vegetation degradation is often a gradual process that may be difficult to distinguish from seasonal variations, environmental fluctuations, and satellite observation noise. A temporary reduction in vegetation indices does not necessarily indicate persistent degradation.

SLOWWATCH addresses this challenge by analyzing vegetation trajectories over time, integrating environmental context, and evaluating predictive models alongside temporal risk, uncertainty, explainability, warning behaviour, and geographical transferability.

The framework combines Harmonized Landsat and Sentinel-2 (HLS) observations, Fmask-based quality screening, NDVI-derived temporal features, ERA5-Land environmental variables, and machine learning models.

Rather than focusing only on classification accuracy, SLOWWATCH investigates how vegetation warnings behave over time, how uncertain the estimated risks are, which features are associated with risk changes, and how predictive performance transfers across geographical regions.

## Research Objectives

- Analyze long-term vegetation dynamics using multi-temporal satellite observations.
- Identify potential early-warning signals from vegetation trajectory changes.
- Distinguish seasonal vegetation variations from abnormal temporal behaviour.
- Integrate temperature and precipitation as environmental context.
- Compare Random Forest and LSTM predictive models.
- Estimate continuous temporal vegetation risk and associated uncertainty.
- Interpret feature-risk relationships through statistical analysis.
- Evaluate feature contributions through ablation experiments.
- Assess warning behaviour against independent disturbance references.
- Investigate geographical transferability through bidirectional cross-region validation.

## Research Gap and Novelty

Existing vegetation-monitoring approaches frequently emphasize vegetation-change detection and predictive performance. However, identifying potential early-warning signals requires more than detecting changes or reporting classification metrics.

There is a need to examine temporal warning behaviour, uncertainty, interpretability, and geographical transferability together.

**SLOWWATCH addresses this gap through an integrated temporal early-warning framework.**

Its research contribution lies in combining:

1. Temporal vegetation trajectory modelling.
2. Environmental context integration.
3. Continuous risk and uncertainty assessment.
4. Feature-risk association analysis.
5. Feature-group ablation experiments.
6. Event-level warning and lead-time evaluation.
7. Bidirectional geographical transfer validation.

The emphasis is on evaluating the reliability and behaviour of vegetation-degradation warning signals rather than introducing an entirely new vegetation index or machine learning algorithm.

## Study Regions and Dataset

The experimental study covers three regions in South India over the period **2018–2025**.

| Region | State | HLS Granules | Valid Observations | Monthly Records |
|---|---|---:|---:|---:|
| Kodagu | Karnataka | 487 | — | 77 |
| Wayanad | Kerala | 1,464 | 508 | 88 |
| Nilgiris | Tamil Nadu | 1,267 | 584 | 93 |

Kodagu serves as the primary model development and evaluation region. Wayanad and Nilgiris are used for geographical transferability experiments.

### Data Sources

**1. Harmonized Landsat and Sentinel-2 (HLS)**

Satellite vegetation-index observations are used to construct multi-temporal vegetation records.

**2. Fmask v2.0**

Provides quality information for filtering observations affected by:

- Clouds and adjacent clouds.
- Cloud shadows.
- Cirrus.
- Snow and water.
- Aerosols.

**3. ERA5-Land**

Provides environmental context through:

- 2-m air temperature.
- Total precipitation.

**4. NASA FIRMS**

Fire detection observations are used to construct independent disturbance reference events for event-level warning evaluation.

FIRMS-derived events are treated as disturbance references, not field-verified vegetation degradation labels.

## System Architecture

The SLOWWATCH workflow consists of satellite data preprocessing, temporal feature engineering, environmental integration, predictive modelling, temporal risk estimation, uncertainty assessment, explainability, and evaluation.

```text
        HLS Satellite Observations
                   |
                   v
          Fmask v2.0 Quality
               Screening
                   |
                   v
          NDVI Extraction
                   |
                   v
        Monthly NDVI Aggregation
                   |
                   v
        Temporal Feature Engineering
                   |
          +--------+--------+
          |                 |
          v                 v
   Vegetation Trends    ERA5-Land
   and Anomalies        Temperature
                        Precipitation
          |                 |
          +--------+--------+
                   |
                   v
          15-Feature Dataset
                   |
                   v
          Predictive Modelling
             /         \
            v           v
      Random Forest    LSTM
            \           /
             \         /
                   |
                   v
         SLOWWATCH Temporal Core
                   |
          +--------+--------+
          |                 |
          v                 v
      Risk Estimation   Uncertainty
                         and Confidence
          |                 |
          +--------+--------+
                   |
                   v
          Feature-Risk Analysis
                   |
                   v
          Early-Warning Analysis
                   |
          +--------+--------+
          |                 |
          v                 v
     Event-Level       Cross-Region
     Evaluation        Validation
          |                 |
          +--------+--------+
                   |
                   v
          Research Evaluation
```

## Methodology

### 1. HLS Preprocessing and Quality Control

HLS vegetation-index observations are matched with their corresponding Fmask products using tile identifiers and acquisition details.

Fmask v2.0 is used to remove unreliable observations affected by clouds, shadows, snow, water, cirrus, and aerosols.

The vegetation-index observations are rescaled using the product scaling factor of 0.0001 and restricted to the NDVI range of [-1, 1].

Observations containing fewer than 100 good pixels are discarded.

For the Kodagu temporal pipeline, a 512 × 512-pixel central window is used.

The remaining quality-controlled observations are sorted chronologically and aggregated into monthly records.

### 2. Temporal Feature Engineering

SLOWWATCH constructs 15 features representing vegetation conditions, temporal behaviour, and environmental context.

Important vegetation and temporal features include:

- Monthly mean and median NDVI.
- Month-to-month NDVI change.
- Three-period moving average.
- Three-period moving standard deviation.
- Six-observation vegetation trajectory slope.
- Seasonal expected NDVI.
- Seasonal NDVI anomaly.
- Anomaly Z-score.

Observation counts, valid-pixel counts, and quality flags are also retained where available.

The six-observation slope is calculated using the most recent available monthly records, which may not necessarily represent six consecutive calendar months.

### 3. Environmental Context Integration

ERA5-Land temperature and precipitation observations are aligned with monthly vegetation records.

The environmental preprocessing includes:

- Temperature conversion from Kelvin to Celsius.
- Precipitation conversion from metres to millimetres.
- Monthly environmental aggregation.
- Temperature and precipitation change features.

Environmental variables provide contextual information for interpreting vegetation dynamics. Their inclusion does not establish that climatic conditions caused vegetation degradation.

### 4. Predictive Modelling

Two predictive models are evaluated using chronological training, validation, and testing splits.

**Random Forest (RF)**

Random Forest is selected as the primary predictive model because it can capture nonlinear relationships within the structured temporal and environmental feature representation.

**Long Short-Term Memory (LSTM)**

LSTM is implemented as a sequence-based deep learning comparison to investigate whether explicitly modelling temporal sequences improves prediction.

LSTM configuration:

- Sequence length: 6 observations.
- Input features: 15.
- Training sequences: 43.
- Validation sequences: 10.
- Testing sequences: 19.
- Early stopping: Epoch 88.
- Test period: January 2024 – December 2025.

The models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.

### 5. Baseline Warning Construction

A trajectory-based warning indicator is constructed using seasonal NDVI anomaly and vegetation trajectory slope.

A candidate trajectory warning is generated when:

- Seasonal NDVI anomaly is negative.
- Six-observation trajectory slope is negative.

A context-supported warning additionally considers precipitation below its historical median.

These indicators represent potential warning signals rather than confirmed vegetation degradation.

The Kodagu seasonal baseline uses the complete 2018–2025 observation period and is therefore not completely free from forecasting leakage.

### 6. SLOWWATCH Temporal Risk Estimation

The SLOWWATCH core calculates the following for each temporal observation:

- Continuous temporal-risk score.
- Risk category.
- Warning flag.
- Uncertainty score.
- Confidence score.

Implemented risk categories:

- LOW
- MINIMAL
- MODERATE

Risk represents the estimated level of concern associated with the observed vegetation trajectory. Uncertainty indicates how cautiously the risk estimate should be interpreted.

### 7. Uncertainty and Confidence Assessment

Uncertainty and confidence are evaluated alongside temporal-risk estimates.

This allows the framework to communicate not only the estimated risk but also the degree of uncertainty associated with the observation.

The analysis investigates how uncertainty varies across the monitoring period and how confidence complements the risk assessment.

### 8. Explainability and Feature-Risk Analysis

SLOWWATCH uses Pearson correlation analysis to examine relationships between selected vegetation and environmental features and the temporal-risk score.

The analysis considers:

- NDVI change.
- Precipitation.
- Vegetation trajectory slope.
- Mean NDVI.
- Anomaly Z-score.
- Temperature.

Correlation magnitude and direction help interpret the association between individual features and temporal risk.

**Important:** This is correlation-based feature-risk analysis, not SHAP-based attribution or causal explanation.

### 9. Feature Ablation Study

Ablation experiments evaluate the incremental contribution of vegetation, temporal, and environmental feature groups.

Four configurations are compared:

| Configuration | Feature Group |
|---|---|
| A | NDVI only |
| B | NDVI + Temporal |
| C | NDVI + Temporal + Environment |
| D | Full 15 features |

The purpose is to investigate how additional temporal and environmental information influences predictive performance.

### 10. Event-Level Evaluation

Event-level evaluation investigates warning generation in relation to FIRMS-derived disturbance reference events.

The evaluation considers:

- Reference-event detection.
- Number of generated warnings.
- Event recall.
- Mean warning lead time.
- Median warning lead time.
- Maximum warning lead time.
- False-warning rate.

Warning lead time is calculated as:

**Lead Time = Reference Event Date − Warning Date**

A positive lead time indicates that the warning occurred before the corresponding reference event.

These events are disturbance references and must not be interpreted as field-verified vegetation degradation ground truth.

### 11. Bidirectional Cross-Region Validation

Geographical transferability is evaluated between Wayanad and Nilgiris.

Two experiments are conducted:

- Train on Wayanad and test on Nilgiris.
- Train on Nilgiris and test on Wayanad.

This investigates how differences in geographical and environmental conditions influence predictive performance across regions.

## Experimental Results

### 1. Predictive Model Comparison

The following table presents the chronological test results.

| Metric | Random Forest | LSTM |
|---|---:|---:|
| Accuracy | **84.21%** | 68.42% |
| Precision | 86.67% | **90.00%** |
| Recall | **92.86%** | 64.29% |
| F1-score | **89.66%** | 75.00% |
| ROC-AUC | **84.29%** | 80.00% |
| PR-AUC | **92.94%** | 86.60% |

Random Forest achieved stronger accuracy, recall, F1-score, ROC-AUC, and PR-AUC than LSTM in the chronological test experiment.

LSTM achieved higher precision, indicating stronger precision among its positive predictions in this particular evaluation.

Random Forest was selected as the primary model based on its stronger overall performance, particularly in recall and F1-score.

### 2. Feature Ablation Results

| Configuration | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| A: NDVI only | 57.89% | 71.43% | 71.43% | 71.43% | 54.29% | 82.71% |
| B: NDVI + Temporal | 68.42% | 78.57% | 78.57% | 78.57% | 61.43% | 84.64% |
| C: NDVI + Temporal + Environment | 78.95% | 85.71% | 85.71% | 85.71% | 85.71% | 94.67% |
| D: Full 15 Features | 78.95% | 85.71% | 85.71% | 85.71% | 82.86% | 93.34% |

**Key observation:**

Adding temporal and environmental information improved predictive performance compared with NDVI-only features.

However, the full 15-feature configuration did not produce additional F1-score improvement over the 12-feature configuration.

This indicates that adding more features does not necessarily result in better predictive performance.

### 3. Temporal Risk and Uncertainty Results

For Kodagu, the SLOWWATCH core generated 77 temporal risk observations between January 2018 and December 2025.

| Measure | Result |
|---|---:|
| Temporal observations | 77 |
| LOW risk observations | 44 |
| MINIMAL risk observations | 19 |
| MODERATE risk observations | 14 |
| Warning periods | 2 |
| Mean temporal-risk score | 0.3584 |
| Mean uncertainty | 0.5454 |
| Mean confidence | 0.4546 |

The uncertainty values ranged from 0.0566 to 0.8293.

The results demonstrate that risk and uncertainty vary across the monitoring period rather than remaining consistently high.

### 4. Feature-Risk Association Results

| Feature | Correlation with Temporal Risk |
|---|---:|
| NDVI change | -0.5321 |
| Precipitation | -0.5248 |
| Slope | -0.3791 |
| Mean NDVI | -0.3080 |
| Anomaly Z-score | -0.1994 |
| Temperature | +0.1776 |

NDVI change and precipitation showed the strongest absolute correlations with temporal risk in the reported analysis.

These relationships are statistical associations and do not establish causality.

### 5. Event-Level Warning Evaluation

| Method | Events Detected | Mean Lead Time | Median Lead Time | Maximum Lead Time |
|---|---:|---:|---:|---:|
| SLOWWATCH | 10 | 0.50 months | 0 months | 2 months |
| Random Forest | 28 | 2.32 months | 3 months | 3 months |
| LSTM | 28 | 1.93 months | 2 months | 3 months |

The SLOWWATCH warning layer generated fewer events and shorter lead times than the underlying predictive models.

This highlights the importance of evaluating the behaviour of the warning layer separately from model-level classification performance.

### 6. Cross-Region Generalization Results

| Train → Test | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Wayanad → Nilgiris | 74.19% | 96.67% | 55.77% | 70.73% | 83.72% | 89.97% |
| Nilgiris → Wayanad | 79.55% | 65.96% | 93.94% | 77.50% | 92.07% | 89.47% |

The results show directional differences in geographical transferability.

Wayanad-to-Nilgiris transfer achieved higher precision but lower recall, whereas Nilgiris-to-Wayanad transfer achieved higher recall but lower precision.

This demonstrates that geographical differences can influence model behaviour and prediction errors.

### 7. Cross-Region Event-Level Evaluation

| Train → Test | Event Recall | False-Warning Rate |
|---|---:|---:|
| Wayanad → Nilgiris | 66.67% | 3.33% |
| Nilgiris → Wayanad | 96.15% | 38.30% |

These results further demonstrate directional differences in warning behaviour across geographical regions.

## Key Research Contributions

The primary contributions of SLOWWATCH include:

1. **Temporal Early-Warning Framework:** Formulates gradual vegetation monitoring as a temporal early-warning problem rather than relying only on isolated vegetation observations.

2. **Multi-Source Context Integration:** Combines satellite-derived NDVI trajectories with ERA5-Land temperature and precipitation information.

3. **Uncertainty-Aware Risk Assessment:** Generates continuous temporal-risk estimates with separate uncertainty and confidence information.

4. **Interpretable Feature-Risk Relationships:** Uses statistical association analysis to investigate the relationship between vegetation and environmental variables and temporal risk.

5. **Feature Ablation:** Evaluates the contribution of NDVI, temporal, and environmental feature groups.

6. **Event-Level Warning Evaluation:** Investigates disturbance-reference detection, warning generation, lead times, and false-warning behaviour.

7. **Bidirectional Geographic Validation:** Examines model transferability between Wayanad and Nilgiris in both geographical directions.

## Technology and Tools

| Category | Technologies |
|---|---|
| Programming | Python |
| Satellite observations | Harmonized Landsat and Sentinel-2 (HLS) |
| Quality assessment | Fmask v2.0 |
| Vegetation index | NDVI |
| Environmental data | ERA5-Land |
| Disturbance references | NASA FIRMS |
| Machine Learning | Random Forest |
| Deep Learning | Long Short-Term Memory (LSTM) |
| Statistical analysis | Pearson Correlation |
| Model evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| Data processing | Temporal aggregation and feature engineering |

## Limitations

The current research has several limitations:

- FIRMS-derived events represent disturbance references and are not field-verified vegetation degradation labels.
- Missing observations and unreadable Fmask files affect temporal continuity.
- The Kodagu temporal analysis uses a 512 × 512-pixel central spatial window, limiting the spatial extent of the primary study.
- The seasonal baseline uses the full 2018–2025 period and is not completely free from forecasting leakage.
- The study considers only three geographical regions, limiting broader generalizability.
- Pearson correlation identifies statistical associations rather than causal relationships.
- The SLOWWATCH warning layer generated fewer events and shorter lead times than the predictive models alone.

## Future Scope

Future research directions include:

- Independent ecological validation using field-based degradation observations.
- Extending the framework to additional geographical regions.
- Improving uncertainty estimation and interpretation.
- Exploring advanced temporal modelling approaches.
- Investigating geographically explicit early-warning systems.
- Improving temporal data completeness and spatial coverage.
- Evaluating long-term warning reliability across diverse environmental conditions.

## Conclusion

SLOWWATCH presents an explainable and uncertainty-aware temporal AI framework for investigating potential early-warning signals of gradual vegetation degradation using multi-temporal satellite observations.

By combining quality-controlled HLS observations, temporal NDVI features, ERA5-Land environmental context, predictive modelling, temporal risk estimation, uncertainty assessment, feature-risk analysis, event-level evaluation, and cross-region validation, the framework extends vegetation monitoring beyond conventional classification-based assessment.

The experiments demonstrate the value of temporal and environmental information over NDVI-only feature configurations and show that Random Forest achieved stronger overall chronological test performance than LSTM in the primary study.

Cross-region experiments also reveal directional differences in geographical transferability, highlighting the importance of evaluating the reliability and behaviour of early-warning systems across different environments.

---

## Project Information

**Project:** SLOWWATCH

**Project Type:** B.Tech Major Capstone / Research Project

**Research Domain:** Artificial Intelligence, Machine Learning, Remote Sensing and Environmental Monitoring

**Study Period:** 2018–2025

**Primary Study Region:** Kodagu, Karnataka, India

**GitHub Repository:** https://github.com/kruthikasrinivasa/SLOWWATCH-

## Authors

- Kruthika S
- Shalini S
- Thungashree I L

Department of Computer Science and Engineering (Data Science)  
Dayananda Sagar University, Bengaluru, India.

## Disclaimer

SLOWWATCH is an academic research framework developed for experimental vegetation monitoring and early-warning analysis. Its risk estimates and disturbance-reference evaluations should not be interpreted as definitive evidence of vegetation degradation. Independent ecological and field-based validation is necessary before operational deployment.

---

<p align="center">
  <strong>SLOWWATCH</strong><br>
  <i>Understanding Vegetation Change Through Temporal Intelligence</i>
</p>
