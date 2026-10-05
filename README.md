# SLOWWATCH 🌿

## An Explainable, Uncertainty-Aware Temporal AI Framework for Early Detection of Gradual Vegetation Degradation from Multi-Temporal Satellite Observations

> **Research Project | AI/ML | Remote Sensing | Environmental Informatics | Temporal Modelling**

SLOWWATCH is a research-oriented temporal AI framework designed to identify **early warning signals of gradual vegetation degradation** from multi-temporal satellite observations.

Unlike conventional vegetation monitoring approaches that primarily focus on detecting vegetation change after it has occurred, SLOWWATCH models vegetation as a **temporal trajectory** and combines vegetation dynamics, environmental context, machine learning, uncertainty estimation, explainability, event-level evaluation, and cross-region validation.

The framework is designed to answer not only **whether vegetation condition is changing**, but also:

- How is vegetation changing over time?
- Is the observed trajectory consistent with degradation?
- What environmental conditions accompany the change?
- How confident is the model?
- Which features contribute to the risk signal?
- Can the model generalize to another geographic region?
- How early can potential disturbance events be identified?

# 🎯 Research Objective

The primary objective of SLOWWATCH is:

> **To develop and evaluate an explainable, uncertainty-aware temporal AI framework for identifying early warning signals of gradual vegetation degradation from multi-temporal satellite observations.**

### Specific Objectives

- Extract vegetation information from multi-temporal satellite observations.
- Construct temporal vegetation trajectories.
- Derive anomaly, trend, and change-based features.
- Integrate environmental context from ERA5-Land.
- Compare machine-learning and temporal deep-learning models.
- Generate temporal vegetation risk scores.
- Quantify prediction uncertainty.
- Provide feature-level explanations.
- Evaluate warning behaviour at the event level.
- Test spatial cross-region generalization.
- Evaluate the contribution of different feature groups through ablation experiments.

# 🔬 Research Motivation

Vegetation degradation is often a **gradual process** rather than an instantaneous event.

A monitoring system that identifies degradation only after a major decline has occurred may provide limited opportunity for early intervention.

SLOWWATCH therefore approaches vegetation monitoring as a **temporal early-warning problem**.

The framework combines:

**Satellite Observations → Temporal Vegetation Dynamics → Environmental Context → ML Prediction → Risk → Uncertainty → Explainability → Early Warning → Spatial Validation**

---

# 🎯 Research Problem

Existing vegetation-monitoring approaches can detect or characterize vegetation change using remote sensing data, machine learning, and temporal indicators.

However, there is a need for an integrated framework that treats gradual vegetation degradation as a **temporal early-warning problem** while simultaneously considering:

1. Predictive performance
2. Temporal trajectory behaviour
3. Environmental context
4. Prediction uncertainty
5. Explainability
6. Event-level warning behaviour
7. Spatial transferability

SLOWWATCH is designed around this integrated evaluation perspective.


# 🏗️ SLOWWATCH Framework

The overall SLOWWATCH framework follows a multi-stage temporal AI pipeline:

```text
                 MULTI-TEMPORAL SATELLITE DATA
                              │
                              ▼
                     HLS DATA ACQUISITION
                              │
                              ▼
                       FMASK QA FILTERING
                              │
                              ▼
                         NDVI EXTRACTION
                              │
                              ▼
                    TEMPORAL AGGREGATION
                              │
                              ▼
                 TEMPORAL FEATURE ENGINEERING
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
             NDVI         Temporal       Anomaly /
             Change        Slope          Z-score
                │             │             │
                └─────────────┼─────────────┘
                              │
                              ▼
                   ENVIRONMENTAL CONTEXT
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
               Temperature        Precipitation
                ERA5-Land           ERA5-Land
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     MULTISOURCE DATASET
                              │
                              ▼
                    ┌───────────────────┐
                    │ PREDICTIVE MODELS │
                    │                   │
                    │ Random Forest     │
                    │ LSTM              │
                    └─────────┬─────────┘
                              │
                              ▼
                      SLOWWATCH RISK
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
            UNCERTAINTY      XAI      EARLY WARNING
                 │            │            │
                 └────────────┼────────────┘
                              ▼
                       RESEARCH EVALUATION
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
         Ablation        Event-Level       Cross-Region
          Study          Evaluation        Validation


# Framework Components
1. Satellite Data Acquisition — Collect multi-temporal HLS observations.
2. Quality Filtering — Apply Fmask-based quality filtering.
3. Vegetation Representation — Extract and aggregate NDVI observations.
4. Temporal Feature Engineering — Generate vegetation change, trend, rolling statistics, and anomaly features.
5. Environmental Context — Integrate temperature and precipitation from ERA5-Land.
6. Predictive Modelling — Compare Random Forest and LSTM models.
7. Risk Estimation — Generate the SLOWWATCH temporal risk trajectory.
8. Uncertainty Analysis — Quantify uncertainty and confidence associated with the risk signal.
9. Explainability — Analyse feature-risk relationships.
10. Early Warning — Identify potential warning periods.
11. Research Evaluation — Perform ablation, event-level, and cross-region evaluation.






