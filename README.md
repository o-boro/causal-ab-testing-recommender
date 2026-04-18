# E-Commerce A/B Test Causal Impact Incrementality of Recommendations

[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

## Executive Summary
Causal evaluation of e-commerce recommendation systems. Measuring true incremental revenue (iRPU) vs. cannibalization using A/B testing, CUPED variance reduction, and counterfactual modeling. 

This project explores the **true incremental impact (iRPU)** of a new Fashion Recommendation System. By building a custom Causal Simulation Engine and applying advanced statistical methods (CUPED, Bootstrap CI, Mediation Analysis), this research demonstrates how to isolate genuine revenue growth from demand redistribution and infrastructure latency.

## The Business Problem
A new ranking policy in the recommendation block showed significant increases in engagement. However, the business needed to answer:
1. **Incrementality:** Is the system generating *new* money, or simply cannibalizing Organic Search?
2. **Infrastructure Tax:** How much does the algorithm's inference latency cost us in lost conversions?
3. **Supply Chain:** Does the new policy lead to out-of-stock cancellations?

##  Methodology & Experiment Design
To establish a verifiable "Ground Truth", I developed a Python-based **Causal Data Generator** simulating 150,000 users. This allowed me to benchmark standard A/B test estimations against actual counterfactual data.

* **Split:** Randomized 50/50 A/B Split (User-level).
* **Metrics:** ARPU (Average Revenue Per User), iRPU (Incremental RPU), Session Depth, Latency-adjusted CTR.
* **Techniques Used:** Welch's T-test, Bootstrap Confidence Intervals, CUPED (Variance Reduction), Causal Deep Dives.

## Key Findings & Causal Insights

### 1. The "Illusion of Growth" (A/B Bias)
* **Estimated Uplift (Standard A/B):** +3.52%
* **True Causal Uplift (Ground Truth):** +2.72%
* **Insight:** Standard A/B testing overestimated the revenue impact by **~29%**. The gap was primarily driven by **demand redistribution**—users shifted their purchases from organic search to the new recommendation block, creating a mirage of hyper-growth.

### 2. Variance Reduction via CUPED
Revenue metrics naturally suffer from heavy-tail distributions. By using `pre_experiment_revenue` as a covariate, **CUPED reduced metric variance by X%** (insert actual %). This significantly narrowed the confidence intervals, allowing for faster decision-making without losing statistical power.

### 3. The Cost of Latency
System performance is a hidden business tax. The analysis proved a direct causal link: server responses exceeding 400ms drastically degraded the Click-Through Rate, offsetting the algorithmic benefits of the new ranking policy.

### 4. Stock-Awareness & Logistics
The new policy successfully penalized low-stock items (`stock < 5`). This behavioral shift protects the supply chain from post-purchase cancellations, improving long-term Customer LTV.

## Business Recommendations
1. **Proceed with Rollout, but Adjust Forecasts:** The algorithm is genuinely incremental, but financial forecasts must be adjusted down by 30% to account for search cannibalization.
2. **Implement SLA for ML Inference:** Engineering must strictly cap the ranking algorithm's latency at <200ms to prevent CTR decay.
3. **Adopt CUPED company-wide:** Transitioning from standard T-tests to CUPED for revenue metrics will save weeks of experimental runtime.

## Quick Start
```bash
git clone [https://github.com/your-username/fashion-recommender-causal-ab.git](https://github.com/o-boro/fashion-recommender-causal-ab.git)
cd fashion-recommender-causal-ab
pip install -r requirements.txt
