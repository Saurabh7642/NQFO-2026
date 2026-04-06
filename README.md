# NATIONAL QUANT FINANCE OLYMPIAD  
## Project:Volatility Surface Reconstruction  

### 1. Problem Statement  
Financial market data is characterized by a complex superposition of long-term trends, cyclic oscillations, and high-frequency microstructure noise. The objective of this study is to develop a robust, causal regression pipeline that predicts a target variable $y$ by deconstructing the primary signal $f_{0}$.  

In the context of derivatives markets, this involves the imputation of the Implied Volatility (IV) Surface. The challenge lies in accurately "filling the holes" in sparse market data while maintaining a continuous and economically consistent manifold that adheres to the fundamental laws of quantitative finance.  

### 2. Methodology: Hybrid Dual-Domain Architecture  
The proposed methodology employs a dual-layered approach to separate structural drifts from microstructure artifacts:  

#### I. Global Phase: Regime-Agnostic Regression (XGBoost)  
- **Objective:** Establish the baseline surface magnitude and term structure.  
- **Logic:** Utilizing Gradient Boosted Decision Trees (XGBoost) to map global coordinates—Time-to-Maturity ($\tau$) and Log-Moneyness ($\xi$)—against a dynamic Regime Proxy.  
- **Regime Analysis:** Applying a windowed median filter to isolate slow-moving structural supply/demand drifts from high-frequency noise.  

#### II. Local Phase: High-Fidelity Calibration (Cubic Splines)  
- **Objective:** Reconstruct the local "Volatility Smile" with mathematical precision.  
- **Logic:** Implementing Natural Cubic Splines to ensure $C^2$ continuity. This transforms discrete observations into a continuous, frequency-stable manifold.  

### 3. Causal Feature Engineering  
To capture the sequential evolution of the price signal, the pipeline employs strictly causal engineering:  

- **Log-Moneyness Transformation:** Normalizing strikes relative to the spot price: $\xi = \ln(K/S)$.  
- **Cyclical Temporal Encoding:** Using trigonometric shift operations to incorporate adjacent temporal information and capture oscillatory market behavior.  
- **Interaction Ratios:** Modeling dependencies between spectral bands (Trend vs. Noise) using numerically stable ratio features.  

### 4. Static Arbitrage Enforcement  
A valid financial manifold must be free of static arbitrage. Our pipeline enforces the following constraints:  

#### Horizontal Convexity (No-Butterfly Arbitrage)  
By utilizing Natural Cubic Splines, we ensure that the option price is a convex function of the strike. This guarantees a positive State Price Density:  

$$
\frac{\partial^2 C}{\partial K^2} > 0
$$  

#### Vertical Monotonicity (No-Calendar Spread Arbitrage)  
The pipeline monitors the Total Implied Variance ($w = \sigma^2 \tau$). We apply a cumulative maximum operation to ensure that total variance is non-decreasing across the term structure, preventing outlier distortion:  

$$
\frac{\partial w}{\partial \tau} \ge 0
$$  

### 5. Results and Validation  
The model's performance is assessed using Root Mean Squared Error (RMSE) for magnitude accuracy and Spearman's Rank Correlation for directional alignment.  

- **Time-Based Validation RMSE:** `0.691849`  
- **Surface Stability:** High spectral interaction stability confirms the model's ability to effectively approximate signals even in zero-inflated market regimes.  

### 6. Repository Structure & Usage  
- **solution.py** – Full runnable Python solution that reproduces the submission  
- **requirements.txt** – Dependencies (`pandas`, `numpy`, `xgboost`, `scipy`)  
- **methodology.pdf** – Comprehensive 2-page write-up

### 7.Execution  

```bash
pip install -r requirements.txt
python solution.py
