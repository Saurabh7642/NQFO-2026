# NQFO-2026 (IIT GUWAHATI): Volatility Surface Reconstruction

## 1. Project Objective

The primary goal of this project is to reconstruct a complete and financially consistent Implied Volatility (IV) surface by predicting missing data points within a synthetic NIFTY50 dataset.

**The Task:**
- **Imputation:** Approximately 30–45% of the IV values are missing because certain options do not trade on a given day.
- **Precision:** Use known IV values to predict missing ones with maximum mathematical accuracy.
- **Consistency:** Ensure the final surface adheres to the laws of quantitative finance and remains "arbitrage-free".

## 2. Methodology: Global-Local Hybrid

To achieve high-precision results, this pipeline utilizes a two-tier modeling approach:

- **Global Regime Analysis (XGBoost):** An XGBoost regressor captures the broad "terrain" of the surface by learning the relationship between market regimes, time-to-maturity ($\tau$), and moneyness.
- **Local Calibration (Cubic Splines):** Known IV values in the test set are treated as anchor points. We apply Natural Cubic Splines to these anchors to interpolate the "smile" of each option chain with high fidelity.

## 3. Causal Feature Engineering

To capture the sequential evolution of the price signal, the pipeline employs strictly causal engineering:

- **Log-Moneyness Transformation:** Normalizing strikes relative to the spot price: $\xi = \ln(K/S)$.
- **Cyclical Temporal Encoding:** Using trigonometric shift operations to incorporate adjacent temporal information and capture oscillatory market behavior.
- **Interaction Ratios:** Modeling dependencies between spectral bands (Trend vs. Noise) using numerically stable ratio features.

## 4. Financial Integrity: Arbitrage-Free Constraints

A valid market surface must be economically sound. Our pipeline enforces the following constraints:

- **No-Butterfly Arbitrage:** By using $C^2$ continuous splines, we ensure the "smile" is convex. This prevents negative probability density:
  $$\frac{\partial^2 C}{\partial K^2} > 0$$

- **No-Calendar Spread Arbitrage:** We monitor Total Implied Variance ($w = \sigma^2 \tau$) and ensure it is non-decreasing over time. This prevents mathematically impossible variance predictions:
  $$\frac{\partial w}{\partial \tau} \ge 0$$

## 5. Performance & Results

The model was validated using a walk-forward, time-based split to ensure robustness against market shifts:

- **Time-Based Validation RMSE:** `0.691849`
- **Calibration:** A final "pinning" logic restores anchor points to their original values to ensure perfect alignment with known market data.

## 6. Repository Structure
- **solution.py** – Full runnable Python solution that reproduces the submission  
- **requirements.txt** – Dependencies (`pandas`, `numpy`, `xgboost`, `scipy`)  
- **methodology.pdf** – Comprehensive 2-page write-up

### 7.Execution  

```bash
pip install -r requirements.txt
python solution.py
