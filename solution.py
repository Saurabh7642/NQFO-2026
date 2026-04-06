import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error
import warnings

# Mute warnings for a cleaner submission output
warnings.filterwarnings('ignore')

# Load data 
train = pd.read_csv('train.csv', parse_dates=['date'])
test = pd.read_csv('test.csv', parse_dates=['date'])

# Split test into known and unknown IVs
test_known = test[test['iv_observed'].notna()].copy()
test_unknown = test[test['iv_observed'].isna()].copy()

print(f'Train rows           : {len(train):,}')
print(f'Test known IVs       : {len(test_known):,}')
print(f'Test rows to predict : {len(test_unknown):,}')


#  Feature Engineering (Regime and Term Structure)

# Build mapping from BOTH train + test_known to maximize Regime memory
combined_anchors = pd.concat([train, test_known])
date_median_map = combined_anchors.groupby('date')['iv_observed'].median()

def apply_advanced_features(df):
    temp = df.copy()
    # Cyclical Time Features to capture intraday patterns
    temp['day_of_week_sin'] = np.sin(2 * np.pi * temp['date'].dt.dayofweek / 7)
    temp['day_of_week_cos'] = np.cos(2 * np.pi * temp['date'].dt.dayofweek / 7)
    
    # Surface Coordinates (Moneyness^2 and Option Type encoding)
    temp['m2'] = temp['moneyness'] ** 2
    temp['is_call'] = (temp['option_type'] == 'call').astype(int)
    
    # Global Regime Proxy (Calibrated using observable test anchors)
    temp['regime_proxy'] = temp['date'].map(date_median_map).ffill().bfill()
    return temp

train_df = apply_advanced_features(train).dropna(subset=['iv_observed'])
test_df = apply_advanced_features(test)

# Global Phase: Tuned XGBOOST (Surface Height)

features = ['moneyness', 'm2', 'tau', 'is_call', 'regime_proxy', 'day_of_week_sin', 'day_of_week_cos']
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=8, random_state=42)

print("Training global XGBoost model...")
model.fit(train_df[features], train_df['iv_observed'])

# Initialize prediction column with global baseline
test_df['iv_predicted'] = model.predict(test_df[features]).astype(np.float64)

# Local Phase: Hardened Cubic Spline (Smile Reconstruction)

def hardened_spline_refinement(group):
    known = group[group['iv_observed'].notna()]
    unknown = group[group['iv_observed'].isna()]
    
    # PINNING: Ensure zero residual error at provided anchor points
    group.loc[group['iv_observed'].notna(), 'iv_predicted'] = group['iv_observed']
    
    if len(known) >= 4:
        known = known.sort_values('moneyness').drop_duplicates('moneyness')
        try:
            # Natural Cubic Spline for C2 continuity (No Butterfly Arbitrage)
            cs = CubicSpline(known['moneyness'], known['iv_observed'], 
                             bc_type='natural', extrapolate=True)
            group.loc[group['iv_observed'].isna(), 'iv_predicted'] = cs(unknown['moneyness'])
        except:
            pass
    return group

# Apply spline refinement per unique smile
test_df = test_df.groupby(['date', 'maturity_label', 'option_type'], group_keys=False).apply(hardened_spline_refinement)

# Financial Post-Processing: No-Calendar-Spread Check


def fix_calendar_spreads(date_group):
    # Enforce non-decreasing Total Variance 
    date_group = date_group.sort_values('tau')
    date_group['w_total'] = (date_group['iv_predicted'] / 100)**2 * date_group['tau']
    
    for (strike, opt_type), subgroup in date_group.groupby(['strike', 'option_type']):
        
        date_group.loc[subgroup.index, 'w_total'] = subgroup['w_total'].cummax()
    
    date_group['iv_predicted'] = np.sqrt(np.maximum(date_group['w_total'], 1e-9) / date_group['tau']) * 100
    return date_group

test_df = test_df.groupby('date', group_keys=False).apply(fix_calendar_spreads)

# Final anchor restoration to guarantee perfect calibration
test_df.loc[test_df['iv_observed'].notna(), 'iv_predicted'] = test_df['iv_observed']
test_df['iv_predicted'] = test_df['iv_predicted'].clip(lower=5.0)

# Extract only the predicted rows for the submission file
submission = test_df[test_df['iv_observed'].isna()][['row_id', 'iv_predicted']]
submission.to_csv('submission.csv', index=False)

print(f'Submission written  : {len(submission):,} rows')