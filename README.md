# Flight Delay Analytics & Prediction

Analysis of US domestic flight delays and a machine learning model to predict delays of 15+ minutes, with an estimate of business impact.

## Problem
Which factors drive flight delays, and can delays be predicted early enough for an airline to act on them?

## Dataset
[2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays) (US Department of Transportation, via Kaggle). About 5.8M flights; a random sample of 500k non-cancelled flights was used. A flight is "delayed" if it arrives 15+ minutes late.

## Approach
1. **Data cleaning & feature engineering:** removed cancelled/diverted flights; created departure hour, weekend flag and delay label.
2. **Exploratory analysis:** delay rates by airline, hour, month, day of week, airport and cause.
3. **Statistical testing:** Chi-square (delay vs. airline), Welch t-test (weekend vs. weekday), ANOVA (delay across months).
4. **Modeling:** Logistic Regression, Random Forest and XGBoost, using only pre-departure information. Trained on Jan-Sep and tested on Oct-Dec (time-based split) to avoid leakage.
5. **Business impact:** estimated the value of proactively managing the top 10% highest-risk flights.

## Key Findings
- 17.9% of flights were delayed 15+ minutes; airline delay rates range from about 11% (Hawaiian) to about 28% (Spirit).
- Delays build through the day (about 7% at 5am to about 25% in the evening) and peak in Feb, Jun-Jul and Dec-Jan.
- About 72% of delay minutes come from late-arriving aircraft (40%) and airline-caused issues (32%); weather is only about 5%. Most delay is operational and controllable.
- All statistical tests were significant (p < 0.001). The weekend vs. weekday difference is small in practice (about 2 minutes).

## Model Results
| Model | ROC-AUC (test) |
|---|---|
| Logistic Regression | 0.598 |
| Random Forest | 0.610 |
| XGBoost | 0.604 |

- The model's top-risk 10% of flights are delayed 23.9% of the time vs. 15.6% overall (**1.53x lift**).
- The modest AUC reflects schedule-only features. Weather, inbound-aircraft status and air-traffic data would improve it.

## Business Impact
Under illustrative assumptions ($75 per delay minute, 10% reduction in delay minutes for flagged flights), proactive handling of the top-risk 10% could save about $1.4M on the Oct-Dec test sample. These assumptions are for illustration and need validation with real operations data.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, SciPy, Matplotlib, Seaborn

## Files
- `flight-delay-analysis.ipynb`: full notebook (EDA, statistical tests, modeling, business impact)

## Limitations & Next Steps
- Add weather forecasts, inbound-aircraft and air-traffic data.
- Validate cost and reduction assumptions with operational data.
- Build a monitoring dashboard (Streamlit/Power BI).
