# ASHRAE Building Energy Prediction Challenge

## Description
Q: How much does it cost to cool a skyscraper in the summer?
A: A lot! And not just in dollars, but in environmental impact.

Thankfully, significant investments are being made to improve building efficiencies to reduce costs and emissions. The question is, are the improvements working? That’s where you come in. Under pay-for-performance financing, the building owner makes payments based on the difference between their real energy consumption and what they would have used without any retrofits. The latter values have to come from a model. Current methods of estimation are fragmented and do not scale well. Some assume a specific meter type or don’t work with different building types.

In this competition, you’ll develop accurate models of metered building energy usage in the following areas: **chilled water**, **electric**, **hot water**, and **steam** meters. The data comes from over 1,000 buildings over a three-year timeframe. With better estimates of these energy-saving investments, large scale investors and financial institutions will be more inclined to invest in this area to enable progress in building efficiencies.

## About the Host
Founded in 1894, **ASHRAE** serves to advance the arts and sciences of heating, ventilation, air conditioning, refrigeration, and their allied fields. ASHRAE members represent building system design and industrial process professionals around the world. With over 54,000 members serving in 132 countries, ASHRAE supports research, standards writing, publishing, and continuing education—shaping tomorrow’s built environment today.

*Banner photo by Federico Beccari on Unsplash*

## Evaluation
The evaluation metric for this competition is **Root Mean Squared Logarithmic Error (RMSLE)**, calculated as:

$$
\epsilon = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \bigl(\log(p_i + 1) - \log(a_i + 1)\bigr)^2}
$$

Where:
- \(n\) is the total number of observations (public + private)
- \(p_i\) is your prediction of the meter reading
- \(a_i\) is the actual meter reading
- \(\log(x)\) is the natural logarithm of \(x\)

*Note: Not all rows in the test set will necessarily be scored.*

## Submission File
For each row in the test set, you must predict the `meter_reading`. The file should contain a header and have the following format:

```
building_id,meter,timestamp,meter_reading
0,0,2016-10-31 00:00:00,0
1,0,2016-10-31 00:00:00,0
2,0,2016-10-31 00:00:00,0
3,0,2016-10-31 00:00:00,0
4,0,2016-10-31 00:00:00,0
5,0,2016-10-31 00:00:00,0
6,0,2016-10-31 00:00:00,0
7,0,2016-10-31 00:00:00,0
7,1,2016-10-31 00:00:00,0
```

## Dataset Description
Assessing the value of energy efficiency improvements can be challenging, since we can’t directly observe a building’s consumption without retrofits. This competition asks you to build **counterfactual models**: predict what each building’s energy usage would have been, then compare against actual post-retrofit readings to quantify savings. More accurate models support better financing and larger scale investments in efficiency.

### Files
- **train_label.csv**
  - `building_id` – building identifier
  - `meter` – meter type {0: electricity, 1: chilledwater, 2: steam, 3: hotwater}
  - `timestamp` – measurement date and time
  - `meter_reading` – energy consumption (kWh or equivalent)

- **train_features.csv**, **test_features.csv**
  - `site_id` – meteorological station link
  - `air_temperature`, `dew_temperature` (℃)
  - `cloud_coverage` (oktas)
  - `precip_depth_1_hr` (mm)
  - `sea_level_pressure` (hPa)
  - `wind_direction` (0–360°), `wind_speed` (m/s)

- **building_meta.csv**
  - `site_id` – links to weather files
  - `building_id` – foreign key for train/test
  - `primary_use` – building category
  - `square_feet` – floor area
  - `year_built` – opening year
  - `floor_count` – number of floors

- **sample_submission.csv**
  - `meter_reading` - fill prediction for meter reading
