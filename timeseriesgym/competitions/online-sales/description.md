# Monthly Online Sales Prediction

## Description
The objective of the competition is to help us build as good a model as possible to predict monthly online sales of a product. Imagine the products are online self-help programs following an initial advertising campaign.

We have shared the data in the comma separated values (CSV) format. Each row in this data set represents a different consumer product.

The first 12 columns (`Outcome_M1` through `Outcome_M12`) contain the monthly online sales for the first 12 months after the product launches.

`Date_1` is the day number the major advertising campaign began and the product launched.

`Date_2` is the day number the product was announced and a pre-release advertising campaign began.

Other columns in the data set are features of the product and the advertising campaign. `Quan_x` are quantitative variables and `Cat_x` are categorical variables. Binary categorical variables are measured as `1` if the product had the feature and `0` if it did not.

## Evaluation
The task is to predict the first 12 months of online sales for a product based on product features. For every record in **test_features.csv**, use the product’s feature set to forecast its online sales for the first 12 months.

The evaluation metric is **Root Mean Squared Logarithmic Error (RMSLE)** across all 12 prediction columns, calculated as:

$$
\mathrm{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \bigl(\log(p_i + 1) - \log(a_i + 1)\bigr)^2}
$$

Where:
- \(n\) is the total number of predictions
- \(p_i\) is your prediction for a given month
- \(a_i\) is the actual sales for that month
- \(\log(x)\) is the natural logarithm of \(x\)

## Dataset Description
We have shared the data in CSV format. Each row represents a different consumer product.

- **Outcome_M1 … Outcome_M12:** Monthly online sales for months 1–12 after launch
- **Date_1:** Day number when the major advertising campaign began and the product launched
- **Date_2:** Day number when the product was announced and a pre-release advertising campaign began
- **Quan_x:** Quantitative features of the product/campaign
- **Cat_x:** Categorical features of the product/campaign (binary measured as 1/0)

### File Descriptions
* **train.csv** – includes both labels (`Outcome_M1`–`Outcome_M12`) and features (`Date_1`, `Date_2`, `Quan_x`, `Cat_x`) for training
* **test_features.csv** – includes test features (`Date_1`, `Date_2`, `Quan_x`, `Cat_x`)
* **sample_submission.csv** – includes sample submission file
