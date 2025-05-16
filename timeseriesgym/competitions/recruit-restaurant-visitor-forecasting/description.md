# Recruit Restaurant Visitor Forecasting

## Description
Running a thriving local restaurant isn't always as charming as first impressions appear. There are often all sorts of unexpected troubles popping up that could hurt business.

One common predicament is that restaurants need to know how many customers to expect each day to effectively purchase ingredients and schedule staff members. This forecast isn't easy to make because many unpredictable factors affect restaurant attendance, like weather and local competition. It's even harder for newer restaurants with little historical data.

Recruit Holdings has unique access to key datasets that could make automated future customer prediction possible. Specifically, Recruit Holdings owns Hot Pepper Gourmet (a restaurant review service), AirREGI (a restaurant point of sales service), and Restaurant Board (reservation log management software).

In this competition, you're challenged to use reservation and visitation data to predict the total number of visitors to a restaurant for future dates. This information will help restaurants be much more efficient and allow them to focus on creating an enjoyable dining experience for their customers.

## Evaluation
Submissions are evaluated on the root mean squared logarithmic error.

The RMSLE is calculated as:

$$
\mathrm{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \bigl(\log(p_i + 1) - \log(a_i + 1)\bigr)^2}
$$

Where:
- \(n\) is the total number of observations
- \(p_i\) is your prediction of visitors
- \(a_i\) is the actual number of visitors
- \(\log(x)\) is the natural logarithm of \(x\)

## Submission File
For every store and date combination in the test set, submission files should contain two columns: `id` and `visitors`. The `id` is formed by concatenating the `air_store_id` and `visit_date` with an underscore. The file should contain a header and have the following format:

```
id,visitors
air_00a91d42b08b08d9_2017-04-23,0
air_00a91d42b08b08d9_2017-04-24,0
air_00a91d42b08b08d9_2017-04-25,0
```

## Dataset Description
In this competition, you are provided a time-series forecasting problem centered around restaurant visitors. The data comes from two separate systems:

- **Hot Pepper Gourmet (hpg):** similar to Yelp, here users can search restaurants and also make a reservation online
- **AirREGI / Restaurant Board (air):** similar to Square, a reservation control and cash register system

You must use the reservations, visits, and other information from these sites to forecast future restaurant visitor totals on a given date. The training data covers the dates from 2016 until Feburary 2017. The test set covers the dates from March 2017 until the last week of April. The test set is split based on time (the public fold coming first, the private fold following the public) and covers a chosen subset of the air restaurants.

### File Descriptions
This is a relational dataset from two systems. Each file is prefaced with the source (`air_` or `hpg_`) to indicate its origin. Each restaurant has a unique `air_store_id` and `hpg_store_id`. Note that not all restaurants are covered by both systems, and that you have been provided data beyond the restaurants for which you must forecast. Latitudes and longitudes are not exact to discourage de-identification of restaurants.

- **air_reserve.csv**
  Contains reservations made in the AirREGI system.
  - `air_store_id` – the restaurant’s id in the air system
  - `visit_datetime` – the time of the reservation
  - `reserve_datetime` – the time the reservation was made
  - `reserve_visitors` – the number of visitors for that reservation

- **hpg_reserve.csv**
  Contains reservations made in the Hot Pepper Gourmet system.
  - `hpg_store_id` – the restaurant’s id in the hpg system
  - `visit_datetime` – the time of the reservation
  - `reserve_datetime` – the time the reservation was made
  - `reserve_visitors` – the number of visitors for that reservation

- **air_store_info.csv**
  Information about select air restaurants.
  - `air_store_id`
  - `air_genre_name`
  - `air_area_name`
  - `latitude`, `longitude` (of the area to which the store belongs)

- **hpg_store_info.csv**
  Information about select hpg restaurants.
  - `hpg_store_id`
  - `hpg_genre_name`
  - `hpg_area_name`
  - `latitude`, `longitude` (of the area to which the store belongs)

- **store_id_relation.csv**
  Allows you to join select restaurants that exist in both systems.
  - `hpg_store_id`
  - `air_store_id`

- **train.csv**
  Historical visit data for the air restaurants used for training.
  - `air_store_id`
  - `visit_date` – the date
  - `visitors` – the number of visitors to the restaurant on that date

- **sample_submission.csv**
  A sample submission file in the correct format, including the days for which you must forecast.
  - `id` – formed by concatenating `air_store_id` and `visit_date`
  - `visitors` – the forecasted number of visitors

- **date_info.csv**
  Basic information about calendar dates in the dataset.
  - `calendar_date`
  - `day_of_week`
  - `holiday_flg` – is the day a holiday in Japan?
