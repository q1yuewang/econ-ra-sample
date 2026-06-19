# econ-ra-sample

Code sample showing basic data cleaning, reshaping, and plotting for empirical research. 

## data

* **df_micro**: micro survey data with intentionally added outliers and missing values.
* **df_wages / df_firms**: worker and firm tables used to show merges.
* **df_panel_wide**: country-year macro panels used to show wide/long transitions.

## tasks covered

* cleaning missing values and trimming extreme outliers.
* merging separate relational datasets on a common key.
* transforming wide cross-country panels to long form and back.
* plotting distributions, correlations, and stratified linear fits.

## run

```bash
pip install numpy pandas matplotlib seaborn
python ra_technical_sample.py
