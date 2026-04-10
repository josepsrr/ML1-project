# Can Words Mitigate the Market for Lemons?

This repository contains a text-mining project on the Spanish used-car market. The project studies whether more informative listing language is associated with a price premium after controlling for observable car characteristics such as age, mileage, power, seller type, and region.

## Research question

Do online used-car listings with more transparent and information-rich descriptions reduce information asymmetry and command higher prices?

## Project structure

- `project.py`: the only script you need. It downloads the data if needed, cleans the listings, builds the disclosure score, runs the models, and saves the outputs.
- `outputs/`: generated tables and figures.
- `report/paper.tex`: two-column report draft in scientific-paper format.

## Dataset

The project uses the public dataset **"spanish_used_car_market: Coches de segunda mano a la venta en España"** published on Zenodo:

- DOI: [10.5281/zenodo.4252636](https://doi.org/10.5281/zenodo.4252636)
- Record: [https://zenodo.org/records/4252636](https://zenodo.org/records/4252636)

The raw CSV is not committed because of its size. The script downloads it automatically if needed.

## Quick start

```bash
python project.py
```

This command creates:

- `outputs/tables/*.csv`
- `outputs/tables/*.tex`
- `outputs/figures/*.png`

## Method overview

The script combines:

1. Text cleaning and normalization for Spanish listing text.
2. Hand-crafted lexical indicators of transparency, maintenance disclosure, defect disclosure, and promotional language.
3. A hedonic price model using structured listing characteristics.
4. A text-augmented residual model using TF-IDF n-grams.
5. An interaction regression testing whether disclosure matters more for private sellers.

## Main variables

- `price_eur`: listing price in euros.
- `km`: mileage.
- `vehicle_age`: ad snapshot year minus registration year.
- `power_cv`: horsepower.
- `advertizer_type_clean`: `Profesional` or `Particular`.
- `disclosure_index_z`: standardized disclosure index built from text cues.

## Key findings

- The cleaned sample contains `180,514` Spanish used-car listings.
- A one-standard-deviation increase in the disclosure index is associated with roughly `2.1%` higher price in the interaction regression.
- The structured baseline reaches holdout `R² = 0.839`.
- Adding disclosure features raises holdout `R²` to `0.845`.
- Adding TF-IDF residual text raises holdout `R²` to `0.852`.
- Private sellers disclose less on average and receive lower conditional prices, but the additional return to disclosure for private sellers is not statistically significant in this specification.

## Reproducibility notes

- The workflow is deterministic through a fixed random seed.
- The code does not require a downloaded spaCy Spanish model.
- The script stores the dataset in `data/raw/dataset.csv`.
