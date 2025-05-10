# Crime and Socioeconomic Factors in Chicago: A Data Driven Approach

## Overview

This project analyzes the relationship between crime rates and various socioeconomic factors across Chicago's 77 community areas. The goal is to move beyond simple crime prediction and identify key socioeconomic drivers that correlate with crime patterns.

The analysis involves:
1.  Merging crime data (2015 onwards) with detailed socioeconomic indicators from the Community Data Snapshots.
2.  Engineering a comprehensive set of features (86) based on domain knowledge and established methodologies.
3.  Employing a multi-stage feature selection process using **Lasso** regression and correlation filtering to identify a robust subset of key predictors.
4.  Utilizing **XGBoost** to validate the selected features, capture non-linear relationships, and generate a final feature importance ranking.
5.  Providing an interactive web-based visualization (built with **D3.js**) to explore the spatial distribution of crime, its relationship with the identified socioeconomic factors at the community level, and city-wide trends.

The ultimate aim is to provide transparent insights into the complex interplay between socioeconomic conditions and crime, supporting data-driven policy and intervention strategies.

## Requirements

* This project uses Python 3.7 or higher. It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
* Install Dependencies: run:
    ```bash
    pip install -r requirements.txt
    ```

## Data

### Sources

* **Community Snapshot Data:** Download the required Community Data Snapshot CSV file from CMAP.
    * Source: [https://cmap.illinois.gov/data/community-data-snapshots/](https://cmap.illinois.gov/data/community-data-snapshots/)
* **Crime Data:** Download the necessary crime data (e.g., "Crimes - 2001 to Present") from the Chicago Data Portal.
    * Source: [https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

### Data Setup

1.  Create a `data` directory in the project root if it doesn't exist.
2.  Inside `data`, create subdirectories: `comm_snapshot` and `crime_data`.
3.  Place the downloaded **Community Snapshot Data** CSV inside the `data/comm_snapshot/` directory with the **exact filename** expected by the notebooks (e.g., `CCA - Community_Data_Snapshots_2024_3269398054420983008.csv`).
4.  Place the downloaded **Crime Data** inside the `data/crime_data/` directory.
    * **Note:** This file is very large.
    * The analysis notebooks expect a pre-processed version (e.g., `crime_data_clean_2015.csv`).
    * Preprocessing is done in `Feature Selection.ipynb` but the code is commented out. **You will need to uncomment the code to perform preprocessing on the raw file** before you can perform feature selection.

## Getting Started

1.  **Clone the repository (or use submitted source code) :**
    ```bash
    git clone https://github.gatech.edu/amoughnieh3/CSE-6242-SP25-team077.git
    cd CSE-6242-SP25-team077
    ```
2.  **Set up Python Environment & Install Dependencies:** (See **Requirements** section above).
3.  **Set up Data:** (See **Data** section above). Ensure data files are downloaded, placed correctly, and the crime data is preprocessed if necessary.
4.  **Exploratory Data Analysis:** You can run the `EDA.ipynb` to see the initial data exploration steps.
5.  **Feature Engineering & Selection:** You can run the `Feature Selection.ipynb` to perform feature engineering and selection.
6.  **View Interactive Visualization:**
    * Navigate to the folder `webapp`.
    * Start a simple local web server, for example:
        ```bash
        python -m http.server
        ```
    * Open your web browser and go to `http://localhost:8000`.
    * Alternatively, the visualization is deployed at: [https://cse6242-team077-vis.web.app/index.html](https://cse6242-team077-vis.web.app/index.html)
