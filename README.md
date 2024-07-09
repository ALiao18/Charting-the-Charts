# Spotify Song Popularity Analysis

## Project Overview
This project analyzes a dataset of 52,000 Spotify songs to understand the factors influencing song popularity on the platform. It was completed as a capstone project for the Principles of Data Science course.

## Dataset
The dataset (`spotify52kData.csv`) contains information on 52,000 songs, including:
- Audio features (danceability, energy, loudness, etc.)
- Metadata (artist, album, track name)
- Popularity scores

## Project Structure
1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Predictive Modeling
4. Dimensionality Reduction (PCA)
5. Classification Tasks

## Key Findings
- No single audio feature strongly predicts song popularity
- Explicit songs tend to be more popular
- Strong correlation between energy and loudness
- Multiple principal components needed to explain data variance

## Technologies Used
- Python
- Libraries: pandas, numpy, matplotlib, scipy, sklearn

## How to Run
1. Clone this repository
2. Ensure you have the required libraries installed
3. Run the Jupyter notebook or Python scripts in the following order:
   - data_preprocessing.py
   - exploratory_data_analysis.py
   - predictive_modeling.py
   - dimensionality_reduction.py

## File Descriptions
- `spotify52kData.csv`: Raw dataset
- `data_preprocessing.py`: Script for cleaning and preparing the data
- `exploratory_data_analysis.py`: Script for EDA and visualizations
- `predictive_modeling.py`: Contains code for building and evaluating predictive models
- `dimensionality_reduction.py`: Implements PCA and related analyses

## Results
Detailed results and visualizations can be found in the Jupyter notebook or in the `results` folder.

## Future Work
- Incorporate non-audio features (e.g., lyrics, artist popularity)
- Explore more advanced machine learning models
- Analyze temporal trends in song popularity

## Author
Andrew Liao

## Acknowledgments
- Prof. Wallisch for providing the dataset
- NYU for the course structure and guidance
