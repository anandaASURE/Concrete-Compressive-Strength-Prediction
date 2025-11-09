# Concrete Compressive Strength Prediction
## ANANDA JANA | @ IISER TVM

This project predicts concrete strength using machine learning to help engineers optimize mix designs and reduce costly lab testing. Think of it as a smart assistant that tells you how strong your concrete will be before you make it.

## What's This About?

In construction, testing concrete strength takes 28 days and requires expensive lab equipment. This notebook uses machine learning to predict compressive strength instantly based on the ingredient mix, potentially saving weeks of waiting and thousands in testing costs.

## What It Does

* Analyzes 1,030 concrete samples with 8 different ingredients
* Compares multiple ML models (Linear Regression, Ridge, Random Forest)
* Identifies which ingredients matter most for strength
* Uses PCA to visualize high-dimensional concrete chemistry
* Discovers natural groupings in concrete mix designs
* Provides practical insights for mix design optimization

## The Workflow

1. **Data Preprocessing**: Handle missing values and prepare features
2. **Exploratory Analysis**: Visualize distributions and correlations
3. **Feature Scaling**: Standardize ingredients for fair comparison
4. **Model Training**: Compare 3 regression algorithms using cross-validation
5. **Feature Importance**: Identify critical ingredients (cement, age, water)
6. **Dimensionality Reduction**: Apply PCA to understand data structure
7. **Clustering**: Find natural concrete "types" in reduced space
8. **Model Persistence**: Save trained model for future deployment

## Files

* `concrete_strength_prediction.ipynb` – Main notebook with complete analysis
* `Concrete Compressive Strength.csv` – Dataset (1,030 samples, 9 columns)
* `concrete_strength_model.joblib` – Saved Random Forest model (ready for deployment)
* `feature_scaler.joblib` – Saved preprocessing scaler

## What You Need
```bash
numpy
pandas
seaborn
matplotlib
scikit-learn
joblib
```

Install everything with:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn joblib
```

## How to Run

1. Place the CSV file in the same directory as the notebook
2. Open `concrete_strength_prediction.ipynb` in Jupyter
3. Run all cells sequentially

You'll see:
* Data quality checks and preprocessing steps
* Visualizations of feature relationships
* Model comparison results (Random Forest performs best)
* Feature importance rankings
* PCA projections colored by strength
* Cluster analysis revealing concrete "families"

## Key Findings

**Model Performance:**
* Random Forest achieves lowest RMSE on test set
* Significant improvement over baseline (predicting mean)
* R² score demonstrates good predictive capability

**Important Ingredients:**
* Cement content is the dominant factor
* Age (curing time) critically affects strength
* Water-to-cement ratio matters significantly
* Superplasticizer and fly ash have moderate effects

**Data Structure:**
* First 2-3 principal components capture majority of variance
* 4 natural clusters exist, corresponding to different strength ranges
* Clusters likely represent distinct design philosophies (high-strength, standard, lean mixes)

## Practical Applications

* **Cost Reduction**: Reduces need for extensive physical testing
* **Speed**: Instant predictions vs. 28-day waiting period
* **Optimization**: Enables faster iteration on mix designs
* **Quality Control**: Early identification of suboptimal batches
* **Sustainability**: Reduces material waste from trial batches

## Dataset

The dataset contains concrete samples tested under laboratory conditions with standardized procedures.

**Input Features:**
- Cement, Blast Furnace Slag, Fly Ash, Water
- Superplasticizer, Coarse Aggregate, Fine Aggregate, Age

**Target Variable:**
- Compressive Strength (MPa)

## Technical Approach

* **Cross-validation**: 5-fold CV for reliable model selection
* **Feature scaling**: StandardScaler for consistent comparisons
* **Dimensionality reduction**: PCA reveals inherent structure
* **Clustering**: K-Means identifies natural groupings
* **Model persistence**: Joblib serialization (deployment-ready)
* **Interpretability**: Feature importance and coefficient analysis

## Using the Saved Model

The trained model and scaler are saved as `.joblib` files and can be loaded for making predictions:
```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('concrete_strength_model.joblib')
scaler = joblib.load('feature_scaler.joblib')

# Example prediction
new_sample = np.array([[cement, slag, fly_ash, water, superplasticizer, 
                        coarse_agg, fine_agg, age]])
scaled_sample = scaler.transform(new_sample)
predicted_strength = model.predict(scaled_sample)
```

*Note: This project focuses on model development and analysis. The saved models are ready for deployment but server/API implementation is not included.*

## References

* Yeh, I-Cheng (1998). "Modeling of strength of high-performance concrete using artificial neural networks." *Cement and Concrete Research*
* Scikit-learn documentation for RandomForestRegressor and PCA
* Industry practices for concrete mix design

## Note

This project demonstrates a straightforward ML workflow for regression problems with practical construction industry applications. The focus is on clarity, interpretability, and extracting actionable insights from data.

---

**License:** MIT  
**Contact:** ANANDA JANA | IISER Thiruvananthapuram  
**Last Updated:** November 2025
