# BDA LAB - Big Data Analytics Practicals

## 📋 Course Information

**Institution**: ISBM College of Engineering  
**Subject**: BDA LAB (Big Data Analytics Laboratory)  
**Student Name**: RUSHIL CHAITANYA DHUBE  
**Year**: BE (2023-2024)  

---

## 📚 Project Overview

This repository contains practical implementations and exercises for the Big Data Analytics Laboratory course. The practicals focus on exploratory data analysis, machine learning classification, and database query operations using Python.

---

## 📁 Project Structure

```
BDA LAB/
├── Iris Dataset/
│   ├── iris_dataset.ipynb                          # Practical 1: EDA
│   ├── iris_histograms.png
│   ├── iris_boxplots.png
│   └── iris_distributions_by_species.png
│
├── Pima Indians Diabetes/
│   ├── pima_indians_diabetes_dataset.ipynb         # Practical 2: Naive Bayes Classification
│   ├── pima_class_distribution.png
│   ├── pima_model_performance.png
│   ├── pima_feature_distributions.png
│   ├── max_snowfall_2013.csv
│   └── station_snowfall_stats_2013.csv
│
├── Weather Dataset/
│   ├── weather_database.ipynb                      # Practical 4: SQLite Database Query
│   ├── weather_data.db
│   ├── weather_snowfall_analysis.png
│   ├── max_snowfall_2013.csv
│   └── station_snowfall_stats_2013.csv
│
├── requirements.txt                                # Python dependencies
├── .gitignore                                      # Git ignore file
└── README.md                                       # This file
```

---

## 🎯 Practicals

### Practical 1: Exploratory Data Analysis of the Iris Dataset
**Location**: `Iris Dataset/iris_dataset.ipynb`

**Objectives**:
- Load the Iris dataset into a DataFrame
- Analyze features (count and data types)
- Compute and display summary statistics
- Create histograms for feature distributions
- Create boxplots to identify outliers
- Compare distributions and identify anomalies

**Key Outputs**:
- Summary statistics for all numeric features
- Distribution histograms with mean/median lines
- Combined boxplots for outlier detection
- Violin plots showing species-wise distributions

---

### Practical 2: Classification Using Naive Bayes Algorithm on Pima Indians Diabetes Dataset
**Location**: `Pima Indians Diabetes/pima_indians_diabetes_dataset.ipynb`

**Objectives**:
- Load and explore the Pima Indians Diabetes dataset
- Split data into training (70%) and test (30%) datasets
- Summarize training data properties by class
- Train a Naive Bayes classifier using scikit-learn
- Make predictions on test data
- Evaluate model performance with multiple metrics

**Key Metrics**:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (raw and normalized)
- Sensitivity & Specificity
- Detailed classification report

**Visualizations**:
- Class distribution charts
- Model performance metrics bar chart
- Confusion matrix heatmap
- Feature distributions by class

---

### Practical 4: Weather Database Analysis - Maximum Snowfall Detection
**Location**: `Weather Dataset/weather_database.ipynb`

**Objectives**:
- Connect to SQLite weather database using Python
- Explore database structure and schema
- Write SQL queries to extract weather data
- Identify the day and station with maximum snowfall in 2013
- Visualize weather data patterns and trends
- Generate comprehensive weather analysis reports

**Key Queries**:
- Find maximum snowfall value in 2013
- Identify corresponding day and station
- Calculate snowfall statistics by station
- Analyze monthly snowfall patterns

**Outputs**:
- SQLite database (weather_data.db)
- CSV reports (max_snowfall, station statistics)
- Visualization charts (snowfall analysis)

---

## 🛠️ Software Requirements

- **Python**: Version 3.5 or higher
- **Key Libraries**:
  - pandas (v1.0+) - Data manipulation and analysis
  - numpy (v1.21+) - Numerical computing
  - matplotlib (v3.1+) - Data visualization
  - seaborn (v0.10+) - Statistical visualization
  - scikit-learn (v0.22+) - Machine learning
  - scipy (v1.7+) - Scientific computing
  - sqlite3 (built-in) - Database management

---

## 💻 Hardware Requirements

- Computer with sufficient processing power and memory
- Minimum 2GB RAM recommended
- Disk space: ~500MB for all practicals and outputs

---

## 📦 Installation & Setup

### 1. Clone or Download the Repository
```bash
cd BDA LAB
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
source myenv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

---

## 🚀 Running the Practicals

### Using Jupyter Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Navigate to the desired practical**:
   - Open `Iris Dataset/iris_dataset.ipynb`
   - Open `Pima Indians Diabetes/pima_indians_diabetes_dataset.ipynb`
   - Open `Weather Dataset/weather_database.ipynb`

3. **Execute cells**: Press `Shift + Enter` to run each cell

### Using VS Code

1. Install the Jupyter extension in VS Code
2. Open any `.ipynb` file
3. Click "Run All" or run individual cells

### Using Python Command Line

```bash
jupyter nbconvert --to notebook --execute Iris\ Dataset/iris_dataset.ipynb
```

---

## 📊 Key Findings

### Practical 1: Iris Dataset Analysis
- **Total Samples**: 150 iris flower measurements
- **Features**: 4 numeric features (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 species (setosa, versicolor, virginica)
- **Key Insight**: No significant outliers; petal measurements are best discriminators

### Practical 2: Naive Bayes Classification
- **Dataset**: 768 patient records for diabetes prediction
- **Train-Test Split**: 70-30 ratio
- **Model**: Gaussian Naive Bayes
- **Performance**: Balanced accuracy with good sensitivity/specificity

### Practical 4: Weather Database Analysis
- **Database**: SQLite with 5 weather stations and 1,825 records
- **Time Period**: 2013 full year data
- **Analysis**: Maximum snowfall identification with monthly trends

---

## 📝 Notes

- All output files (databases, CSVs, PNGs) are saved in their respective practical folders
- The `requirements.txt` file contains all necessary Python packages
- Virtual environment `myenv/` is already configured and can be activated as needed
- All practicals follow academic standards and best practices

---

## ⚙️ Troubleshooting

### Issue: Module not found error
**Solution**: Install missing packages using `pip install <package-name>`

### Issue: Database file not found
**Solution**: Ensure you're running the notebook from the correct directory

### Issue: Jupyter not starting
**Solution**: Install Jupyter using `pip install jupyter`

### Issue: Plot not displaying
**Solution**: Ensure matplotlib is installed and run `%matplotlib inline` in notebook

---

## 📞 Support

For issues or questions regarding these practicals, refer to:
- Lab manual documentation
- Python official documentation
- scikit-learn documentation
- Jupyter Notebook documentation

---

## 📄 License

This work is part of the coursework for ISBM College of Engineering.

---

## ✍️ Author

**Name**: RUSHIL CHAITANYA DHUBE  
**Institution**: ISBM College of Engineering  
**Course**: Big Data Analytics Lab (BDA LAB)  
**Year**: BE (2023-2024)  

**Date**: April 2026

---

## 📌 References

- Iris Dataset: https://archive.ics.uci.edu/ml/datasets/Iris
- Pima Indians Diabetes Dataset: https://www.kaggle.com/uciml/pima-indians-diabetes-database
- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/
- Matplotlib Documentation: https://matplotlib.org/

---

**Last Updated**: April 18, 2026
#   B i g - D a t a - A n a l y t i c s - P r a c t i c a l - S e m - 7  
 