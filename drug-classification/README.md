# Drug Classification Using Decision Trees

A machine learning project that classifies different types of drugs based on patient characteristics using Decision Tree algorithm.

## 📊 Dataset

The dataset contains 200 patient records with the following features:
- **Age**: Patient age
- **Sex**: Gender (M/F)
- **BP**: Blood Pressure level (HIGH/LOW/NORMAL)
- **Cholesterol**: Cholesterol level (HIGH/NORMAL)
- **Na_to_K**: Sodium to Potassium ratio in blood
- **Drug**: Target variable (DrugA, DrugB, DrugC, DrugX, DrugY)

## 🚀 Project Overview

This project demonstrates:
- Data preprocessing with Label Encoding
- Decision Tree classification with entropy criterion
- Model evaluation and visualization
- Perfect classification accuracy on test data

## 📋 Requirements

```
pandas
scikit-learn
matplotlib
```

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/drug-classification.git
cd drug-classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python drug_classification.py
```

## 📈 Results

- **Accuracy**: 100% on test set
- **Model**: Decision Tree with entropy criterion and max depth of 5
- **Features**: All features contribute to the classification decision

## 📊 Model Performance

The Decision Tree classifier achieved perfect classification with:
- Precision: 1.00 for all classes
- Recall: 1.00 for all classes
- F1-Score: 1.00 for all classes

## 📸 Visualization

The project generates a decision tree visualization showing the classification logic and decision boundaries.

## 🗂️ Project Structure

```
drug-classification/
├── drug200.csv                 # Dataset
├── drug_classification.py      # Main script
├── decision_tree.png          # Generated tree visualization
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies
```

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.
