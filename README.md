# Advanced Topics in Machine Learning

This is the repository for "Advanced Topics in Machine Learning" MSc courses' project.

Contributors

* Stergios Bampakis
* Themistoklis Spanoudis 

## Problems addressed

Our focus of work is around the medical and healthcare sector.

### Class imbalance

**Dataset**

Predict whether a fetal's CTG is
1. Normal
2. Suspect
3. Pathological

Link: https://www.kaggle.com/andrewmvd/fetal-health-classification

**Models**

- Logistic Regression
- Random Forest
- Gradient Boosting
- Multilayer Perceptron

**Sampling techniques**

*Undersampling*

- Random undersampling
- Removal of Tomek links
- NearMiss version 1

*Oversampling*

- Random oversampling
- SMOTE for synthetic examples
- Borderline SMOTE
- ADASYN

*Hybrid*

- SMOTE + Removal of Tomek links

### Multi-label learning

**Dataset**


**Methods**



## Project setup

Create virtual environment

```
$ python3 -m venv venv
$ source venv/bin/activate
```

Upgrade pip

```
$ python -m pip install --upgrade pip
```

Install dependencies

```
$ pip install -r requirements.txt
```
