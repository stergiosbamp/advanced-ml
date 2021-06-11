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

NIH chest X-rays images to predict 14 possible deseases.
However an X-ray may be annotated with more than one desease.

Link: https://www.kaggle.com/nih-chest-xrays/data


**Methods**

Extract features via pre-trained convolutional neural network. 
We utilize the **Xception** model and apply **fine-tuning** on our dataset, with
- *binary-cross* entropy loss function
- *BP-MLL* loss functions


*Multi-label classification techniques*

- Binary Relevance
- Classifier chain
- Label Powerset
- Random K-Labelsets (RAKEL)

with SGD classifier as a base classifier.

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
