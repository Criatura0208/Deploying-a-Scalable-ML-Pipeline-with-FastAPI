# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier built using the `sklearn.ensemble.RandomForestClassifier` algorithm. It is designed to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related attributes from the UCI Census Income (Adult) dataset. Developed in Python with `scikit-learn`, the model uses `OneHotEncoder` for preprocessing categorical variables and applies label binarization for the target. Features such as education, workclass, marital status, race, and sex are encoded accordingly. The dataset was split into 80% for training and 20% for evaluation to assess the model’s performance.

## Intended Use

This model is intended for educational and research purposes, particularly to illustrate the machine learning development lifecycle—including data preprocessing, model training, evaluation, persistence, and fairness assessment through performance slicing. It is not suitable for deployment in real-world applications or high-stakes decision-making scenarios such as hiring, credit assessment, or policy development without substantial additional fairness, ethical, and regulatory evaluation.

## Training Data

The model was trained on the *UCI Adult Census Income dataset*, which includes demographic information from the 1994 U.S. Census. The dataset includes 32,561 rows and 15 features, including:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Native Country
- Hours per Week
- and others

Categorical features were one-hot encoded, and the label (`salary`) was binarized into two classes: `>50K` and `<=50K`.

## Evaluation Data

The evaluation dataset comprises a 20% split from the original dataset, held out during training to ensure unbiased performance assessment. Evaluation was carried out on two levels:
- The entire test dataset, to measure overall model performance.
- Specific subsets (or "slices") of the test data, defined by unique values of categorical features such as `education`, `workclass`, and `race`. This approach helps identify potential disparities in model performance across different demographic subgroups.

## Metrics

The model is evaluated using:
- Precision: Proportion of positive identifications that were actually correct.
- Recall: Proportion of actual positives that were correctly identified.
- F1 Score: Harmonic mean of precision and recall, balancing both.
 
# Overall Performance

|Metric|Value| |---------|---------| |Precision|0.7391| |Recall|0.6384| |F1 Score|0.6851|

# Sliced Performance Examples

Below are selected examples from the `slice_output.txt` that demonstrate model performance across subgroups:
Workclass: Federal-gov
- Precision: 0.7971 | Recall: 0.7857 | F1: 0.7914
Workclass: Private
- Precision: 0.7362 | Recall: 0.6384 | F1: 0.6838
Education: Bachelors
- Precision: 0.7569 | Recall: 0.7333 | F1: 0.7449
Education: 7th-8th
- Precision: 1.0000 | Recall: 0.0000 | F1: 0.0000
Race: White
- Precision: 0.7372 | Recall: 0.6366 | F1: 0.6832
Race: Asian-Pac-Islander
- Precision: 0.7857 | Recall: 0.7097 | F1: 0.7458
Sex: Female
- Precision: 0.7256 | Recall: 0.5107 | F1: 0.5995
Sex: Male
- Precision: 0.7410 | Recall: 0.6607 | F1: 0.6985

The model tends to perform better on individuals with higher levels of education (e.g., Master's or Doctorate degrees) and those employed in government-related workclasses. However, recall is notably lower for certain groups—particularly individuals with less formal education or those belonging to smaller demographic subgroups—indicating potential disparities in model performance.

## Ethical Considerations

# Bias and Fairness
- Performance Disparities:
The model exhibits varying levels of performance across demographic groups, with noticeable differences in recall. Certain groups—such as women and individuals with lower levels of formal education—tend to be underrepresented or underpredicted, indicating potential bias in the model’s outcomes.
- Sensitive Features:
The dataset includes protected attributes like race and sex, which raises ethical concerns regarding possible discriminatory behavior. While these features were not used to implement fairness interventions, they were analyzed to assess disparities in model performance across subgroups.
- Small Sample Groups:
Some demographic categories, such as individuals from "Laos" or "Greece," are represented by very few samples. This data sparsity can result in unreliable metrics, including artificially perfect or zero F1 scores, which do not accurately reflect model performance.
- Temporal Limitations:
The model was trained on data from the 1994 U.S. Census and may not reflect current demographic distributions, labor markets, or income trends. As a result, its generalizability to present-day applications is limited.

## Caveats and Recommendations

# Limitations and Considerations
- Not Production-Ready:
This model is not suitable for deployment in real-world systems without comprehensive bias and fairness evaluation. Its current use is intended solely for educational and research purposes.
- Additional Auditing Required:
Before considering any production use, fairness metrics—such as disparate impact ratio or equal opportunity difference—should be applied to audit and address potential disparities across demographic groups.
- Sample Imbalance:
Several demographic slices suffer from low sample sizes, leading to unreliable performance metrics. Future work should focus on balancing the dataset or exploring data augmentation techniques to improve representation.
- Model Interpretability:
Random Forests, while powerful, lack inherent interpretability. For applications where explainability is crucial, consider using more interpretable models or applying post-hoc interpretability tools like SHAP or LIME.
- Continuous Monitoring:
If the model is adapted for deployment, it should be continuously monitored for changes in data distribution, model performance, and fairness over time to ensure reliability and ethical use.
