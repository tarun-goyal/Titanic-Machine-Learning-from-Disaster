import pandas as pd
from sklearn.linear_model import LinearRegression


def _impute_missing_values(design_matrix):
    """Impute values for all the features having missing values."""
    missing_cols = {'mean': ['Fare'],
                    'mode': ['Embarked']}
    for col in missing_cols['mean']:
        design_matrix[col] = design_matrix[col].fillna(
            design_matrix[col].mean())
    for col in missing_cols['mode']:
        design_matrix[col] = design_matrix[col].fillna(
            design_matrix[col].value_counts().index[0])
    design_matrix = _imputation_using_regression(design_matrix)
    return design_matrix


def _convert_data_types(design_matrix):
    """Conversion of categorical type continuous features into objects"""
    conversion_list = ['Pclass', 'SibSp', 'Parch']
    for column in conversion_list:
        design_matrix[column] = design_matrix[column].apply(str)
    return design_matrix


def _capture_title_from_names(design_matrix):
    design_matrix['Mr'] = design_matrix['Name'].str.contains("Mr. ")
    design_matrix['Mrs'] = design_matrix['Name'].str.contains("Mrs. ")
    design_matrix['Master'] = design_matrix['Name'].str.contains("Master. ")
    design_matrix['Miss'] = design_matrix['Name']\
        .str.contains("Miss. |Ms. |Mme. |Mlle. ")
    design_matrix['Sir'] = design_matrix['Name'].str.contains(
        "Col. |Rev. |Dr. |Dona. |Don. |Major. |Sir. |Capt. |Countess. |"
        "Jonkheer. ")
    design_matrix.drop('Name', axis=1, inplace=True)
    return design_matrix


def _create_dummies_for_categorical_features(design_matrix):
    """Create dummies for categorical features"""
    feature_types = dict(design_matrix.dtypes)
    categorical_features = [feature for feature, type in feature_types
                            .iteritems() if type == 'object']
    design_matrix = pd.get_dummies(design_matrix, prefix=categorical_features,
                                   columns=categorical_features)
    return design_matrix


def _imputation_using_regression(design_matrix):
    missing_col = 'Age'
    predictors = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
    non_na_rows = design_matrix.dropna(subset=[missing_col])
    na_rows = design_matrix[design_matrix[missing_col].isnull()]

    non_na_rows = _create_dummies_for_categorical_features(non_na_rows)
    na_rows = _create_dummies_for_categorical_features(na_rows)

    predictors_after_dummy_creation = []
    # column names of categorical var have changed after dummy creation
    for pred in predictors:
        _ = [x for x in non_na_rows.columns if pred in x]
        predictors_after_dummy_creation.extend(_)

    for pred in predictors_after_dummy_creation:
        if pred not in na_rows.columns:
            na_rows[pred] = 0.

    model = LinearRegression(normalize=True)
    model.fit(non_na_rows[predictors_after_dummy_creation],
              non_na_rows['Age'])

    na_rows['Age'] = model.predict(
        na_rows[predictors_after_dummy_creation])

    # nom_na rows and na_rows now contain extra dummy variables, so can't
    # directly use their concatenated DF.
    design_matrix_helper = pd.concat([non_na_rows, na_rows], ignore_index=True)
    design_matrix = pd.merge(
        design_matrix, design_matrix_helper[['PassengerId', 'Age']],
        how='left', on='PassengerId')
    design_matrix.drop('Age_x', axis=1, inplace=True)
    design_matrix.rename(columns={'Age_y': 'Age'}, inplace=True)
    return design_matrix


def clean_design_matrix(design_matrix, train=False):
    """Clean/transform design matrix"""
    if train:
        design_matrix = design_matrix[
            ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
             'Parch', 'Fare', 'Embarked', 'Name']]
    else:
        design_matrix = design_matrix[
            ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
             'Embarked', 'Name']]
    design_matrix = _impute_missing_values(design_matrix)
    design_matrix = _convert_data_types(design_matrix)
    design_matrix = _capture_title_from_names(design_matrix)
    design_matrix = _create_dummies_for_categorical_features(design_matrix)
    return design_matrix
