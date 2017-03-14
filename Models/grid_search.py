from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
import pandas as pd
from data_cleansing import clean_design_matrix


# Reading data
train_survived = pd.read_csv('../Data/train.csv')


class Model(object):

    def __init__(self):
        self.design_matrix = clean_design_matrix(train_survived, train=True)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in ['Survived']]

    @staticmethod
    def _define_regressor_and_parameter_candidates():
        """Define model fit function & parameters"""
        regressor = GradientBoostingClassifier(
            random_state=99, max_depth=3,
            min_samples_split=10, min_samples_leaf=10)
        parameters = {'n_estimators': range(10, 200, 10),
                      'learning_rate': [i/100.0 for i in range(1, 11)]}
        return regressor, parameters

    def grid_search_for_best_estimator(self):
        """Comprehensive search over provided parameters to find the best
        estimator"""
        regressor, parameters = self\
            ._define_regressor_and_parameter_candidates()
        model = GridSearchCV(regressor, parameters, cv=10, verbose=4,
                             scoring='accuracy', iid=False, n_jobs=8)
        model.fit(self.design_matrix[self.predictors],
                  self.design_matrix['Survived'])
        print model.best_params_
        print model.best_score_
        cv_results = model.cv_results_
        results = DataFrame.from_dict(cv_results, orient='columns')
        results.to_csv('../Model_results/GB_GridSearch2.csv',
                       index=False)

Model().grid_search_for_best_estimator()
