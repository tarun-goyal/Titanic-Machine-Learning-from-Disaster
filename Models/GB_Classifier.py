import pandas as pd
import numpy as np
from data_cleansing import clean_design_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier


# Reading training and test data
train_survived = pd.read_csv('../Data/train.csv')
test_survived = pd.read_csv('../Data/test.csv')


def _train_validation_split(design_matrix):
    """Split into training and validation sets"""
    train, validation = train_test_split(design_matrix, test_size=0.3)
    return train, validation


class GradientBoostingModel(object):

    def __init__(self):
        self.design_matrix = clean_design_matrix(train_survived, train=True)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in [
            'PassengerId', 'Survived']]

    @staticmethod
    def _build_model(design_matrix, predictors):
        """Building multi-class logistic model with default parameters using
        cross-validation."""
        model = GradientBoostingClassifier(
            n_estimators=80, learning_rate=0.05, min_samples_split=10,
            min_samples_leaf=10, max_depth=3)
        model.fit(design_matrix[predictors], design_matrix['Survived'])
        return model

    def _calculate_accuracy(self, iterations=10):
        """Compute accuracy score using cross-validations"""
        mean_accuracy = []
        for itr in range(iterations):
            training, validation = _train_validation_split(self.design_matrix)
            model = self._build_model(training, self.predictors)
            predicted = model.predict(validation[self.predictors])
            accuracy = accuracy_score(validation['Survived'], predicted)
            print 'Accuracy score: ', accuracy
            mean_accuracy.append(accuracy)
        print mean_accuracy
        print np.mean(mean_accuracy)
        return np.mean(mean_accuracy)

    def _make_predictions(self):
        """Make predictions on test data"""
        test_data = clean_design_matrix(test_survived)
        predictors = [pred for pred in self.predictors if pred in list(
            test_data.columns.values)]
        model = self._build_model(self.design_matrix, predictors)
        predictions = model.predict(test_data[predictors])
        feature_imp = pd.DataFrame(columns=predictors)
        feature_imp.loc[0] = model.feature_importances_
        return model, predictions, feature_imp

    def submission(self):
        """Submitting solutions"""
        model, predictions, feature_imp = self._make_predictions()
        submission = test_survived[['PassengerId']]
        submission['Survived'] = predictions
        feature_imp.to_csv('../Model_results/GB_FeatureImp3.csv', index=False)
        submission.to_csv('../Submissions/GB3_tuned_' + str(
            self._calculate_accuracy()) + '.csv', index=False)

GradientBoostingModel().submission()
