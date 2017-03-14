import pandas as pd
import numpy as np
from data_cleansing import clean_design_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# Reading training and test data
train_survived = pd.read_csv('../Data/train.csv')
test_survived = pd.read_csv('../Data/test.csv')


def _train_validation_split(design_matrix):
    """Split into training and validation sets"""
    train, validation = train_test_split(design_matrix, test_size=0.3)
    return train, validation


class BinomialLogRegression(object):

    def __init__(self):
        self.design_matrix = clean_design_matrix(train_survived, train=True)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in [
            'PassengerId', 'Survived']]

    @staticmethod
    def _build_model(design_matrix, predictors):
        """Building multi-class logistic model with default parameters using
        cross-validation."""
        model = LogisticRegression(solver='newton-cg')
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
        return model, predictions

    def submission(self):
        """Submitting solutions"""
        model, predictions = self._make_predictions()
        submission = test_survived[['PassengerId']]
        submission['Survived'] = predictions
        submission.to_csv('../Submissions/LogReg4_tuned_' + str(
            self._calculate_accuracy()) + '.csv', index=False)

BinomialLogRegression().submission()
