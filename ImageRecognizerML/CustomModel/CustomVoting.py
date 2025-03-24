import numpy as np


class CustomVoting:
    def __init__(self, models: list, weights: list = None):
        self.models = models
        self.fitted_models = []

        if weights is None:
            n_models = len(self.models)
            self.weights = np.ones(n_models) / n_models
        else:
            self.weights = weights
        
    def fit(self, train_datas: list[dict]):
        print('Training Models Started')

        for i, model in enumerate(self.models):
            X_train = train_datas[i]['X_train']
            y_train = train_datas[i]['y_train']
            model.fit(X_train, y_train)
            self.fitted_models.append(model)
            print(f'Training Model {i+1} Done')

        print('Training Models Finished')

    def predict(self, test_datas: list):
        print('Predicting Models Started')

        predicts = []

        for i, model in enumerate(self.fitted_models):
            X_test = test_datas[i]
            y_pred_proba = model.predict_proba(X_test)
            weighted_proba = self.weights[i] * y_pred_proba
            predicts.append(weighted_proba)

        predicts_proba = np.sum(predicts, axis=0)
        best_predict = np.argmax(predicts_proba, axis=1)

        print('Predicting Models Finished')

        return best_predict



