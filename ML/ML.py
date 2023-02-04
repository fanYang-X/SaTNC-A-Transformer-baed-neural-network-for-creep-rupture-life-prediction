import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb

param_dict = {'RandomForest': {'n_estimators': 585,
                               'max_features': 0.9106427119689653, 'max_depth': 12},
              'SVR': {'kernel': 'rbf', 'C': 3.23}, 
              'LGBMRegressor': {'n_estimators': 4378, 'subsample': 0.9800506703432134, 
                                'colsample_bytree':  0.9996391650494586, 'reg_alpha': 0.45034241712270956, 
                                'reg_lambda': 1.1276283910137448, 'min_child_samples': 37}}

def score(label_, pre):
    label_ = np.array([np.exp(i) - 1 for i in label_])
    pre = np.array([np.exp(i) - 1 for i in pre])
    mape = np.mean(abs(label_ - pre)/(label_ + 1))
    smape = 2.0*np.mean(np.abs(pre - label_)/(np.abs(pre) + np.abs(label_)))
    r2 = r2_score(label_, pre)
    return mape, smape, r2


class RegressionModel:
    def __init__(self, data_dict, param_dict=param_dict):
        """_RegressionModel on 5-CV_

        Args:
            data_dict (_Dict_): {fold: {train: (train_data, train_label), val: (val_data, val_label), ...}}
            param_dict (_Dict_, optional): _parameters for SVR, RF, LGB_. Defaults to param_dict.
        """
        self.data_dict = data_dict
        self.param_dict = param_dict
        
    def model_train(self):
        for clf_name in self.param_dict.keys():
            mape_list, smape_list, r2_list = [], [], []
            for fold_ in range(5):
                train_data, train_label = self.data_dict[fold_ + 1]['train'][0], self.data_dict[fold_ + 1]['train'][1]
                test_data, test_label = self.data_dict[fold_ + 1]['test'][0], self.data_dict[fold_ + 1]['test'][1]
                if clf_name == 'RandomForest':
                    clf = RandomForestRegressor(**self.param_dict[clf_name])
                elif clf_name == 'SVR':
                    clf = SVR(**self.param_dict[clf_name])
                else:
                    clf = lgb.LGBMRegressor(**self.param_dict[clf_name], silent=False, verbosity=-1)
                clf.fit(train_data, train_label)
                test_pre = clf.predict(test_data)
                fold_score = score(test_label, test_pre)
                mape_list.append(fold_score[0])
                smape_list.append(fold_score[1])
                r2_list.append(fold_score[2])
            print(clf_name + ' Test CV Results Are: ')
            print()
            print('====== CV Mean Score === Score Std ======')
            print('MAPE     {:4f}        {:4f}'.format(np.mean(mape_list), np.std(mape_list)))
            print('SMAPE    {:4f}        {:4f}'.format(np.mean(smape_list), np.std(smape_list)))
            print('R2       {:4f}        {:4f}'.format(np.mean(r2_list), np.std(r2_list)))
            print()