""" End-to-end data pipelines for student_score_prediction project.

Author: Zhang Zhou
Version: 0.24.4.10
"""

# parent class
from src.DataPipeline import ScorePipeline

# library and environment setup
import numpy as np
import pandas as pd
import sqlite3
import pickle
from datetime import datetime
import optuna
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.inspection import permutation_importance

# specific classes: ScorePipeline[Model]

class ScorePipelineRF(ScorePipeline):
    """End-to-end data pipelines for student_score_prediction project.

    - Estimator: RandomForestRegressor
    - Metric: root_mean_squared_error

    'main' pipeline blueprint is established by default.

    Parent class:
        ScorePipeline

    Attributes:
        trained (dict['name': bool]): 
            flag blueprint 'name' is trained or not.
        blueprints (dict['name': blueprint (dict[pipe_id (str): pipe_func (callable)])]):
            Designs of pipelines.
            only the 1st level pipes are recorded in sequence.
            Lower level pipes should be wrapped in higher level pipes.
            Use print_design method to show all level pipes.
        score (dict['name': score (float)]):
            Metric score of specific blueprint.
            Only available after the blueprint is trained.
        random_state (int | None):
            Seed of random number.

    Methods:
        add_pipe:
            Add one piece 1st level pipe component into a specific blueprint.
        process:
            Activate pipeline data flow according to specific blueprint and mode.
        fit:
            Train pipeline according to specific blueprint.
        transform:
            Transform data according to specific blueprint.
        predict:
            Make prediction according to specific blueprint.
        print_design:
            Print specific blueprint design.
        transfer_registry:
            Transfer pipe training registry from one blueprint to another.
        permu_importance:
            Feature permutation importance.
            Features are according to raw data, not including engineered ones.

    Methods (pipe functions): to build blueprints:
        data_input:
            Load raw data from arguments or data_file.
        common_clean_FE:
            Common data cleaning + feature engineering.
            No data leaking activity.
        cv_prepare:
            Prepare corss-validate datasets, including fill null.
        fill_attendance_null:
            Fill null values in 'attendance_rate'.
        tune_parameter:
            Model hyper parameter tuning by optuna.
        final_model:
            Final model training or prediction.
    """
    def __init__(self, verbose = 1, random_state = None, **kwargs):
        """End-to-end data pipelines for student_score_prediction project.

        - Estimator: RandomForestRegressor
        - Metric: root_mean_squared_error

        'main' pipeline blueprint is established by default.

        Args:
            verbose (int, optional): level of print info. Defaults to 1.
            random_state (int | None, optional): seed of random number.
                Defaults to None.
        """
        super().__init__(
            verbose = verbose,
            random_state = random_state,
            estimator = RandomForestRegressor,
            metric = root_mean_squared_error
        )

        # build 'main' pipeline blueprint
        self.add_pipe(self.data_input)
        self.add_pipe(self.common_clean_FE)
        self.add_pipe(self.cv_prepare)
        self.add_pipe(self.fill_attendance_null)
        self.add_pipe(self.tune_parameter)
        self.add_pipe(self.final_model)
        
        return
    
    def permu_importance(self, X=None, y=None, data_file=None,
                         permu_importance_params: dict = {}, **kwargs):
        """Feature permutation importance according to  trained 'main' blueprint pipeline.

        - Features are from raw data.
        - Engineered features are not included

        Args:
            X (DataFrame, optional): X data as DataFrame. Defaults to None.
            y (Series, optional): y data as Series, ignored in predict mode. Defaults to None.
            data_file (str, optional): raw data file name including path. Defaults to None.
            permu_importance_params (dict, optional): 
                parameters to be passed to permutation_importance function. 
                Defaults to empty dict.

        Raises:
            NotImplementedError: the blueprint is not trained.
            ValueError: data is provided by both arguments and data_file.
            TypeError: data_file type is not supported.

        Returns:
            Series: permu_importance
        """
        if self.trained['main'] == False:
            raise NotImplementedError('Pipeline is NOT trained on this blueprint.')
        
        if (X is not None) and (data_file is not None):
            raise ValueError('Provide data by either (X, y) or data_file, not both.')
        
        if X is not None:
            data = pd.concat([X, y], axis=1)
        else: # read data from data_file
            file_type = data_file.split('.')[-1]
            if file_type not in ['db', 'csv']:
                raise TypeError(f'*.{file_type} file is not supported.')
            if file_type == 'csv':
                data = pd.read_csv(data_file)
            elif file_type == 'db':
                conn = sqlite3.connect(data_file) # connection to db
                query = 'SELECT * FROM score'
                data = pd.read_sql_query(query, conn) # DataFrame of raw data
                conn.close() # close the connection
        
        # check if 'index' column exist or not
        if 'index' in data.columns:
            data.drop(columns=['index'], inplace=True)
        
        # drop 'final_test' is null rows, reset index
        data.dropna(subset='final_test', inplace=True)
        data.reset_index(drop=True, inplace=True)

        # split X, y data
        X = data.drop(columns=['final_test'])
        y = data['final_test']

        # get permutation importance
        importance = permutation_importance(
            estimator = self,
            X = X,
            y = y,
            scoring = 'neg_root_mean_squared_error',
            random_state = self.random_state,
            **permu_importance_params
        )['importances_mean']

        # convert to pd.Series
        importance = pd.Series(
            data = importance,
            index = X.columns,
            name = 'Permutation_importance'
        ).sort_values() # sort values

        return importance

class ScorePipelineXGB(ScorePipeline):
    """End-to-end data pipelines for student_score_prediction project.

    - Estimator: XGBRegressor
    - Metric: root_mean_squared_error

    'main' pipeline blueprint is established by default.

    Parent class:
        ScorePipeline

    Attributes:
        trained (dict['name': bool]): 
            flag blueprint 'name' is trained or not.
        blueprints (dict['name': blueprint (dict[pipe_id (str): pipe_func (callable)])]):
            Designs of pipelines.
            only the 1st level pipes are recorded in sequence.
            Lower level pipes should be wrapped in higher level pipes.
            Use print_design method to show all level pipes.
        score (dict['name': score (float)]):
            Metric score of specific blueprint.
            Only available after the blueprint is trained.
        random_state (int | None):
            Seed of random number.

    Methods:
        add_pipe:
            Add one piece 1st level pipe component into a specific blueprint.
        process:
            Activate pipeline data flow according to specific blueprint and mode.
        fit:
            Train pipeline according to specific blueprint.
        transform:
            Transform data according to specific blueprint.
        predict:
            Make prediction according to specific blueprint.
        print_design:
            Print specific blueprint design.
        transfer_registry:
            Transfer pipe training registry from one blueprint to another.
        permu_importance:
            Feature permutation importance.
            Features are according to raw data, not including engineered ones.

    Methods (pipe functions): to build blueprints:
        data_input:
            Load raw data from arguments or data_file.
        common_clean_FE:
            Common data cleaning + feature engineering.
            No data leaking activity.
        cv_prepare:
            Prepare corss-validate datasets, including fill null.
        fill_attendance_null:
            Fill null values in 'attendance_rate'.
        tune_parameter:
            Model hyper parameter tuning by optuna.
        final_model:
            Final model training or prediction.
    """
    def __init__(self, verbose = 1, random_state = None, **kwargs):
        """End-to-end data pipelines for student_score_prediction project.

        - Estimator: XGBRegressor
        - Metric: root_mean_squared_error

        'main' pipeline blueprint is established by default.

        Args:
            verbose (int, optional): level of print info. Defaults to 1.
            random_state (int | None, optional): seed of random number.
                Defaults to None.
        """
        super().__init__(
            verbose = verbose,
            random_state = random_state,
            estimator = XGBRegressor,
            metric = root_mean_squared_error
        )

        # build 'main' pipeline blueprint
        self.add_pipe(self.data_input)
        self.add_pipe(self.common_clean_FE)
        self.add_pipe(self.cv_prepare)
        self.add_pipe(self.fill_attendance_null)
        self.add_pipe(self.tune_parameter)
        self.add_pipe(self.final_model)
        
        return
    
    def tune_parameter(self, pipe_id: str, pipe_buff: dict, pipe_mode: str, **kwargs):
        """Model hyper parameter tuning by optuna.

        This is a pipe function.

        Args:
            pipe_id (str): pipe_id assigned by process.
            pipe_buff (dict): buffer in this pipeline run.
            pipe_mode (str): mode of this pipeline process.

        Returns:
            dict: updated pipe_buff
        """
        # pipe description
        info = 'Model hyper parameter tuning'
        buff = pipe_buff
        if pipe_mode == 'print':
            self.__print_pipe_info__(pipe_id, info)
            return
        
        # pipe_mode != 'print'
        ## Log print: start pipe
        if self.__verbose__ > 0: # if 0, no print
            print(datetime.now())
            self.__print_pipe_info__(pipe_id, info, ': start\n')
        
        if pipe_mode in ['transform', 'predict']: # no action taken
            ## Log print: end pipe
            if self.__verbose__ > 0:
                self.__print_pipe_info__(pipe_id, info, ': Done!\n')
            return buff
        
        # 'train' mode only
        # estimator, metric
        if 'estimator' in self.__kwargs__:
            estimator = self.__kwargs__['estimator']
        else:
            estimator = RandomForestRegressor
        if 'metric' in self.__kwargs__:
            metric = self.__kwargs__['metric']
        else:
            metric = root_mean_squared_error
        
        # def objective func for optuna
        def objective(trial):
            """objective func for optuna"""
            hyper_params = kwargs['hyper_params']
            params = dict()
            for parameter in hyper_params:
                match hyper_params[parameter][0]:
                    case 'int':
                        x1 = hyper_params[parameter][1]
                        x2 = hyper_params[parameter][2]
                        params[parameter] = trial.suggest_int(parameter, x1, x2)
                    case 'float':
                        x1 = hyper_params[parameter][1]
                        x2 = hyper_params[parameter][2]
                        params[parameter] = trial.suggest_float(parameter, x1, x2)
                    case 'categorical':
                        x = hyper_params[parameter][1]
                        params[parameter] = trial.suggest_float(parameter, x)
                    case 'const':
                        x = hyper_params[parameter][1]
                        params[parameter] = x

            model = estimator(**params)
            scores = []
            for i in range(len(buff['cv_data']['X_train'])):
                X_train = buff['cv_data']['X_train'][i]
                y_train = buff['cv_data']['y_train'][i]
                X_test  = buff['cv_data']['X_test'][i]
                y_test  = buff['cv_data']['y_test'][i]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores.append(metric(y_test, y_pred))
            
            return np.mean(scores)
        
        # create a study object
        study = optuna.create_study(direction='minimize')

        # Optimize the objective function
        optimize_params = kwargs['optimize_params']
        study.optimize(objective, **optimize_params)

        ## Log print: optimize result
        if self.__verbose__ > 0:
            print('best params =')
            print(study.best_params)
            print(f'best score = {study.best_value}')

        # save cv score to self.score
        self.score[(kwargs['name'])] = study.best_value
        
        # save study to pipe registry and buffer
        self.__pipe_registry__[(kwargs['name'], pipe_id)] = {'study': study}
        buff['study'] = study

        # update buff['estimator'] with best params
        best_params = study.best_params.copy()
        buff['estimator'] = estimator(**best_params)
        
        ## Log print: end pipe
        if self.__verbose__ > 0:
            self.__print_pipe_info__(pipe_id, info, ': Done!\n')
        
        return buff
    
    permu_importance = ScorePipelineRF.permu_importance

### TODO: constructing...
class ScorePipelineLGBM(ScorePipeline):
    def __init__(self, verbose = 1, random_state = None, **kwargs):
        """End-to-end data pipelines for student_score_prediction project.

        - Estimator: LGBMRegressor
        - Metric: root_mean_squared_error

        'main' pipeline blueprint is established by default.

        Args:
            verbose (int, optional): level of print info. Defaults to 1.
            random_state (int | None, optional): seed of random number.
                Defaults to None.
        """
        super().__init__(
            verbose = verbose,
            random_state = random_state,
            estimator = LGBMRegressor,
            metric = root_mean_squared_error
        )

        # build 'main' pipeline blueprint
        self.add_pipe(self.data_input)
        self.add_pipe(self.common_clean_FE)
        self.add_pipe(self.cv_prepare)
        self.add_pipe(self.fill_attendance_null)
        self.add_pipe(self.tune_parameter)
        self.add_pipe(self.final_model)
        
        return
    
    tune_parameter = ScorePipelineXGB.tune_parameter
    permu_importance = ScorePipelineRF.permu_importance

### TODO: constructing...
class ScorePipelineKNN(ScorePipeline):
    def __init__(self, **kwargs):
        """Data pipeline for student_score_prediction with XGBoost model
        """
        super().__init__(**kwargs)
        pass

### TODO: constructing...
class ScorePipelineLinear(ScorePipeline):
    def __init__(self, **kwargs):
        """Data pipeline for student_score_prediction with XGBoost model
        """
        super().__init__(**kwargs)
        pass