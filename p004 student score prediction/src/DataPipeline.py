""" Light data pipeline framework for machine learning.

Author: Zhang Zhou
Version: 0.24.4.10
"""

# library and environment setup
import numpy as np
import pandas as pd
import sqlite3
import pickle
from datetime import datetime
import optuna
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# suppress warning messages
import warnings
warnings.filterwarnings("ignore")

# base class: DataPipeline
class DataPipeline():
    """Framework of a basic ML data pipeline

    Attributes:
        trained (dict['name': bool]): if blueprint 'name' is trained or not.
        blueprints (dict['name': blueprint (dict[pipe_id (str): pipe_func (callable)])]):
            Designs of pipelines.
            only the 1st level pipes are recorded in sequence.
            Lower level pipes should be wrapped in higher level pipes.
            Use print_design method to show all level pipes.

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
    """
    def __init__(self, verbose=1, **kwargs):
        """Framework of a basic ML data pipeline

        Args:
            verbose (int, optional): level of print info. Defaults to 1.
                Set 0 to suppress all print info.
        """
        self.trained = dict()
        self.__verbose__ = verbose
        self.blueprints = dict()

        self.__pipe_registry__ = dict()
        # store training data (e.g. trained models, transformers, variables, etc.)
        # format: dict[('name', 'pipe_id'): dict[data]]
        # pipe_id format: 
        #   '0', '1', '2', ... are 1st level pipes
        #   '0.0' 2nd level pipe under pipe '0'

        return
    
    def __dummy_pipe__(self, pipe_id: str, pipe_buff: dict, pipe_mode: str, **kwargs):
        """Template for pipe function construction.

        Args:
            pipe_id (str): id assigned to this pipe in process.
            pipe_buff (dict): buffer through the pipelinedic
            pipe_mode (str): action mode (print, train, transform, predict, etc.)

        Returns:
            dict: pipe_buff
        """
        info = 'pipe description' # brief description of this pipe
        buff = pipe_buff
        if pipe_mode == 'print':
            self.__print_pipe_info__(pipe_id, info)
            child_pipes = [] # if this pipe call other pipes, add them to list in sequence
            child_id = self.__next_lvl_id__(pipe_id)
            for pipe in child_pipes:
                buff = pipe(pipe_id, buff, pipe_mode, **kwargs)
                child_id = self.__next_id__(child_id)
            return buff
        
        # pipe_mode != 'print'
        ## Log print: start pipe
        if self.__verbose__ > 0: # if 0, no print
            print(datetime.now())
            self.__print_pipe_info__(pipe_id, info, ': ')
        
        # TODO: add your code here...
        pass
        
        ## Log print: end pipe
        if self.__verbose__ > 0:
            print('Done!')
        
        return buff
    
    def add_pipe(self, pipe_func: callable, name = 'main', **kwargs):
        """Add one piece 1st level pipe component into a specific blueprint.

        Args:
            pipe_func (callable): e.g. self.__dummy_pipe__
            name (str, optional): name of the blueprint. Defaults to 'main'.
        """
        if name not in self.blueprints:
            self.trained[name] = False # this blueprint is not trained
            self.blueprints[name] = [('0', pipe_func)]
        else:
            last_id = self.blueprints[name][-1][0]
            pipe_id = self.__next_id__(last_id)
            self.blueprints[name].append(
                (pipe_id, pipe_func)
            )
        return
    
    def process(self, pipe_mode: str, name = 'main', **kwargs) -> dict:
        """Activate pipeline data flow according to specific blueprint and mode.

        Args:
            pipe_mode (str): action mode (print, train, transform, predict, etc.) 
            name (str, optional): name of the blueprint. Defaults to 'main'.
        
        Returns:
            dict: pipe_buff
        """
        buff = dict() # initiate pipe_buff
        for pipe_id, func in self.blueprints[name]:
            buff = func(pipe_id, buff, pipe_mode, name=name, **kwargs)
        return buff
    
    def fit(self, X=None, y=None, data_file=None, name='main',
            save_pickle=False, pickle_file=None,
            file_prefix='', **kwargs):
        """Train pipeline according to specific blueprint.

        Compatable to scikit-learn API.

        Args:
            X (DataFrame, optional): X data as DataFrame. Defaults to None.
            y (Series, optional): y data as Series. Defaults to None.
            data_file (str, optional): raw data file name including path. Defaults to None.
            name (str, optional): name of the blueprint. Defaults to 'main'.
            save_pickle (bool, optional): if need to save the pipeline as .pickle file or not. Defaults to False.
            pickle_file (str, optional): customized file name including path. Defaults to None.
            file_prefix (str, optional): customized file name prefix if pickle_file=None. Defaults to ''.
        
        Returns:
            class <DataPipeline>: self
        """
        # Log print: train start
        if self.__verbose__ > 0:
            print(datetime.now())
            print('Pipeline activated.')
            print('  blueprint = ' + name)
            print('  mode = train')

        pipe_mode = 'train' # put pipeline into 'train' mode
        pipe_buff = self.process(pipe_mode, name, X=X, y=y, data_file=data_file,
                                 **kwargs) # active data flow
        self.trained[name] = True # flag pipeline as trained

        # save pickle file
        if save_pickle == True: # user requests to save pipeline as .pickle file
            if pickle_file is not None: # user specify file path and name
                pickle.dump(self, open(pickle_file, 'wb'))
            else: # use default path and file name
                file_name = ('./output/train/' + file_prefix + '_' +
                             self.__class__.__name__ + '.pickle')
                pickle.dump(self, open(file_name, 'wb'))

        # Log print: train complete
        if self.__verbose__ > 0:
            print(datetime.now())
            print('Pipeline completed.')

        return self
    
    def transform(self, X=None, y=None, data_file=None, name='main',
                  save_output=False, output_file=None,
                  file_prefix='', **kwargs):
        """Transform data according to specific blueprint.

        Compatable to scikit-learn API.

        Args:
            X (DataFrame, optional): X data as DataFrame. Defaults to None.
            y (Series, optional): y data as Series, ignored in transform mode. Defaults to None.
            data_file (str, optional): raw data file name including path. Defaults to None.
            name (str, optional): name of blueprint. Defaults to 'main'.
            save_output (bool, optional): if transformed data need to be saved as .csv or not. Defaults to False.
            output_file (str, optional): customized output file name including path. Defaults to None.
            file_prefix (str, optional): default output file name prefix when output_file=None. Defaults to ''.

        Raises:
            NotImplementedError: this blueprint is not trained.

        Returns:
            DataFrame: transformed X data.
        """
        if self.trained[name] == False:
            raise NotImplementedError('Pipeline is NOT trained on this blueprint.')
        
        # Log print: transform start
        if self.__verbose__ > 0:
            print(datetime.now())
            print('Pipeline activated.')
            print('  blueprint = ' + name)
            print('  mode = transform')

        pipe_mode = 'transform' # put pipeline into 'transform' mode
        pipe_buff = self.process(pipe_mode, name, X=X, y=y, data_file=data_file,
                                 **kwargs) # active data flow
        
        # save output file
        if save_output == True: # user requests to save pipeline as .pickle file
            if output_file is not None: # user specify file path and name
                pipe_buff['X'].to_csv(output_file, index=False)
            else: # use default path and file name
                file_name = ('./output/transform/' + file_prefix + '_' +
                             self.__class__.__name__ + '.csv')
                pipe_buff['X'].to_csv(file_name, index=False)

        # Log print: transform complete
        if self.__verbose__ > 0:
            print(datetime.now())
            print('Pipeline completed.')

        return pipe_buff['X']
    
    def predict(self, X=None, y=None, data_file=None, name='main',
                save_output=False, output_file=None,
                file_prefix='', **kwargs):
        """Make prediction according to specific blueprint.

        Compatable to scikit-learn API.

        If input data include 'index' column, it will be included in the output .csv
        file, but not in the return data.

        Args:
            X (DataFrame, optional): X data as DataFrame. Defaults to None.
            y (Series, optional): y data as Series, ignored in predict mode. Defaults to None.
            data_file (str, optional): raw data file name including path. Defaults to None.
            name (str, optional): name of the blueprint. Defaults to 'main'.
            save_output (bool, optional): If need to save output as .csv file. Defaults to False.
            output_file (str, optional): output file name including path. Defaults to None.
            file_prefix (str, optional): prefix of default output file name if output_file=None. Defaults to ''.

        Raises:
            NotImplementedError: the blueprint is not trained.

        Returns:
            Series: y_pred
        """
        if self.trained[name] == False:
            raise NotImplementedError('Pipeline is NOT trained on this blueprint.')
        
        # Log print: predict start
        if self.__verbose__ > 0:
            print(datetime.now())
            print('Pipeline activated.')
            print('  blueprint = ' + name)
            print('  mode = predict')

        pipe_mode = 'predict' # put pipeline into 'predict' mode
        pipe_buff = self.process(pipe_mode, name, X=X, y=y, data_file=data_file,
                                 **kwargs) # active data flow
        
        # save output file
        if 'index' in pipe_buff:
            output = pd.concat([pipe_buff['index'], pipe_buff['y_pred']],
                               axis=1)
        else:
            output = pipe_buff['y_pred'].to_frame()
        if save_output == True: # user requests to save pipeline as .pickle file
            if output_file is not None: # user specify file path and name
                output.to_csv(output_file, index=False)
            else: # use default path and file name
                file_name = ('./output/predict/' + file_prefix + '_' +
                             self.__class__.__name__ + '.csv')
                output.to_csv(file_name, index=False)

        # Log print: predict complete
        if self.__verbose__ > 0:
            print(datetime.now())
            print('Pipeline completed.')

        return pipe_buff['y_pred']
    
    def print_design(self, name = 'main', **kwargs):
        """Print specific blueprint design.

        Args:
            name (str, optional): name of the blueprint. Defaults to 'main'.
        """
        pipe_mode = 'print' # put pipeline into 'print' mode
        self.process(pipe_mode, name, **kwargs) # active data flow
        return
    
    def __next_lvl_id__(self, pipe_id: str):
        """Return next level 1st pipe_id.

        Args:
            pipe_id (str): 

        Returns:
            str: next level 1st pipe_id.
        """
        return pipe_id + '.0'

    def __next_id__(self, pipe_id: str):
        """Return next pipe_id.

        Args:
            pipe_id (str): 

        Returns:
            str: next pipe_id
        """
        codes = pipe_id.split('.')
        r = codes[:-1] # ignore the last code first
        r.append(str(1 + int(codes[-1]))) # last code +1
        return '.'.join(r)
    
    def __print_pipe_info__(self, pipe_id: str, info: str, end: str | None = "\n"):
        """Print 'pipe_id + info' with proper indentation.

        Args:
            pipe_id (str): 
            info (str): description of the pipe.
            end (str | None, optional): string printed at the end. 
                Defaults to new line.
        """
        id_level = len(pipe_id.split('.')) # level of pipe_id
        indent = '  ' * (id_level - 1) # indent 2 whitespaces per level
        print(indent + pipe_id + ' ' + info, end=end)
        return
    
    def transfer_registry(self, name1: str, name2: str, 
                          ids1: list, ids2: list):
        """Transfer pipe training registry from one blueprint to another.

        * The purpose is to reuse training data from the first blueprint and
        avoid training of the second one, which will be flagged as trained
        after this pipe registry transfer process.

        * Make sure the transfer is between same pipe. You also need to know
        the exact pipe_id in both blueprints.

        Args:
            name1 (str): name of the source blueprint.
            name2 (str): name of the target blueprint.
            ids1 (list): list of pipe_id from source blueprint.
            ids2 (list): list of pipe_id from target blueprint.

        Raises:
            NotImplementedError: the first blueprint is not trained.
            Exception: pipe_id lists provided are not same length.
        """
        # blueprint name1 must be trained
        if self.trained[name1] == False:
            raise NotImplementedError(
                f'Blueprint {name1} is not trained.'
            )
        # ids1 & ids2 must be same length
        if len(ids1) != len(ids2):
            raise Exception('Source pipe_id list and target pipe_id list must be same length.')
        # copy pipe registry of blueprint name1 to name2
        for pipe_id1, pipe_id2 in zip(ids1, ids2):
            if (name1, pipe_id1) in self.__pipe_registry__:
                self.__pipe_registry__[(name2, pipe_id2)] = \
                    self.__pipe_registry__[(name1, pipe_id1)]
        # flag blueprint name2 as trained
        self.trained[name2] = True

# general class: ScorePipeline
class ScorePipeline(DataPipeline):
    """Pipeline for student score prediction project.

    - With pipe functions ready to build an end-to-end pipeline blueprint.
    - Default estimator: RandomForestRegressor
    - Default metric: root_mean_squared_error
    - You need to build blueprint by add_pipe before run any process
    - You can also inherit this class to build new pipes and blueprints

    Parent class:
        DataPipeline

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
    def __init__(
            self,
            verbose = 1, 
            random_state: int | None = None,
            **kwargs
    ):
        """Pipeline for student score prediction project.

        Args:
            verbose (int, optional): level of print info. Defaults to 1.
            random_state (int | None, optional): seed of random number.
                Defaults to None.
            estimator (class, optional): estimator with scikit-learn API.
                Defaults to RandomForestRegressor
            metric (callable, optional): metric function with scikit-learn API.
                Defaults to root_mean_squared_error
        """
        super().__init__(verbose, **kwargs)
        self.__kwargs__ = kwargs
        self.random_state = random_state
        self.score = dict() # {name: score} score of cv training for each blueprint
        return
    
    def __fill_null__(self, row, null_map, median_rate):
        """map null value in attendance_rate

        Args:
            row (pd.Series): one row of data
            null_map (dict): dict to map sleep_length to attendance_rate
            median_rate (float): median value of all attendance_rate

        Returns:
            float: attendance_rate data
        """
        if row.isna()['attendance_rate'] == True:
            sleep_length = row['sleep_length']
            if sleep_length in null_map:
                return null_map[sleep_length]
            else: # in case this train split don't have all sleep_length data
                return median_rate
        else:
            return row['attendance_rate']
    
    def data_input(self, pipe_id: str, pipe_buff: dict, pipe_mode: str, **kwargs):
        """Load raw data from arguments or data_file.

        This is a pipe function.

        Args:
            pipe_id (str): pipe_id assigned by process.
            pipe_buff (dict): buffer in this pipeline run.
            pipe_mode (str): mode of this pipeline process.

        Raises:
            ValueError: data is provided by both arguments and data_file.
            TypeError: data_file type is not supported.

        Returns:
            dict: updated pipe_buff
        """
        info = 'Load raw data' # brief description of this pipe
        buff = pipe_buff
        if pipe_mode == 'print':
            self.__print_pipe_info__(pipe_id, info)
            return
        
        # pipe_mode != 'print'
        ## Log print: start pipe
        if self.__verbose__ > 0: # if 0, no print
            print(datetime.now())
            self.__print_pipe_info__(pipe_id, info, ': start')
        
        X = kwargs['X']
        y = kwargs['y']
        data_file = kwargs['data_file']
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
            buff['index'] = data['index']
            data.drop(columns=['index'], inplace=True)
        
        buff['data'] = data
        
        ## Log print: end pipe
        if self.__verbose__ > 0:
            print('Done!')
        
        return buff

    def common_clean_FE(self, pipe_id: str, pipe_buff: dict, pipe_mode: str, **kwargs):
        """Common data cleaning + feature engineering.

        This is a pipe function.

        NOTE: 
            Null values in 'attendance_rate' is not filled due
            to data leaking concern.

        Args:
            pipe_id (str): pipe_id assigned by process.
            pipe_buff (dict): buffer in this pipeline run.
            pipe_mode (str): mode of this pipeline process.

        Returns:
            dict: updated pipe_buff
        """
        # pipe description
        info = 'Common data cleaning + feature engineering'
        buff = pipe_buff
        if pipe_mode == 'print':
            self.__print_pipe_info__(pipe_id, info)
            return
        
        # mode: 'train' | 'transform' | 'predict'
        ## log: start
        if self.__verbose__ > 0:
            print(datetime.now())
            self.__print_pipe_info__(pipe_id, info, ': ')
        
        # 'train' mode only: drop null in 'final_test'
        if pipe_mode == 'train':
            buff['data'].dropna(subset='final_test', ignore_index=True, inplace=True)
        
        # feature lists
        num_cols = [] # continuous data
        dis_cols = [] # discrete data
        cat_cols = [] # categorical data
        str_cols = [] # string data
        bin_cols = [] # binary data

        # add feature name to above lists based on dtype
        str_cols += buff['data'].select_dtypes(include='object').columns.to_list()
        dis_cols += buff['data'].select_dtypes(include='int64').columns.to_list()
        num_cols += buff['data'].select_dtypes(include='float64').columns.to_list()
        if pipe_mode == 'train':
            num_cols.remove('final_test') # remove target feature

        # string data cleaning
        for feature in str_cols:
            buff['data'][feature] = ( 
                buff['data'][feature]
                .str.strip() # trim white space
                .str.lower() # change to lower case
                .replace(r'\s+', ' ', regex=True) # trim duplicate white space inside string
            )
        
        # convert 'sleep_time' and 'wake_time' to numeric data
        def time2hr(time_str: str) -> float:
            """Convert time string to float representing hours.

            Args:
                time_str (str): in the form of '22:00'

            Returns:
                hour (float): in the range (-12.0, +12.0].
                    pm hours are negative. 
            """
            time = time_str.split(':')
            hour = time[0]
            min = time[1]
            hour = float(hour) + float(min)/60.
            if hour > 12: # convert pm hour to negative
                hour -= 24
            return hour

        for col in ['sleep_time', 'wake_time']: # convert to float
            buff['data'][col] = buff['data'][col].apply(time2hr)
            # put it to correct feature list
            num_cols.append(col)
            str_cols.remove(col)

        # convert 'tuition' values: 'yes', 'no' --> 'y', 'n'
        buff['data']['tuition'] = buff['data']['tuition'].str[0]

        # convert 'direct_admission' to binary
        buff['data']['direct_admission'] = buff['data']['direct_admission'] == 'yes'
        bin_cols.append('direct_admission')
        str_cols.remove('direct_admission')

        # convert 'tuition' to binary
        buff['data']['tuition'] = buff['data']['tuition'] == 'y'
        bin_cols.append('tuition')
        str_cols.remove('tuition')

        # convert 'learning_style' to binary
        # change name to 'learn_by_visual' for readability
        buff['data']['learn_by_visual'] = buff['data']['learning_style'] == 'visual'
        buff['data'].drop(columns=['learning_style'], inplace=True)
        bin_cols.append('learn_by_visual')
        str_cols.remove('learning_style')

        # convert 'gender' to binary
        # change name to 'is_male' for readability
        buff['data']['is_male'] = buff['data']['gender'] == 'male'
        buff['data'].drop(columns=['gender'], inplace=True)
        bin_cols.append('is_male')
        str_cols.remove('gender')

        # convert 'CCA' to category
        buff['data']['CCA'] = \
            pd.Categorical(values=buff['data']['CCA'], 
                           categories=['none', 'sports', 'clubs', 'arts'],
                           ordered=True) # order is assigned for convienence, not because it is ordinal
        cat_cols.append('CCA')
        str_cols.remove('CCA')

        # convert 'mode_of_transport' to category
        buff['data']['mode_of_transport'] = \
            pd.Categorical(values=buff['data']['mode_of_transport'], 
                           categories=['private transport', 'public transport', 'walk'],
                           ordered=True) # order is assigned for convienence, not because it is ordinal
        cat_cols.append('mode_of_transport')
        str_cols.remove('mode_of_transport')

        # convert 'bag_color' to category
        buff['data']['bag_color'] = \
            pd.Categorical(values=buff['data']['bag_color'], 
                           categories=['yellow', 'green', 'white', 'red', 'blue', 'black'],
                           ordered=True) # order is assigned for convienence, not because it is ordinal
        cat_cols.append('bag_color')
        str_cols.remove('bag_color')

        # 'train' mode only: drop duplicate rows based on 'student_id'
        if pipe_mode == 'train':
            buff['data'] = (
                buff['data']
                    .sort_values('attendance_rate') # make sure null values appear last
                    .drop_duplicates('student_id', ignore_index=True)
            )

        # drop 'student_id' column
        buff['data'].drop(columns=['student_id'], inplace=True)
        str_cols.remove('student_id')

        # numeric data cleaning
        # clean 'age' column and convert dtype to int
        def correct_age(x: float):
            """Correct age value and convert to integer"""
            if x < 0:
                y = x + 20
            elif x < 10:
                y = x + 10
            else:
                y = x
            return int(y)

        buff['data']['age'] = buff['data']['age'].apply(correct_age)
        # treat 'age' as discrete instead of continuous values
        dis_cols.append('age')
        num_cols.remove('age')

        # new features
        buff['data']['class_size'] = buff['data']['n_female'] + buff['data']['n_male']
        buff['data']['male_ratio'] = buff['data']['n_male'] / buff['data']['class_size']
        buff['data']['sleep_length'] = buff['data']['wake_time'] - buff['data']['sleep_time']
        num_cols = num_cols + ['class_size', 'male_ratio', 'sleep_length']

        # drop 'n_male' and 'n_female' columns
        buff['data'].drop(columns=['n_male', 'n_female'], inplace=True)
        num_cols.remove('n_male')
        num_cols.remove('n_female')

        # generate new features
        ## creat 'study_hrs'
        def create_study_hrs(h):
            """convert hours_per_week to study_hrs, for single value"""
            if h < 5: return 'short'
            elif h <= 10: return 'mid'
            elif h <= 15: return 'long'
            else: return 'longer'
        buff['data']['study_hrs'] = buff['data']['hours_per_week'].apply(create_study_hrs)
        buff['data']['study_hrs'] = ( # string --> category
            pd.Categorical(buff['data']['study_hrs'],
                           ['short', 'mid', 'long', 'longer'],
                           ordered=True)
        )
        cat_cols.append('study_hrs')
        ## creat 'attendance'
        def create_attendance(rate):
            """convert attendance_rate to attendance, for single value"""
            if rate < 50: return 'min'
            elif rate < 90: return 'low'
            else: return 'norm'
        buff['data']['attendance'] = buff['data']['attendance_rate'].apply(create_attendance)
        buff['data']['attendance'] = ( # string --> category
            pd.Categorical(buff['data']['attendance'],
                           ['min', 'low', 'norm'],
                           ordered=True)
        )
        cat_cols.append('attendance')
        ## creat 'bed_time'
        def create_bed_time(t):
            """convert sleep_time to bed_time, for single value"""
            if t < -0.5: return 'early'
            elif t <= 0: return 'norm'
            else: return 'late'
        buff['data']['bed_time'] = buff['data']['sleep_time'].apply(create_bed_time)
        buff['data']['bed_time'] = ( # string --> category
            pd.Categorical(buff['data']['bed_time'],
                           ['early', 'norm', 'late'],
                           ordered=True)
        )
        cat_cols.append('bed_time')
        ## creat 'enough_sleep'
        buff['data']['enough_sleep'] = buff['data']['sleep_length'] >= 7
        bin_cols.append('enough_sleep')

        # drop 'age' and 'bag_color' columns as EDA found them unimportant
        buff['data'].drop(columns=['age', 'bag_color'], inplace=True)

        # encode category features
        for feature in ['study_hrs', 'attendance', 'bed_time']:
            # ordinal encoded category features
            buff['data'][feature] = buff['data'][feature].cat.codes
        # one hot encoded category features
        buff['data'] = pd.concat(
            [buff['data'], 
            pd.get_dummies(buff['data'][['CCA', 'mode_of_transport']],
                            prefix=['CCA', ''])
            ],
            axis=1
        )
        buff['data'].drop(columns=['CCA', 'mode_of_transport'], inplace=True)

        # make sure self.data columns only keep what we want and in correct order
        feature_list = [
            'number_of_siblings',
            'direct_admission',
            'tuition',
            'hours_per_week',
            'attendance_rate',
            'sleep_time',
            'wake_time',
            'learn_by_visual',
            'is_male',
            'class_size',
            'male_ratio',
            'sleep_length',
            'study_hrs',
            'attendance',
            'bed_time',
            'enough_sleep',
            'CCA_none',
            'CCA_sports',
            'CCA_clubs',
            'CCA_arts',
            '_private transport',
            '_public transport',
            '_walk'
        ]
        if pipe_mode == 'train':
            # put target feature in the 1st colunm
            feature_list = ['final_test'] + feature_list
        buff['data'] = buff['data'][feature_list]
        
        ## log: end
        if self.__verbose__ > 0:
            print('Done!')
        
        return buff

    def cv_prepare(self, pipe_id: str, pipe_buff: dict, pipe_mode: str, **kwargs):
        """Prepare corss-validate datasets, including fill null.

        This is a pipe function.

        Args:
            pipe_id (str): pipe_id assigned by process.
            pipe_buff (dict): buffer in this pipeline run.
            pipe_mode (str): mode of this pipeline process.

        Returns:
            dict: updated pipe_buff
        """
        # pipe description
        info = 'Prepare corss-validate datasets'
        buff = pipe_buff
        if pipe_mode == 'print':
            self.__print_pipe_info__(pipe_id, info)
            return
        
        # pipe_mode != 'print'
        ## Log print: start pipe
        if self.__verbose__ > 0: # if 0, no print
            print(datetime.now())
            self.__print_pipe_info__(pipe_id, info, ': ')
        
        if pipe_mode in ['transform', 'predict']:
            pass # no action taken in these modes
        else: # 'train' mode
            kf = KFold(n_splits=kwargs['cv'], shuffle=True, random_state=self.random_state)
            kf.get_n_splits(buff['data'])
            buff['cv_data'] = {'X_train': [], 'y_train': [],
                               'X_test':  [], 'y_test':  []}
            for i, (train_index, test_index) in enumerate(kf.split(buff['data'])):
                train_data = buff['data'][buff['data'].index.isin(train_index)]
                test_data  = buff['data'][buff['data'].index.isin(test_index)]
                # generate map for null filling
                null_map = (
                    train_data # only use train_data to avoid data leak
                    .dropna()
                    .groupby('sleep_length')
                    ['attendance_rate']
                    .median()
                    .to_dict()
                )
                median_rate = train_data['attendance_rate'].median()
                # map nulls in train_data
                train_data['attendance_rate'] = \
                    train_data.apply(lambda x: self.__fill_null__(x, null_map, median_rate),
                                     axis=1)
                # map nulls in test_data
                test_data['attendance_rate'] = \
                    test_data.apply(lambda x: self.__fill_null__(x, null_map, median_rate),
                                     axis=1)
                # append to buff['cv_data']
                buff['cv_data']['X_train'].append(train_data.drop(columns=['final_test']))
                buff['cv_data']['y_train'].append(train_data['final_test'])
                buff['cv_data']['X_test'].append(test_data.drop(columns=['final_test']))
                buff['cv_data']['y_test'].append(test_data['final_test'])
        
        ## Log print: end pipe
        if self.__verbose__ > 0:
            print('Done!')
        
        return buff

    def fill_attendance_null(
            self, pipe_id: str, pipe_buff: dict, pipe_mode: str, **kwargs
            ):
        """Fill null values in 'attendance_rate'.

        This is a pipe function.

        This pipe should be placed after self.cv_parepare()
        to avoid data leaking.

        Args:
            pipe_id (str): pipe_id assigned by process.
            pipe_buff (dict): buffer in this pipeline run.
            pipe_mode (str): mode of this pipeline process.

        Returns:
            dict: updated pipe_buff
        """
        # pipe description
        info = 'Fill null values in attendance_rate'
        buff = pipe_buff
        if pipe_mode == 'print':
            self.__print_pipe_info__(pipe_id, info)
            return
        
        # pipe_mode != 'print'
        ## Log print: start pipe
        if self.__verbose__ > 0: # if 0, no print
            print(datetime.now())
            self.__print_pipe_info__(pipe_id, info, ': ')
        
        # get null_map
        if pipe_mode in ['transform', 'predict']:
            registry = self.__pipe_registry__[(kwargs['name'], pipe_id)]
            null_map = registry['null_map']
            median_rate = registry['median_rate']
        elif pipe_mode == 'train': # 'train' mode
            # calculate null_map
            null_map = (
                buff['data']
                .dropna()
                .groupby('sleep_length')
                ['attendance_rate']
                .median()
                .to_dict()
            )
            # calculate median_rate
            median_rate = buff['data'].dropna()['attendance_rate'].median()
            # put null_map & median_rate into pipe_registry
            self.__pipe_registry__[(kwargs['name'], pipe_id)] = {
                'null_map': null_map,
                'median_rate': median_rate
            }
        # fill null in attendance_rate
        buff['data']['attendance_rate'] = \
            buff['data'].apply(lambda x: self.__fill_null__(x, null_map, median_rate),
                               axis=1)
        
        # split X, y data
        if 'final_test' in buff['data'].columns:
            buff['X'] = buff['data'].drop(columns=['final_test'])
            buff['y'] = buff['data']['final_test']
        else:
            buff['X'] = buff['data']
            buff['y'] = None
        
        ## Log print: end pipe
        if self.__verbose__ > 0:
            print('Done!')
        
        return buff

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
            # specially treat 'bootstrap' as it determines if 
            # 'max_samples' should be included or not
            params['bootstrap'] = trial.suggest_categorical('bootstrap', [True, False])
            for parameter in hyper_params:
                if parameter == 'bootstrap': continue
                if (parameter == 'max_samples') and (params['bootstrap'] == False):
                    x1 = hyper_params[parameter][1]
                    x2 = hyper_params[parameter][2]
                    max_samples =  trial.suggest_float(parameter, x1, x2)
                    continue
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
        # NOTE: 'max_samples' cannot exist if 'boostrap' == False
        best_params = study.best_params.copy()
        if best_params['bootstrap'] == False:
            best_params.pop('max_samples')
        buff['estimator'] = estimator(**best_params)
        
        ## Log print: end pipe
        if self.__verbose__ > 0:
            self.__print_pipe_info__(pipe_id, info, ': Done!\n')
        
        return buff

    def final_model(self, pipe_id: str, pipe_buff: dict, pipe_mode: str, **kwargs):
        """Final model training or prediction.

        This is a pipe function.

        Args:
            pipe_id (str): pipe_id assigned by process.
            pipe_buff (dict): buffer in this pipeline run.
            pipe_mode (str): mode of this pipeline process.

        Returns:
            dict: updated pipe_buff
        """
        # pipe description
        info = 'Final model training or prediction'
        buff = pipe_buff
        if pipe_mode == 'print':
            self.__print_pipe_info__(pipe_id, info)
            return
        
        # pipe_mode != 'print'
        ## Log print: start pipe
        if self.__verbose__ > 0: # if 0, no print
            print(datetime.now())
            self.__print_pipe_info__(pipe_id, info, ': ')
        
        # TODO: add your code here...
        if pipe_mode == 'train':
            buff['estimator'].fit(buff['X'], buff['y'])
            # add the trained estimator to pipe registry
            self.__pipe_registry__[(kwargs['name'], pipe_id)] =\
                {'estimator': buff['estimator']}
        elif pipe_mode == 'transform':
            pass # no action in transform mode
        elif pipe_mode == 'predict':
            estimator = self.__pipe_registry__[(kwargs['name'], pipe_id)]['estimator']
            y_pred = estimator.predict(buff['X'])
            # y_pred is array now, convert to Series
            y_pred = pd.Series(y_pred, index=buff['X'].index, name='final_test')
            buff['y_pred'] = y_pred
        
        ## Log print: end pipe
        if self.__verbose__ > 0:
            print('Done!')
        
        return buff

#######################################################
# Debug & Testing Zone
if __name__ == '__main__':
    print(datetime.now())

#######################################################