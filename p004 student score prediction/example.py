"""Example command line interface to call and run data pipeline in terminal.

Author: Zhang Zhou
Version: 0.24.4.12
"""

# library and environment setup
from src.ScoreModels import *
import sys
import json
import pickle

if __name__ == '__main__':
    print('Welcome to Student Score Prediction ML pipeline interface!')
    print('Please choose ONE model from below list:')
    print('  1. Random Forest')
    print('  2. XGBoost')
    print('  3. LightGBM')
    print('  4. Linear: ElasticNet')
    print('  5. K Nearest Neighbors')

    # get input for model choice
    try:
        model_choice = int(input())
    except:
        print('Error: invalid choice.')
        sys.exit()
    
    if (model_choice < 1) or (model_choice > 5): # invalid choice
        print('Error: invalid choice.')
        sys.exit()
    
    print('Please choose ONE task from below list:')
    print('  1. Train model')
    print('  2. Make prediction with trained model')
    print('  3. Data transform with trained model')

    # get input for task mode
    try:
        task_mode = int(input())
    except:
        print('Error: invalid choice.')
        sys.exit()
    
    if (task_mode < 1) or (task_mode > 3): # invalid choice
        print('Error: invalid choice.')
        sys.exit()
    
    default_kwargs_files = {
        1: './params/RF.json',
        2: './params/XGB.json',
        3: './params/LGBM.json',
        4: './params/Linear.json',
        5: './params/KNN.json'
    }

    default_trained_models = {
        1: '_ScorePipelineRF.pickle',
        2: '_ScorePipelineXGB.pickle',
        3: '_ScorePipelineLGBM.pickle',
        4: '_ScorePipelineLinear.pickle',
        5: '_ScorePipelineKNN.pickle'
    }

    # get kwargs file
    print('Please provide path of *.json file containing parameters')
    print(f'(ENTER to choose default: {default_kwargs_files[model_choice]}):')
    kwargs_file = input()
    if kwargs_file == '':
        kwargs_file = default_kwargs_files[model_choice]
    
    # get kwargs from file
    kwargs = json.load(open(kwargs_file, 'r'))

    if task_mode in [2, 3]: # predict, transfrom
        # get pickle file
        print('Please provide path of *.pickle file containing trained model')
        print('(ENTER to choose default: ./output/train/' + 
              f'{kwargs["file_prefix"]}' + 
              f'{default_trained_models[model_choice]}):')
        pickle_file = input()
        if pickle_file == '':
            pickle_file = (
                './output/train/' + 
                f'{kwargs["file_prefix"]}' + 
                f'{default_trained_models[model_choice]}'
            )
        
        # get trained model
        model = pickle.load(open(pickle_file, 'br'))

        # activate task
        if task_mode == 2: # predict
            model.predict(**kwargs)
        else: # transform
            model.transform(**kwargs)
    else: # train
        # initiate model pipeline
        match model_choice:
            case 1: model = ScorePipelineRF(**kwargs)
            case 2: model = ScorePipelineXGB(**kwargs)
            case 3: model = ScorePipelineLGBM(**kwargs)
            case 4: model = ScorePipelineLinear(**kwargs)
            case 5: model = ScorePipelineKNN(**kwargs)
        model.fit(**kwargs)
    
    print('Task complete successfully!')

# EOF