
import csv
import datetime
from os.path import join,abspath,curdir
import numpy as np
import pandas as pd

'''
Use this with 
import sys
sys.path.insert(0, './common/')
import csv_utils
csv_utils.create_csv(predicted, test_ids)

Given the predicted outputs for each model:
predicted = [[0,1,0,0,1,0],[0,1,0,1,1,0],[0,1,0,0,1,1]]
test_ids = [12,32,43,44,11]
Create the csv to submit to kaggle
'''

def create_csv(predicted, test_ids):
    EXPECTED_ROWS = 81126 
    tests_ids_len = len(test_ids)
    assert(tests_ids_len == EXPECTED_ROWS)
    for l in predicted:
        assert(len(l)==tests_ids_len)
    FILENAME = 'submission_' + datetime.datetime.now().strftime("%I%M%p-%B-%d-%Y") + '.csv'
    
    FOLDER = join('data', 'submissions')
    CURRENT_PATH = abspath(curdir)
    FULL_PATH = join(CURRENT_PATH, FOLDER, FILENAME)

    merged = {'ids': test_ids}
    for idx,v in enumerate(predicted):
        merged[str(idx)] = v
    
    df = pd.DataFrame.from_dict(merged)

    df.set_index('ids', inplace=True)

    df.to_csv(path_or_buf=FULL_PATH, sep=',')

    print('saved in: ', FULL_PATH)
