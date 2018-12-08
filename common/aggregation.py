import numpy as np

def aggregateAlbert(predictions_df, kaggleScores):
    df = predictions_df.copy()
    scores = kaggleScores

    _checkParameters(df, scores)
    allColumnsButIdWinner = list(set(df.columns)-set(['test_id']))
    normalizedKaggleScores = [float(i)/sum(kaggleScores) for i in kaggleScores]
    df['winner'] = df.apply(lambda row: row[np.random.choice(  allColumnsButIdWinner,   1,  p=normalizedKaggleScores)[0]], axis=1)
    return df[['test_id', 'winner']]
    
        
#(0.78*model1+0.68*model2+0.63*model3...)/numero de models            
def aggregateXavi(predictions_df, kaggleScores):
    df = predictions_df.copy()
    scores = kaggleScores

    num_models = len(kaggleScores)
    allColumnsButIdWinner = list(set(df.columns)-set(['test_id']))
    _checkParameters(df, scores)
    df['winner'] = df.apply(lambda row: _calculateTotalXavi(row, num_models, allColumnsButIdWinner, scores), axis=1)
    return df[['test_id', 'winner']]

def _calculateTotalXavi(row, num_models, allColumnsButIdWinner, scores):
    total = 0
    j = 0
    for columnName in allColumnsButIdWinner:
        score = scores[j]
        prediction = row[columnName]
        total += score*prediction
        j+=1
    return 1 if total/num_models > 0.5 else 0


def _checkParameters(predictions_df, kaggleScores):
    EXPECTED_ROWS = 81126 
    assert(EXPECTED_ROWS == predictions_df.shape[0])
    assert(len(set(predictions_df.columns)-set(['test_id'])) == len(kaggleScores))