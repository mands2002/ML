
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from datetime import date, timedelta
import itertools
import calendar
import sqlalchemy
from datetime import date
import calendar
from sqlalchemy import create_engine
import math

## define the engine to sql
engine = create_engine('mysql+pymysql://hamode:hamode@ic-phy-11.cisco.com:3306/north_central_att')

## calculate the KPI diff before/after an event
def intervalCalcDelta(data_orig, events, pm, func=np.mean, steps_before=96, steps_after=96,steps_skip=1):
    '''input| data: data frame, events: list of event times in datetime64 format in numpy array - df['time'].values, func: aggregated function, pm: target pm, steps before the event, steps after the events, steps skip for the after part..
    output| results df with event|before|after'''
    data = data_orig.reset_index()
    data = data.dropna(how='all', subset = [pm])
    data[pm] = pd.to_numeric(data[pm])
    results = pd.DataFrame(columns=['event', 'before','after'])

    for event in events:
        try:
            event_index = data[data['time'] == event].index.values[0]
            before_event = data.iloc[max(event_index-steps_before, 0):event_index,:][[pm]].apply(func).values[0]

            # adding handling of restart steps to the calc: skipping samples after the event where the kpi value is 0
            reset_steps = 0

            while True:
                if data[pm].iloc[event_index+reset_steps] == 0:
                    reset_steps += 1
                else:
                    steps_skip=max(reset_steps, steps_skip)
                    break

            after_event = data.iloc[event_index+steps_skip:min( (event_index+steps_skip+steps_after), len(data[pm]) ),:][[pm]].apply(func).values[0]
            diff = after_event - before_event
            percent = (after_event - before_event)/before_event
            result = {'event':event, 'before':before_event, 'after':after_event, 'diff': diff, 'percent': percent}
            results = results.append(result, ignore_index=True)
            
        except:
            print str.format("failed on event {}", event)
    return results

# find the valid cells per each event/PM pair
def findValidCells(df, event, pm):
    '''Finds the cells which have both events and PM values'''
    cell_list=df[df['newAttributeValues']==event]['name'].unique() # cell list for the current event
    return np.array(df[df['name'].isin(cell_list)].groupby('name')['name'].count().index)

def check_step(x):
    ''' add a +/- step calculation per each KPI percent'''
    x = x
    if x > 0.15:
        return 1
    elif x < -0.15:
        return -1
    else:
        return 0  

# run a single event/PM model loop and get the score/F
def pmEventScore(df, cell_list, event, pm, func=np.mean):
    '''runs a model for event/pm pair for all the relevant cells and returns the score (R^2), F, percision, recall, support per each run'''
    
    df2 = pd.DataFrame() #results from all the cells
    
    for cell in cell_list:

        df_cell = df[df['name']==cell]
        df_filtered = df_cell[df_cell['newAttributeValues']==event]
        
        event_list = df_filtered['time'].values
        events_df = df_filtered[df_filtered['time'].isin(event_list)][['time', 'value_0']]
        event_metadata = pd.DataFrame({'pm': pm, 'cell': cell, 'eventName': event, 'eventTime': event_list, 'eventValue': events_df['value_0'], 'day': df_filtered['day'], 'weekend': df_filtered['weekend']}).reset_index()

        cell_results = intervalCalcDelta(df_cell, event_list, pm, func, steps_before=96, steps_after=92, steps_skip=4)
        cell_results.columns = ['event', str.split(str(func))[1].capitalize() + 'Before', str.split(str(func))[1].capitalize() + 'After', str.split(str(func))[1].capitalize() + 'Diff', str.split(str(func))[1].capitalize() + 'Percent']
        cell_results = pd.concat([cell_results, event_metadata], axis=1)
        
        df2 = pd.concat([df2, cell_results])
    
    df2 = df2.reset_index()
    del df2['level_0']
    
    # generate pivot table
    df2 = pd.get_dummies(df2,columns=['eventName'], prefix='', prefix_sep="")
    for column in df2.columns[(len(df2.columns)-1):]:
        df2[column] = df2[df2[column] == 1][['eventValue']]
    
    # add the weekday and weekend as dummies:
    df2 = pd.get_dummies(df2, columns=['day', 'weekend'])
    
    # remove inf/nan in targets
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna(subset=['MeanDiff', 'MeanPercent'], how="any").reset_index()

    # fill the nan between events and calculate the diff between the current event value and the previous one
    df2['event_after_ffill']=df2.groupby('cell')['eventValue'].fillna(method='ffill')
    df2['eventValue_shifted'] = df2.groupby('cell')['event_after_ffill'].apply(lambda x: x.shift(+1))
    df2['event_diff']=df2.groupby('cell')['event_after_ffill'].fillna(0).astype(int)-df2.groupby('cell')['eventValue_shifted'].fillna(float('nan')).astype(float)
    df2 = df2.dropna(how='any', axis='rows')
    
    # add step as a target
    df2['stepMeanPercent'] = df2['MeanPercent'].apply(check_step).values
    
    # run a random forest model on features: event, event_delta, calculated_pm_diff, day | target: step
    features = [u'cellRange', u'event_diff',u'day_Friday',
       u'day_Monday', u'day_Saturday', u'day_Sunday', u'day_Thursday',
       u'day_Tuesday', u'day_Wednesday', u'weekend_no', u'weekend_yes']
    target = ['stepMeanPercent']
    
    X = df2[features]
    y = df2[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    RF = RandomForestClassifier(random_state=0, n_estimators=100)
    RF.fit(X_train,y_train)
    y_pred = RF.predict(X_test)
    y_pred = y_pred.reshape(-1,1)
    
    # output prints
    print(str.format( 'Train score: {}', RF.score(X_train, y_train)) )
    score = RF.score(X_test, y_test)
    print(str.format( 'Test score: {}',  score) )

    feature_importance = RF.feature_importances_
    print('\nfeature importance:')
    for i in range(0,len(features)):
        print features[i], feature_importance[i]
    
    # prediction and results
    prediction = pd.DataFrame.copy(y_test)
    prediction['stepMeanPercent_predict'] = y_pred
    
    p,r,f,s = precision_recall_fscore_support(prediction['stepMeanPercent'], prediction['stepMeanPercent_predict'])

    return score, f, p, r, s

# round the event to the next quarter hour
def ceil_dt(dt):
    # how many secs have passed this hour
    nsecs = dt.minute*60 + dt.second + dt.microsecond*1e-6  
    delta = math.ceil(nsecs / 900) * 900 - nsecs
    #time + number of seconds to quarter hour mark.
    return dt + timedelta(seconds=delta)


# create the DF_orig from the event,pm list
pm_list = ['cbra_discard_due_cell_range','dl_prb_utilization', 'dl_throughput_den', 'active_ue_dl_avg', 'erab_estab_succ_rate_num']
event_list = ['cellRange', 'freqBand', 'primaryPlmnReserved', 'physicalLayerCellIdGroup']
results_heatmap = pd.DataFrame()
results = pd.DataFrame()


## Pipe
''' event->pm->findValidCells->pmEventScore->update result heatmap '''

for event in event_list:
    
    with engine.connect() as con:
    
        print "performing events query"
        sql_events = 'SELECT eutrancellfdd, eventDetectionTimestamp, newAttributeValues, value_0' + ' FROM events_utrancellfdd_feb25' + ' WHERE newAttributeValues = \'' + event + '\''
        df_events = pd.read_sql(sql_events, con)
        event_cells = "\",\"".join(df_events['eutrancellfdd'].unique())
                
    for pm in pm_list:
        
        print str.format("event: {0},  pm: {1}", event, pm)
        
        try:
        
            #create the df_orig table for this pair
            with engine.connect() as con:
                print "performing pms query"
                sql_pm = 'SELECT name, start_time_utc, start_time, ' + pm + ' From pms WHERE start_time_utc > \'2018-01-05\' AND name in (\"' + event_cells + '\")'
                df_pm = pd.read_sql(sql_pm,con)
            
            # round the events time to the closest upper quarter hour
            print "rounding the event time"
            df_events['eventTimeRounded'] = pd.to_datetime(pd.to_datetime(df_events['eventDetectionTimestamp'].apply(ceil_dt)))
        
            # join the events and pms dataframes to create df_orig
            print "joining the pm and event dfs"
            df_orig = df_pm.merge(right=df_events, left_on=['start_time_utc', 'name'], right_on=['eventTimeRounded', 'eutrancellfdd'], how='left')
        
            # add days and weekend
            print "add days and weekend"
            df_orig['time'] = pd.to_datetime(df_orig['start_time_utc'])
            df_orig['time_orig'] = pd.to_datetime(df_orig['start_time'])
            df_orig['day'] = df_orig['time_orig'].apply(lambda x: calendar.day_name[x.weekday()]).astype('category')
            df_orig['weekend'] = df_orig['day'].apply(lambda x: 'yes' if( (x == 'Saturday') or (x == 'Sunday') ) else 'no' ).astype('category')
            
            # create the cell list
            print "creating a cell list"
            cell_list = findValidCells(df_orig, event, pm) 
            print str.format("participating cells: {}", len(cell_list))
        
            # find the pm/event test score
            print "executing the model and getting the score"
            score, f, p, r, s = pmEventScore(df_orig, cell_list, event, pm)
        
            # update the heatmap
            print "updating the heatmap"
            current_result = pd.DataFrame({'event': event, 'pm': pm, 'score': score}, index=range(1))
            results = pd.concat([results, pd.DataFrame( {'pm': pm, 'score': score, 'F_minus1': f[0], 'S_minus1': s[0], 'F_noStep': f[1], 'S_noStep': s[1], 'F_plus1': f[2], 'S_plus1': s[2]}, index=range(1) )])
            results_heatmap = pd.concat([results_heatmap, current_result], ignore_index=True)
        except:
            print "Failed"


