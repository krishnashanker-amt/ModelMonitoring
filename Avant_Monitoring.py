###############################################################################
#Library Imports
###############################################################################

import pandas as pd
import numpy as np
import trellis
import os
from avant_python_utils.email import send_email
from datalaketools.connectors.presto_db import PrestoDB
presto = PrestoDB()
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, f1_score,recall_score,precision_score, average_precision_score
from datetime import date, timedelta, datetime

#importing functions and variables for creating monitoring datasets
import model_monitoring_modules as mmm

#modules to write to google sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials

###############################################################################
#Constants Definition
###############################################################################

# Avant prod details
connection_name = 'us_fraud_follower'
sheet_key = '10aJZFUxDhEoa1uBw47sB0SS7Re5QQ5a3twujraopm84'
google_key_file = 'service_key.json'


#CONSTANTS
#global variables - these store column names that will be used in functions below
SCORE_COL = 'score_5'
YPRED_COL = 'prediction'
YTRUE_COL = 'suspected_fraud'
TIME_COL = 'created_date'
WEEKSTART_COL = 'created_week'
#TIME_COL = 'created_at'
AMOUNT_COL = 'loan_amount'
THRESHOLD = 0.05
MODEL_START_DATE = '2018-09-15'

#Google Sheet names
BASELINE_WORKSHEET = 'Baselines Data'
WEEKLY_WORKSHEET = 'Charts Data'
TABLES_WORKSHEET = 'Tables Data'

###############################################################################
#Function Definitions
###############################################################################

#list of functions to do everything
#SQL Query to pull base table data
base_table_query = """
SELECT
  l.id as loan_id
, l.created_date
, date_trunc('week', l.created_date) as created_week
, l.status
, case when l.status in ('current','late','paid_off','charged_off') then 1 else 0 end as issued
, case when c.high_confidence_fraud_indicator=true or cfl.id is not null then 1 else 0 end as high_confidence_fraud_indicator
, case when cfr.customer_id is not null then 1 else 0 end as suspected_fraud 
--, cfrt.name as fraud_reason
, cast(fd.score_5_old as double) as score_5_old
, cast(fd.score_5_new as double) as score_5_new
, coalesce(cast(fd.score_5_old as double), cast(fd.score_5_new as double)) as score_5
, l.state
, l.payment_method
, l.loan_amount
, ca.product_type

FROM avant.dw.customer_applications ca
LEFT JOIN avant.dw.loans l on l.customer_application_id = ca.id
JOIN avant.dw.customers c
  ON c.id = l.customer_id
  
  -- getting dependent variable
  
LEFT JOIN (
select customer_id 
from avant.avant_basic.customer_fraud_reasons cfr 
group by 1
) cfr on c.id = cfr.customer_id
  
 -- LEFT JOIN avant.avant_basic.customer_fraud_reason_types cfrt on cfr.customer_fraud_reason_type_id = cfrt.id
  
  -- getting fraud scores
LEFT JOIN (
  SELECT
    l.id as loan_id
  , json_extract_scalar(fd.model_scores, '$["fraud/en-US/4.1.0"]["score"]') as score_4
  , json_extract_scalar(fd.model_scores, '$["fraud/en-US/5.0.0"]["score"]') as score_5_old
  , json_extract_scalar(fd.model_scores, '$["fraud/en-US/5.0.0/avant"]["score"]') as score_5_new
  
  , fd.id as fraud_decision_id
  , row_number() over (partition by l.id order by fd.created_at desc) as row_num
  FROM avant.dw.loans l
  JOIN avant.avant_basic.fraud_decisions fd
    ON fd.customer_application_id = l.customer_application_id
    AND fd.created_at AT TIME ZONE 'America/Chicago' >= l.created_date
WHERE l.created_date > date '{START_DATE}'
) fd 
  ON fd.loan_id = l.id 
  AND fd.row_num=1
  -- getting fraud indicator
LEFT JOIN avant.avant_basic.confirmed_fraud_logs cfl 
  ON cfl.customer_id = c.id
  
    -- filtering for valid loans to evaluate performance on
  -- JOIN avant.dw.loan_performance_by_installment lp 
  -- ON lp.loan_id = l.id 
  -- AND lp.installment_number = 1
  -- AND lp.installment_date <= date_add('day', -64, current_timestamp)
  
  
WHERE l.created_date > date '{START_DATE}'
""".format(START_DATE = MODEL_START_DATE)


def base_table_creator(query = base_table_query):
    df_raw = presto.execute_df(base_table_query)
    #df_raw = pd.read_sql(query, connector)
    df = df_raw[df_raw[SCORE_COL].notnull()]
    df[YPRED_COL] = np.where(df[SCORE_COL] > THRESHOLD, 1, 0)
    return df
    

#Get monitoring metrics for each week
def weekly_evaluator(dframe, ytrue = YTRUE_COL, ypred = YPRED_COL, scores = SCORE_COL, amount = AMOUNT_COL):
    
    #false positives, true negatives for false positive rate
    true_positives = (dframe[ytrue] * dframe[ypred]).sum()
    false_positives = ((1-dframe[ytrue]) * dframe[ypred]).sum()
    false_negatives =  (dframe[ytrue] * (1-dframe[ypred])).sum()
    true_negatives = ((1-dframe[ytrue]) * (1-dframe[ypred])).sum()
    #calculating multiple metrics
    precision = precision_score(y_true = dframe[ytrue], y_pred = dframe[ypred], pos_label = 1, zero_division = 0)
    recall = recall_score(y_true = dframe[ytrue], y_pred = dframe[ypred], pos_label = 1, zero_division = 0)
    false_positive_rate = false_positives/(false_positives+true_negatives)
    f1score = f1_score(y_true = dframe[ytrue], y_pred = dframe[ypred], pos_label = 1)
    auc_pr = average_precision_score(y_true = dframe[ytrue], y_score = dframe[scores], pos_label=1)
    fraudmissed_dollar = (dframe[amount]*dframe[ytrue]*(1-dframe[ypred])).sum()
    fraudmissed_dollar_rate = 100*(dframe[amount]*dframe[ytrue]*(1-dframe[ypred])).sum()/(dframe[amount]*dframe[ytrue]).sum()
    fraud_rate = dframe[ytrue].sum()/len(dframe.index)
    avg_score = dframe[scores].sum()/len(dframe.index)
    try:
        auc_roc = roc_auc_score(y_true = dframe[ytrue], y_score = dframe[scores])
    except ValueError:
        auc_roc = ""

    
    
    return pd.Series({'precision': precision, 'recall': recall, 'f1score': f1score, 'auc_pr':auc_pr, 'auc_roc':auc_roc,
                     'fraud_rate': fraud_rate, 'avg_score': avg_score,'false_positive_rate':false_positive_rate,
                      'fraudmissed_dollar': fraudmissed_dollar,'fraudmissed_dollar_rate':fraudmissed_dollar_rate})

#function to create metric values for tables in Google Sheets
#function to create metric values for tables in Google Sheets
def values_for_cells(dframe, ytrue = YTRUE_COL, ypred = YPRED_COL, scores = SCORE_COL, timecol = TIME_COL, amount = AMOUNT_COL):
   
    #Setting up variables with different date values
    
    model_start_date = min(dframe[TIME_COL])
    model_start_date = datetime.strptime(model_start_date, '%Y-%m-%d %H:%M:%S.%f').date()
    today_date = date.today().strftime("%Y-%m-%d")
    prev30_date = (date.today() - timedelta(days = 30)).strftime("%Y-%m-%d")
    prev60_date = (date.today() - timedelta(days = 60)).strftime("%Y-%m-%d")
    
    modeltrain_date_start = (model_start_date + timedelta(days = 30)).strftime("%Y-%m-%d")
    modeltrain_date_end = (model_start_date + timedelta(days = 60)).strftime("%Y-%m-%d")
   
    #creating different datasets for the different time periods

    data_overall = dframe.query('{0} > @modeltrain_date_start'.format(TIME_COL))
    data_last30 = dframe.query('{0} > @prev30_date & {0} < @today_date'.format(TIME_COL))
    data_prev30 = dframe.query('{0} > @prev60_date & {0} < @prev30_date'.format(TIME_COL))   

    #PRECISION
    precision_last30 = precision_score(y_true = data_last30[ytrue], y_pred = data_last30[ypred], pos_label = 1)
    precision_overall = precision_score(y_true = data_overall[ytrue], y_pred = data_overall[ypred], pos_label = 1)
    precision_prev30 = precision_score(y_true = data_prev30[ytrue], y_pred = data_prev30[ypred], pos_label = 1)

    #recall values
    recall_last30 = recall_score(y_true = data_last30[ytrue], y_pred = data_last30[ypred], pos_label = 1)
    recall_overall = recall_score(y_true = data_overall[ytrue], y_pred = data_overall[ypred], pos_label = 1)
    recall_prev30 = recall_score(y_true = data_prev30[ytrue], y_pred = data_prev30[ypred], pos_label = 1) 
    
    #False positive rates
    fp_last30 = ((1-data_last30[ytrue]) * data_last30[ypred]).sum()/(((1-data_last30[ytrue]) * data_last30[ypred]).sum() + ((1-data_overall[ytrue]) * (1-data_overall[ypred])).sum())
    fp_overall = ((1-data_overall[ytrue]) * data_overall[ypred]).sum()/(((1-data_overall[ytrue]) * data_overall[ypred]).sum() + ((1-data_overall[ytrue]) * (1-data_overall[ypred])).sum())
    fp_prev30 = ((1-data_prev30[ytrue]) * data_prev30[ypred]).sum()/(((1-data_prev30[ytrue]) * data_prev30[ypred]).sum() + ((1-data_prev30[ytrue]) * (1-data_prev30[ypred])).sum())

    #F1 score
    f1_last30 = f1_score(y_true = data_last30[ytrue], y_pred = data_last30[ypred], pos_label = 1)
    f1_overall = f1_score(y_true = data_overall[ytrue], y_pred = data_overall[ypred], pos_label = 1)
    f1_prev30 = f1_score(y_true = data_prev30[ytrue], y_pred = data_prev30[ypred], pos_label = 1) 

    #auc pr
    aucpr_last30 = average_precision_score(y_true = data_last30[ytrue], y_score = data_last30[scores], pos_label = 1)
    aucpr_overall = average_precision_score(y_true = data_overall[ytrue], y_score = data_overall[scores], pos_label = 1)
    aucpr_prev30 = average_precision_score(y_true = data_prev30[ytrue], y_score = data_prev30[scores], pos_label = 1) 

    #auc roc
    aucroc_last30 = roc_auc_score(y_true = data_last30[ytrue], y_score = data_last30[scores])
    aucroc_overall = roc_auc_score(y_true = data_overall[ytrue], y_score = data_overall[scores])
    aucroc_prev30 = roc_auc_score(y_true = data_prev30[ytrue], y_score = data_prev30[scores]) 

    #TODO - Confirm fraud rate definition
    #fraud rate
    fraudrate_last30 = data_last30[ytrue].sum()/len(data_last30.index)
    fraudrate_overall = data_overall[ytrue].sum()/len(data_overall.index)
    fraudrate_prev30 = data_prev30[ytrue].sum()/len(data_prev30.index)
    
    #avg score
    avgscore_last30 = data_last30[scores].sum()/len(data_last30.index)
    avgscore_overall = data_overall[scores].sum()/len(data_overall.index)
    avgscore_prev30 = data_prev30[scores].sum()/len(data_prev30.index)

    #$ value of fraud missed
    fraudmissed_dollar_last30 = (data_last30[amount]*data_last30[ytrue]*(1-data_last30[ypred])).sum()
    fraudmissed_dollar_overall = (data_overall[amount]*data_overall[ytrue]*(1-data_overall[ypred])).sum()
    fraudmissed_dollar_prev30 = (data_prev30[amount]*data_prev30[ytrue]*(1-data_prev30[ypred])).sum()
    
    # $ value fraud rate
    fraudmissed_dollar_rate_last30 = 100*(data_last30[amount]*data_last30[ytrue]*(1-data_last30[ypred])).sum()/(data_last30[amount]*data_last30[ytrue]).sum()
    fraudmissed_dollar_rate_overall = 100*(data_overall[amount]*data_overall[ytrue]*(1-data_overall[ypred])).sum()/(data_overall[amount]*data_overall[ytrue]).sum()
    fraudmissed_dollar_rate_prev30 = 100*(data_prev30[amount]*data_prev30[ytrue]*(1-data_prev30[ypred])).sum()/(data_prev30[amount]*data_prev30[ytrue]).sum()

    

    output = {"metric": ['precision', 'recall','f1score', 'auc_pr', 'auc_roc', 'fraudrate', 'avg_score', 'falsepositive_rate', 'fraudmissed_dollar', 'fraudmissed_dollar_rate'],
             "current_values":[precision_last30, recall_last30, f1_last30, aucpr_last30, aucroc_last30, fraudrate_last30, avgscore_last30, fp_last30, fraudmissed_dollar_last30, fraudmissed_dollar_rate_last30],
             "initial_values":[precision_overall, recall_overall, f1_overall, aucpr_overall, aucroc_overall, fraudrate_overall, avgscore_overall, fp_overall, fraudmissed_dollar_overall, fraudmissed_dollar_rate_overall],
             "prev30_values":[precision_prev30, recall_prev30, f1_prev30, aucpr_prev30, aucroc_prev30, fraudrate_prev30, avgscore_prev30, fp_prev30, fraudmissed_dollar_prev30, fraudmissed_dollar_rate_prev30]}    
        
    return output



#Function to create baseline data that will be used in charts
def create_baseline_data(dframe, ytrue = YTRUE_COL, ypred = YPRED_COL, scores = SCORE_COL, timecol = TIME_COL, amount = AMOUNT_COL):
    #Setting up variables with different date values
    modeltrain_date_start = datetime.strptime(MODEL_START_DATE, "%Y-%m-%d")
    modeltrain_date_end = (modeltrain_date_start + timedelta(days = 60)).strftime("%Y-%m-%d")
    
    #creating different datasets for the different time periods
    
    #dataset 1 - 60 days after model was trained
    data_first30 = dframe.query('{0} > @MODEL_START_DATE & {0} < @modeltrain_date_end'.format(TIME_COL))
    
    #PRECISION
    precision_initial = precision_score(y_true = data_first30[ytrue], y_pred = data_first30[ypred], pos_label = 1)
    
    #recall values
    recall_initial = recall_score(y_true = data_first30[ytrue], y_pred = data_first30[ypred], pos_label = 1)
    
    #False positive rate values
    fp_initial = ((1-data_first30[ytrue]) * data_first30[ypred]).sum()/(((1-data_first30[ytrue]) * data_first30[ypred]).sum() + ((1-data_first30[ytrue]) * (1-data_first30[ypred])).sum())
    
    #F1 score
    f1_initial = f1_score(y_true = data_first30[ytrue], y_pred = data_first30[ypred], pos_label = 1)
    
    #auc pr
    aucpr_initial = average_precision_score(y_true = data_first30[ytrue], y_score = data_first30[scores], pos_label = 1)
    
    #auc roc
    aucroc_initial = roc_auc_score(y_true = data_first30[ytrue], y_score = data_first30[scores])
    
    #TODO - Confirm fraud rate definition
    #fraud rate
    fraudrate_initial = data_first30[ytrue].sum()/len(data_first30.index)
    
    #avg score
    avgscore_initial = data_first30[scores].sum()/len(data_first30.index)

    #TODO - Confirm fraud missed definition
    #fraud rate with dollar values
    fraudmissed_dollar_initial = (data_first30[amount]*data_first30[ytrue]*(1-data_first30[ypred])).sum()

    #$ value of fraud missed
    fraudmissed_dollar_rate_initial = 100*(data_first30[amount]*data_first30[ytrue]*(1-data_first30[ypred])).sum()/(data_first30[amount]*data_first30[ytrue]).sum()
    
    #creating grouped by data frame with needed weeks
    baseline_dataframe = pd.DataFrame(dframe[WEEKSTART_COL].unique()).rename(columns={0: WEEKSTART_COL}).sort_values(by = WEEKSTART_COL)
    baseline_dataframe = baseline_dataframe.assign(precision_baseline = precision_initial,
                              recall_baseline = recall_initial, 
                              f1_baseline = f1_initial, 
                              aucpr_baseline = aucpr_initial,
                              aucroc_baseline = aucroc_initial,
                              fraudrate_baseline = fraudrate_initial,
                              avgscore_baseline = avgscore_initial,
                              falsepositive_rate_baseline =  fp_initial,                     
                              fraudmissed_dollar_baseline = fraudmissed_dollar_initial,
                              fraudmissed_dollar_rate_baseline = fraudmissed_dollar_rate_initial
                              )
    
    return baseline_dataframe
    
    

def sheets_updater(workbook_key, google_key_file, byweek_dataset, tables_dataset, baselines_dataset):
    
    #authorization
    keys = json.dumps(trellis.keys('amount_drive_details'))
    loaded_keys = json.loads(keys)
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials._from_parsed_json_keyfile(loaded_keys, scope)
    gc = gspread.authorize(credentials)
    
    
    workbook = gc.open_by_key(workbook_key)
    
    #opening worksheets
    weekly_worksheet = workbook.worksheet(WEEKLY_WORKSHEET)
    tablesdata_worksheet = workbook.worksheet(TABLES_WORKSHEET)
    baselinedata_worksheet = workbook.worksheet(BASELINE_WORKSHEET)
    
    #clearing worksheets
    weekly_worksheet.clear()
    tablesdata_worksheet.clear()
    baselinedata_worksheet.clear()
    
    #updating worksheets  (first list out columns, and then add values for each column)
    weekly_worksheet.update([byweek_dataset.columns.values.tolist()] + byweek_dataset.values.tolist())
    tablesdata_worksheet.update([tables_dataset.columns.values.tolist()] + tables_dataset.values.tolist())
    baselinedata_worksheet.update([baselines_dataset.columns.values.tolist()] + baselines_dataset.values.tolist())
    
    

###############################################################################
#Creating datasets and update google sheets
###############################################################################

#creating base table
applications_data = base_table_creator()

#creating weekly data
byWeek_stats = applications_data.groupby(WEEKSTART_COL, as_index = False).apply(weekly_evaluator)
byWeek_stats[WEEKSTART_COL] = byWeek_stats[WEEKSTART_COL].astype(str)
byWeek_stats = byWeek_stats.fillna("")
byWeek_stats.replace(0, "", inplace=True)

#creating data for tables
tables_data = pd.DataFrame.from_dict(values_for_cells(applications_data))

#creating baseline data
baseline_data = create_baseline_data(applications_data)

#updating google sheets
sheets_updater(workbook_key = sheet_key, google_key_file = google_key_file, 
               byweek_dataset = byWeek_stats, tables_dataset = tables_data, baselines_dataset = baseline_data)






