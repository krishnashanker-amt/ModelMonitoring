
# Library Imports

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

#modules to write to google sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials

#global variables - these store column names that will be used in functions below
SCORE_COL = 'score_5'
YPRED_COL = 'prediction'
YTRUE_COL = 'suspected_fraud'
TIME_COL = 'loan_processing_start_time'
WEEKSTART_COL = 'entered_lp_week'
#TIME_COL = 'created_at'
AMOUNT_COL = 'loan_amount'
THRESHOLD = 0.05
MODEL_START_DATE = '2018-09-15'

#Google Sheet names
BASELINE_WORKSHEET = 'Baselines Data'
WEEKLY_WORKSHEET = 'Raw Data'
TABLES_WORKSHEET = 'Tables Data'



#SQL Query to pull base table data
base_table_query = """
SELECT
  l.id as loan_id
, l.created_at
, date_trunc('week', l.created_at) as entered_lp_week
, l.status
, case when l.status in ('current','late','paid_off','charged_off') then 1 else 0 end as issued
--, case when c.high_confidence_fraud_indicator=true or cfl.id is not null then 1 else 0 end as high_confidence_fraud_indicator
, case when cfr.customer_id is not null then 1 else 0 end as suspected_fraud 
--, cfrt.name as fraud_reason
, cast(fd.score_5_old as float8) as score_5_old
, cast(fd.score_5_new as float8) as score_5_new
, coalesce(cast(fd.score_5_old as float8), cast(fd.score_5_new as float8)) as score_5
, l.state
, l.payment_method
, (l.amount_cents/100) as loan_amount
, ca.product_type
FROM customer_applications ca
LEFT JOIN loans l on l.customer_application_id = ca.id
JOIN customers c
  ON c.id = l.customer_id
  
  -- getting dependent variable
  
LEFT JOIN (
select customer_id 
from customer_fraud_reasons cfr 
group by 1
) cfr on c.id = cfr.customer_id
  
 -- LEFT JOIN avant.avant_basic.customer_fraud_reason_types cfrt on cfr.customer_fraud_reason_type_id = cfrt.id
  
  -- getting fraud scores
LEFT JOIN (
  SELECT
    l.id as loan_id
  , fd.model_scores -> 'fraud/en-US/4.1.0' ->> 'score' as score_4
  , fd.model_scores -> 'fraud/en-US/5.0.0' ->> 'score' as score_5_old
  , fd.model_scores -> 'fraud/en-US/5.0.0/avant' ->> 'score' as score_5_new
  , fd.id as fraud_decision_id
  , row_number() over (partition by l.id order by fd.created_at desc) as row_num
  FROM loans l
  JOIN fraud_decisions fd
    ON fd.customer_application_id = l.customer_application_id
    AND fd.created_at AT TIME ZONE 'America/Chicago' >= l.created_at
WHERE l.created_at > date '{START_DATE}'
) fd 
  ON fd.loan_id = l.id 
  AND fd.row_num=1
  -- getting fraud indicator
LEFT JOIN confirmed_fraud_logs cfl 
  ON cfl.customer_id = c.id
  
    -- filtering for valid loans to evaluate performance on
  -- JOIN avant.dw.loan_performance_by_installment lp 
  -- ON lp.loan_id = l.id 
  -- AND lp.installment_number = 1
  -- AND lp.installment_date <= date_add('day', -64, current_timestamp)
  
WHERE l.created_at > date '{START_DATE}'
""".format(START_DATE = MODEL_START_DATE)


def base_table_creator(connection_name, query = base_table_query):
    connector = trellis.connect(connection_name)
    df_raw = pd.read_sql(query, connector)
    df = df_raw[df_raw[SCORE_COL].notnull()]
    df[YPRED_COL] = np.where(df[SCORE_COL] > THRESHOLD, 1, 0)
    return df
    

#Get monitoring metrics for each week
def weekly_evaluator(dframe, ytrue = YTRUE_COL, ypred = YPRED_COL, scores = SCORE_COL):
    
    #calculating multiple metrics
    precision = precision_score(y_true = dframe[ytrue], y_pred = dframe[ypred], pos_label = 1, zero_division = 0)
    recall = recall_score(y_true = dframe[ytrue], y_pred = dframe[ypred], pos_label = 1, zero_division = 0)
    f1score = f1_score(y_true = dframe[ytrue], y_pred = dframe[ypred], pos_label = 1)
    auc_pr = average_precision_score(y_true = dframe[ytrue], y_score = dframe[scores], pos_label=1)
    fraud_rate = dframe[ytrue].sum()/len(dframe.index)
    avg_score = dframe[scores].sum()/len(dframe.index)
    try:
        auc_roc = roc_auc_score(y_true = dframe[ytrue], y_score = dframe[scores])
    except ValueError:
        auc_roc = ""

    
    
    return pd.Series({'precision': precision, 'recall': recall, 'f1score': f1score, 'auc_pr':auc_pr, 'auc_roc':auc_roc,
                     'fraud_rate': fraud_rate, 'avg_score': avg_score})


#function to create metric values for tables in Google Sheets
def values_for_cells(dframe, ytrue = YTRUE_COL, ypred = YPRED_COL, scores = SCORE_COL, timecol = TIME_COL, amount = AMOUNT_COL):
   
    #Setting up variables with different date values
    model_start_date = min(dframe[TIME_COL]).date()
    today_date = date.today().strftime("%Y-%m-%d")
    prev30_date = (date.today() - timedelta(days = 30)).strftime("%Y-%m-%d")
    prev60_date = (date.today() - timedelta(days = 60)).strftime("%Y-%m-%d")
    
    modeltrain_date_start = model_start_date.strftime("%Y-%m-%d")
    modeltrain_date_end = (model_start_date + timedelta(days = 30)).strftime("%Y-%m-%d")
    
    #creating different datasets for the different time periods
    
    #dataset 1 - 30 days after model was trained

    
    data_first30 = dframe.query('{0} > @modeltrain_date_start & {0} < @modeltrain_date_end'.format(TIME_COL))
    data_last30 = dframe.query('{0} > @prev30_date & {0} < @today_date'.format(TIME_COL))
    data_prev30 = dframe.query('{0} > @prev60_date & {0} < @prev30_date'.format(TIME_COL))   

    #PRECISION
    precision_current = precision_score(y_true = data_last30[ytrue], y_pred = data_last30[ypred], pos_label = 1)
    precision_initial = precision_score(y_true = data_first30[ytrue], y_pred = data_first30[ypred], pos_label = 1)
    precision_prev30 = precision_score(y_true = data_prev30[ytrue], y_pred = data_prev30[ypred], pos_label = 1)

    #recall values
    recall_current = recall_score(y_true = data_last30[ytrue], y_pred = data_last30[ypred], pos_label = 1)
    recall_initial = recall_score(y_true = data_first30[ytrue], y_pred = data_first30[ypred], pos_label = 1)
    recall_prev30 = recall_score(y_true = data_prev30[ytrue], y_pred = data_prev30[ypred], pos_label = 1) 

    #F1 score
    f1_current = f1_score(y_true = data_last30[ytrue], y_pred = data_last30[ypred], pos_label = 1)
    f1_initial = f1_score(y_true = data_first30[ytrue], y_pred = data_first30[ypred], pos_label = 1)
    f1_prev30 = f1_score(y_true = data_prev30[ytrue], y_pred = data_prev30[ypred], pos_label = 1) 

    #auc pr
    aucpr_current = average_precision_score(y_true = data_last30[ytrue], y_score = data_last30[scores], pos_label = 1)
    aucpr_initial = average_precision_score(y_true = data_first30[ytrue], y_score = data_first30[scores], pos_label = 1)
    aucpr_prev30 = average_precision_score(y_true = data_prev30[ytrue], y_score = data_prev30[scores], pos_label = 1) 

    #auc roc
    aucroc_current = roc_auc_score(y_true = data_last30[ytrue], y_score = data_last30[scores])
    aucroc_initial = roc_auc_score(y_true = data_first30[ytrue], y_score = data_first30[scores])
    aucroc_prev30 = roc_auc_score(y_true = data_prev30[ytrue], y_score = data_prev30[scores]) 

    #TODO - Confirm fraud rate definition
    #fraud rate
    fraudrate_current = data_last30[ytrue].sum()/len(data_last30.index)
    fraudrate_initial = data_first30[ytrue].sum()/len(data_first30.index)
    fraudrate_prev30 = data_prev30[ytrue].sum()/len(data_prev30.index)
    
    #avg score
    avgscore_current = data_last30[scores].sum()/len(data_last30.index)
    avgscore_initial = data_first30[scores].sum()/len(data_first30.index)
    avgscore_prev30 = data_prev30[scores].sum()/len(data_prev30.index)


    #TODO - Confirm fraud missed definition
    #fraud rate with dollar values
    fraudrate_dollar_current = (data_last30[amount]*data_last30[YTRUE_COL]).sum()/data_last30[amount].sum()
    fraudrate_dollar_initial = (data_first30[amount]*data_first30[YTRUE_COL]).sum()/data_first30[amount].sum()
    fraudrate_dollar_prev30 = (data_prev30[amount]*data_prev30[YTRUE_COL]).sum()/data_prev30[amount].sum()

    #$ value of fraud missed
    fraudmissed_dollar_current = data_last30[scores].sum()/len(data_last30.index)
    fraudmissed_dollar_initial = data_first30[scores].sum()/len(data_first30.index)
    fraudmissed_dollar_prev30 = data_prev30[scores].sum()/len(data_prev30.index)

    output = {"metric": ['precision', 'recall','f1score', 'auc_pr', 'auc_roc', 'fraudrate', 'avg_score', 'fraudrate_dollar', 'fraudmissed_dollar'],
             "current_values":[precision_current, recall_current, f1_current, aucpr_current, aucroc_current, fraudrate_current, avgscore_current, fraudrate_dollar_current, fraudmissed_dollar_current],
             "initial_values":[precision_initial, recall_initial, f1_initial, aucpr_initial, aucroc_initial, fraudrate_initial, avgscore_initial, fraudrate_dollar_initial, fraudmissed_dollar_initial],
             "prev30_values":[precision_prev30, recall_prev30, f1_prev30, aucpr_prev30, aucroc_prev30, fraudrate_prev30, avgscore_prev30, fraudrate_dollar_prev30, fraudmissed_dollar_prev30]}    
        
    return output


#Function to create baseline data that will be used in charts
def create_baseline_data(dframe, ytrue = YTRUE_COL, ypred = YPRED_COL, scores = SCORE_COL, timecol = TIME_COL, amount = AMOUNT_COL):
    #Setting up variables with different date values
    modeltrain_date_start = datetime.strptime(MODEL_START_DATE, "%Y-%m-%d")
    modeltrain_date_end = (modeltrain_date_start + timedelta(days = 30)).strftime("%Y-%m-%d")
    
    #creating different datasets for the different time periods
    
    #dataset 1 - 30 days after model was trained
    data_first30 = dframe.query('{0} > @MODEL_START_DATE & {0} < @modeltrain_date_end'.format(TIME_COL))
    
    #PRECISION
    precision_initial = precision_score(y_true = data_first30[ytrue], y_pred = data_first30[ypred], pos_label = 1)
    
    #recall values
    recall_initial = recall_score(y_true = data_first30[ytrue], y_pred = data_first30[ypred], pos_label = 1)
    
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
    fraudrate_dollar_initial = (data_first30[amount]*data_first30[YTRUE_COL]).sum()/data_first30[amount].sum()

    #$ value of fraud missed
    fraudmissed_dollar_initial = data_first30[scores].sum()/len(data_first30.index)
    
    #creating grouped by data frame with needed weeks
    baseline_dataframe = pd.DataFrame(dframe[WEEKSTART_COL].unique()).rename(columns={0: WEEKSTART_COL}).sort_values(by = WEEKSTART_COL)
    baseline_dataframe = baseline_dataframe.assign(precision_baseline = precision_initial,
                              recall_baseline = recall_initial, 
                              f1_baseline = f1_initial, 
                              aucpr_baseline = aucpr_initial,
                              aucroc_baseline = aucroc_initial,
                              fraudrate_baseline = fraudrate_initial,
                              avgscore_baseline = avgscore_initial,
                              fraudrate_dollar_baseline = fraudrate_dollar_initial,
                              fraudmissed_dollar_baseline = fraudmissed_dollar_initial
                              )
    
    return baseline_dataframe


def sheets_updater(workbook_key, google_key_file, byweek_dataset, tables_dataset, baselines_dataset):
    
    #authorization
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(google_key_file, scope)
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
    
    
    
    
    
    
    
    

    
    
    
    
        
        
        