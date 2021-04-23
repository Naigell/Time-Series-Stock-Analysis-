#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#Suppressing `logger.error('Importing plotly failed. Interactive plots will not work.')'
class suppress_stdout_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

with suppress_stdout_stderr(): 
    from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')    

#read the csv file
df= pd.read_csv("all_stocks_5yr.csv")
df = df.rename(columns={'Name': 'Corp'})

print(df.head())

#change date column from object to datetime data type
df['date']= pd.to_datetime(df.date)

print(df.info())

#set date column as index 
df= df.set_index("date")

#create dataframe for apple stocks
aapl = df.loc[df['Corp'] == 'AAPL']
aapl_df= aapl.copy()
print(aapl_df.head())

#select date and close columns only for the dataframe
aapl_df = aapl_df.filter(['date','close'], axis=1)

#plot closing price history
plt.figure(figsize=(10,8))
plt.plot(aapl_df, label='close price history')
plt.grid()
plt.show()

#test for stationarity
def check_stationarity(df):
    # rolling statistics
    rolling_mean = df.rolling(window=12).mean()
    rolling_std = df.rolling(window=12).std()
        
    # rolling statistics plot
    original = plt.plot(df, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
        
    #Augmented Dickey-Fuller test
    from statsmodels.tsa.stattools import adfuller
    series= df.values
    check_stationarity.result = adfuller(series)
    print('adfstatistic : %f' %check_stationarity.result[0])
    print('P-value : %f' %check_stationarity.result[1])
    for key, value in check_stationarity.result[4].items():
        print('\t%s: %.3f' % (key, value)) 
        check_stationarity.critical = check_stationarity.result[4]    

check_stationarity(aapl_df)

not_stationary ='''from the ADF test it is observed that the test statistic is greater than the critical 
values and the P-value is much greater than the 0.05 threshold, hence, we fail to 
reject the null hypothesis (non-stationarity). The data is not stationary. '''
   
stationary = '''from the ADF test it is observed that the adf statistic is smaller than the critical 
values and the P-value is less than 0.05 which allows for a rejection of null hypothesis 
(non-stationarity) and acceptance of tke alternative hypothesis (stationarity)'''

if check_stationarity.result[1] > 0.05 and check_stationarity.result[0] > check_stationarity.critical['1%']:
    print(not_stationary)
else:
    print(stationary)

''' Fbprophet has the capability of handling this type of data so 
we need not worry about making the data stationary'''

#rename the columns as desired by prophet
aapl_df.reset_index(inplace=True)    
aapl_df.rename(columns= {'close': 'y', 'date':'ds'}, inplace = True)
print(aapl_df.tail())

#split the data 90:10
aapl_df_train = aapl_df[:1123]
aapl_df_test = aapl_df[1123:]

# Define a function to implement the fbprophet model
def proph_it(train, test, whole, forecast_periods1, 
             forecast_periods2, interval=0.95):
    '''Uses Facebook Prophet to fit model to train set, evaluate  
    performance with test set, and forecast with whole dataset. The 
    model has a 95% confidence interval by default.
       
       Remember: datasets need to have two columns, `ds` and `y`.
       Dependencies: fbprophet, matplotlib.pyplot as plt
       Parameters:
          train: training data
          test: testing/validation data
          whole: all available data for forecasting
          interval: confidence interval (percent)
          forecast_periods1: number of months for forecast on 
              training data
          forecast_periods2: number of months for forecast on whole 
              dataset'''
    
    # Fit model to training data and forecast
    model = Prophet(interval_width=interval)
    model.fit(train)
    future = model.make_future_dataframe(periods=forecast_periods1)
    forecast = model.predict(future)
    
    # Plot the model and forecast
    model.plot(forecast, uncertainty=True)
    plt.plot(test['ds'], test['y'], label='y_hat')
    plt.title('Training data with forecast')
    plt.legend()
    
    # Make predictions and compare to test data
    y_pred = model.predict(test)
    
    # Plot the model, forecast, and actual (test) data
    model.plot(y_pred, uncertainty=True)
    plt.plot(test['ds'], test['y'], color='r', label='actual')
    plt.title('Validation data v. forecast')
    plt.legend()
    
    # Fit a new model to the whole dataset and forecast
    model2 = Prophet(interval_width=interval)
    model2.fit(whole)
    future2 = model2.make_future_dataframe(periods=forecast_periods2)
    forecast2 = model2.predict(future2)
    
    # Plot the model and forecast
    model2.plot(forecast2, uncertainty=True)
    plt.plot(test['ds'], test['y'], label='y_hat')
    plt.title('{}-day forecast'.format(forecast_periods2))
    plt.legend()
    
    # Plot the model components
    model2.plot_components(forecast)
    
    return y_pred, forecast2

# Run the wrapper function, supplying numbers of months for forecasts
short_term_pred, long_term_pred= proph_it(aapl_df_train, aapl_df_test, aapl_df, 365, 3650)

conclusion = '''From these plots it is observed that Apple stock prices are expected to 
have an upward trend for the forseeable future. It is also observed that yearly apple 
stock prices are usually at their lowest in February and at their highest in May/June'''

print(conclusion)







