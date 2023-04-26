import http.client
import json
import matplotlib.pyplot as plt
import datetime
import numpy as np
import numpy.linalg as alg
import pandas as pd



# connection to the API
conn = http.client.HTTPSConnection("yh-finance.p.rapidapi.com")

headers = {
    'X-RapidAPI-Key': "******************************************", # insert your Rapidapi Key instead of stars ***
    'X-RapidAPI-Host': "yh-finance.p.rapidapi.com"
}

# create a dictionary with the SNP500's companies
payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = payload[0]
second_table = payload[1]

df = first_table
symbols = df['Symbol'].values.tolist()
security= df['Security'].values.tolist()

dict_stock=dict(zip(symbols,security))



# 1st question and suggestion for help
print('\n')
print("Welcome to the Asset Management Assistant!\n")
need_help=input('Type "help" or "?" if you do not know how to use the Asset Management Assistant\n\n*** Clique Enter to continue ***\n: ').upper()

if need_help=="HELP" or need_help=="?":
    print("Asset Management Assistant is a program which helps to calculate the weighting of each asset. This program use the maximistion of sharp ratio.\n")
    print("You can chose how many stock you want to include in your portfolio and which one.\n")
    print("*** Disclaimer: Can only chose stocks inside the S&P500 ***\n")
    print("You can use this dictionary to know which company's choose")
    print(dict_stock)
    need_help = input('*** Enter to continue ***\n: ')

# error handling
number_of_stock = input("How many stock do you want to include in your Portfolio? Write a number between 2 and 10\n: ")
while type(number_of_stock)!=int or number_of_stock<2 or number_of_stock>10:
    number_of_stock = input("Please write a valide number between 2 and 10\n: ")
    try:
        number_of_stock=int(number_of_stock)
    except:
        pass

number_of_stock=int(number_of_stock)


# adding all the stocks in a list
print("The stock should be traded in the S&P500\n")


stock_list=[]

while len(stock_list) < number_of_stock:
    stock = input(f"Write your stock number {len(stock_list)+1}: ").upper()
    if stock not in dict_stock:
        print(f"{stock} is not a valid stock symbol. Please try again.") #error handling (avoid inclusion of firms other than the ones in SNP500)
    elif stock in stock_list:
        print(f"{stock} has already been chosen, please chose another stock.") #error handling (to avoid same stock 2 times in PF)
    else:
        stock_list.append(stock)

# create a list with the name of the chosen companies name
stock_name_list=[]
for i in stock_list:
    stock_name_list.append(dict_stock[i.upper()])
print('Your portfolio is composed with the stock of: ',stock_name_list)

stock_list.append('SPY') # add the snp500 to the list of stock (because we will create a matrice with all the stocks' price data


matrice_data =[]

for firm in stock_list:
    conn.request("GET", f"/stock/v2/get-chart?interval=1d&symbol={firm}&range=1y", headers=headers) # get info about each stock
    res = conn.getresponse()
    data = res.read()
    json_data = json.loads(data.decode("utf-8"))
    chart_data = json_data["chart"]["result"][0]
    adjclose_data = chart_data["indicators"]["adjclose"][0]["adjclose"] # select
    matrice_data.append(adjclose_data)


#matrice_data is a list of list, md is a matrixe so it's easier for calculation
md=np.array(matrice_data)
#create matrixe of daily return
daily_return = (md[:, 1:] - md[:, :-1]) / md[:, :-1]

#create a separate matrice to put the benchmark inside (SNP500)
snp500=daily_return[-1]
daily_return=np.delete(daily_return,(-1),axis=0)

# get the data for the 10 year treasury (it is our risk free rate)
conn.request("GET", "/market/get-charts?symbol=%5ETNX&interval=1d&range=1y&region=US", headers=headers)

res = conn.getresponse()
data = res.read()
json_data = json.loads(data.decode('utf-8'))
chart_data = json_data['chart']['result'][0]
tnx_data = chart_data['indicators']['quote'][0]['close'] #create a list of data

# delete the non-float data
while None in tnx_data:
    tnx_data.remove(None)

# delete the 1st day (or 1st 2 day) of data because we want to have the same number of data for tnx_data and daily_return
while len(tnx_data)-len(daily_return[0])>0:
    tnx_data.pop(0)

# build a matrixe of risk free rate
rf=np.array(tnx_data)
rf=(rf/100)/len(rf) # I did that to get the percentage of rf rate daily

# add the rf rate to the big matrice with all the stock
daily_return=np.append(daily_return,[rf],axis=0)

# variance of stocks + rf
import statistics
variance_list=[]
for i in daily_return:
    variance_list.append(statistics.variance(i))

variance_matrixe=np.array(variance_list)

# cov_matrice
cov_matrixe=np.cov(daily_return,bias=bool)

# beta of data
Daily_return2 = np.append(daily_return,[snp500],axis= 0)
Cov_matrixe2= np.cov(Daily_return2,bias=bool)
variance_market=Cov_matrixe2[-1][-1]

cov_line_matrixe=Cov_matrixe2[-1][:-1]

beta_matrixe=cov_line_matrixe/variance_market

# market risk premium
market_risk_premium_matrixe= snp500 - daily_return[-1]
if np.mean(market_risk_premium_matrixe)>5 and np.mean(market_risk_premium_matrixe)<6:
    market_risk_premium=np.mean(market_risk_premium_matrixe)
else:
    market_risk_premium=0.055/len(snp500) #because the 10year average MRP is 5.5 so if we are in an unusual market trend, we can still have some decent MRP number


# average risk free rate
average_rf=np.mean(daily_return[-1])

# expected return according to CAPM for each stock (and rf rate)
expected_return=np.array(average_rf+market_risk_premium*beta_matrixe)

#####################################################################################################################
# PF optimisation part
from scipy.optimize import minimize

# objective function to maximize (Sharpe ratio)
def objective(weights, rf, ret, cov):
    weights = np.resize(weights, (len(daily_return)))
    port_return = np.sum(weights * ret)
    port_var = np.dot(weights.T, np.dot(cov, weights))
    sharpe_ratio = (port_return - rf) / np.sqrt(port_var)
    return -sharpe_ratio # minimize the negative sharpe ratio = maximize the sharpe ratio

# define the constraint function
def constraint(weights):
    return np.sum(weights) - 1

# bounds for the weights
bounds = [(0, 1)] * len(stock_list)

# define the initial guess for the weights
x0 = np.ones(len(stock_list)) / len(stock_list)

# define the constraint object
cons = {'type': 'eq', 'fun': constraint}

# run the optimization
result = minimize(objective, x0, args=(average_rf, expected_return, cov_matrixe), method='SLSQP', bounds=bounds, constraints=cons)

# print the optimized weights
stock_name_list.append('Risk free asset')

for i in range(len(stock_name_list)):
    print(stock_name_list[i],':  ',round(result.x[i]*100,2),'%')

# print the expected return of PF

expected_return_pf=0
for i in range(len(stock_name_list)):
    expected_return_pf=expected_return_pf+(result.x[i]*expected_return[i])

expected_return_pf=expected_return_pf*len(tnx_data)

print("Expected return of PF: ",round(expected_return_pf * 100,2),'%')
