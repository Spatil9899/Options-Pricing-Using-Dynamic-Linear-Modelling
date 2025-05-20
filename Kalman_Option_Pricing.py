import pandas as pd
import pickle as pk
import numpy as np
import OLSprograms as op
import matplotlib.pyplot as plt
from numpy import matrix as mat
from scipy.stats import norm
from scipy.stats import t
# import yfinance as yf
import Garch as ga
import BlackScholes as bs
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import glob
import os
from datetime import datetime
# C:\A_BL\Teaching\AAA_StandardCourses\MSF_567_Bayesian_Econometrics\BayesianEconometrics\Class_6_Python
# cwd = os.getcwd()
# os.listdir(cwd + '\\' + 'STOCKS')
# os.listdir(cwd + '\\' + 'STOCKS' + '\\' +'NEE(NextEra Energy)')
# glob.glob('*.rar')
# pip3 install yfinance
# C:\A_BL\Teaching\AAA_StandardCourses\MSF_567_Bayesian_Econometrics\AAA_Fall2015\Week_11\NonlinearOptionsModel\ClassIMixture\Experimental
# C:\A_BL\Teaching\AAA_StandardCourses\MSF_567_Bayesian_Econometrics\AAA_Fall2015\Week_11\NonlinearOptionsModel
# See Option_Pricing_using_a_Nonlinear_Dynamic_Model.pdf
# import fnmatch
# l = ['RT07010534.txt', 'RT07010533.txt', 'RT02010534.txt']
# pattern = 'RT0701*.txt'
# matching = fnmatch.filter(l, pattern)
# print(matching)


def PlotsDLM(AssetDLM):
    Asset  = SelectAsset(AssetDLM)
    Continue = True
    while Continue:
        Strike = SelectStrike(AssetDLM,Asset)
        Plot   = SelectPlot()
        Plot   = int(Plot)
        Continue = False
        Df = AssetDLM[Asset]['OptionChain']['call'][Strike]
        if 'AssetType' in AssetDLM[Asset]['OptionChain']:
            AssetType = AssetDLM[Asset]['OptionChain']['AssetType']
        if not 'AssetType' in AssetDLM[Asset]['OptionChain']:
            AssetType = 'Simulated'  
        N       = len(Df)-1
        ImpVol  = Df['ImpVol'][0:N].values 
        OptionPrice  = Df['OptionPrice'][0:N].values 
        a       = Df['a'][0:N].values
        a_sqroot = np.sqrt(a)
        m       = Df['m'][0:N].values
        m_sqroot = np.sqrt(m)
        f       = Df['f'][0:N].values
        e       = Df['e'][0:N].values
        
        Obs     = np.vstack((ImpVol,a)).T
        GoodObs = np.isnan(np.sum(Obs,1)) == False
        ImpVol  = ImpVol[GoodObs]
        OptionPrice = OptionPrice[GoodObs]
        a           = a[GoodObs]
        a_sqroot    = a_sqroot[GoodObs]
        m           = m[GoodObs]
        m_sqroot    = m_sqroot[GoodObs]
        f           = f[GoodObs]
        e           = e[GoodObs]
        N           = len(ImpVol)
        DD          = 20
        if Plot == 1:
            days = list(range(1, N+1))
            plt.plot(days,ImpVol, '-b', days,
                     a_sqroot, '-g',  days,m_sqroot, '-r',linewidth=3)
            plt.title(AssetType + ' ' +Asset + ': Volatility by Day')
            plt.xlabel('Day')
            plt.ylabel('Volatility')
            plt.legend(['Implied', 'Prior Garch(1,1)','Posterior Garch(1,1)'])
        if Plot == 2:
            m1, b = np.polyfit(a_sqroot[DD:],ImpVol[DD:], 1)
            days = list(range(1, N+1))
            plt.plot(a_sqroot[DD:],ImpVol[DD:], '.g')
            plt.plot(a_sqroot[DD:], m1*a_sqroot[DD:]+b, color='red')
            plt.suptitle(AssetType + ' ' +Asset +  ': Implied vs Prior Garch(1,1) Volatility')
            plt.title('Intercept = ' + str(round(b,4)) + ' slope = ' + str(round(m1,4)))
            plt.xlabel('Prior Garch(1,1) Volatility')
            plt.ylabel('Implied Volatility')
        if Plot == 3:
            days = list(range(1, N+1))
            plt.plot(days,OptionPrice,'-g',days,f,'-b',linewidth = 3)
            plt.suptitle(AssetType + ' ' +Asset + ': Observed Option and Forecast Price by Day')
            # plt.title('Intercept = ' + str(round(b,4)) + ' slope = ' + str(round(m,4)))
            plt.xlabel('Day')
            plt.ylabel('Price')  
            plt.legend(['Observed Price', 'Forecasted Price'])
        if Plot == 4:
            days = list(range(1, N+1))
            plt.plot(OptionPrice[DD:],f[DD:],'.b')
            plt.suptitle(AssetType + ' ' +Asset + ': Observed Option and Forecast Option Price')
            # plt.title('Intercept = ' + str(round(b,4)) + ' slope = ' + str(round(m,4)))
            plt.xlabel('Option Price')
            plt.ylabel('Forecast Option Price')  
    return Df

def SelectAsset(AssetDLM):
    ListAll  = list(AssetDLM.keys())
    remains  = ListAll.copy()
    N       = 10
    if len(ListAll) > 10:
        s       = 0
        e       = N
    if len(ListAll) < 10:
        s       = 0
        e       = len(ListAll) 
    if len(ListAll) < 10:
        L3 = [str(ListAll.index(L)) + ' - ' + L for L in ListAll[s:e]]
        print(L3)

    if len(ListAll) > 10:
        while len(remains) > 0:
            L3 = [str(ListAll.index(L)) + ' - ' + L for L in ListAll[s:e]]
            print(L3)
            for i in list(range(s,e)):
                # print('removing ' + ListAll[i])
                remains.remove(ListAll[i])
            if len(remains) > N:
                s  = s + 10 
                e  = e + 10
            if len(remains) < N: 
                s  = len(ListAll)-len(remains)
                e  = len(ListAll)
            
    Pos = input('Enter asset number: ')
    Asset = ListAll[int(Pos)]
    return Asset
def SelectStrike(AssetDLM,Asset):
    ListAll  = list(AssetDLM[Asset]['OptionChain']['call'])
    N  = len(ListAll)
    L3 = [str(ListAll.index(L)) + ' for strike ' + L for L in list(ListAll)[0:N]]
    print(L3)
    Pos = input('Enter strike number: ')
    Strike = ListAll[int(Pos)]
    return Strike

def SelectPlot():
    ListAll  = ['1','2','3','4']
    N       = len(ListAll)
    L3      = [str(ListAll.index(L)) + ' for plot  ' + L for L in list(ListAll)[0:N]]
    print(L3)
    Pos = input('Enter plot number: ')
    Plot = ListAll[int(Pos)]
    return Plot
def ProcessAllAssets():
    global DaysPerYear
    DaysPerYear = 365
    AssetDLM = {}
    AssetList = GetAssetList()
    # AssetList = list(['jpm'])
    PriceDf, ReturnsDf = LoadStockPrice()
    GarchDf = EstimateGarch11(ReturnsDf)  # Estimate parameters of Garch(1,1)
    RFDf = LoadRiskFreeRate()
    for Asset in AssetList:
        AssetUpper = Asset.upper()
        OptionChain = LoadOptionChain(AssetUpper, GarchDf)
        if 'Asset' in OptionChain:
            OptionChain = PreProcessOptionChain(
                OptionChain, PriceDf, ReturnsDf, RFDf)
            AssetDLM[AssetUpper] = {}
            AssetDLM[AssetUpper]['OptionChain'] = OptionChain
            Garch11Param = AssetDLM[AssetUpper]['OptionChain']['Garch11Param']
            CallStrikeList = list(
                AssetDLM[AssetUpper]['OptionChain']['call'].keys())
            for Strike in CallStrikeList:
                Df = AssetDLM[AssetUpper]['OptionChain']['call'][Strike]
                D = SetupDLM362(Df, Garch11Param)
                D = RunDLM362(D, Optimize=0)
                AssetDLM[AssetUpper]['OptionChain']['callKalman'][Strike] = D
    AssetDLM = Add_ImpVol_DLMresults(AssetDLM)
 
    save(AssetDLM, 'AssetDLM.pkl')
    return AssetDLM


def Add_ImpVol_DLMresults(AssetDLM):
    AssetList = list(AssetDLM.keys())
    # AssetList = [AssetList[1]]
    for asset in AssetList:
        callstrikelist = AssetDLM[asset]['OptionChain']['call']
        # putstrikelist   = AssetDLM[asset]['OptionChain']['put']
        for strike in callstrikelist:
            print('Asset = ' + asset + ' call strike = ' + str(strike))
            Df = AssetDLM[asset]['OptionChain']['call'][strike].copy()
            # print(Df1)
            D = AssetDLM[asset]['OptionChain']['callKalman'][strike]
            KalmanResults = np.hstack(
                (D['a'], D['R'], D['m'], D['C'], D['f'], D['Q'], D['e']))
            implied_vol = np.zeros((len(Df), 1))
            if 'vwap' in Df:
                usevwap = 1
                # print('1.1 usevwap = ' + str(usevwap))
            if not 'vwap' in Df:
                usevwap = 0
                # print('1.2 usevwap = ' + str(usevwap))
            for i in range(0, len(Df)):
                if usevwap == 1:
                    # print('Asset = ' + asset + ' call strike = ' + str(strike))
                    # print('2.1 usevwap = ' + str(usevwap))
                    V_market = Df['vwap'].values[i]
                if usevwap == 0:
                    V_market = Df['OptionPrice'].values[i]
                    # print('2.2 usevwap = ' + str(usevwap))
                S = Df['StockPrice'].values[i]
                K = Df['Strike'].values[i]
                T = Df['Maturity'].values[i]
                r = Df['RiskFree'].values[i]
                iv = bs.find_vol(V_market, S, K, T, r)
                # if np.isnan(iv):
                #     print(i)
                #     print((V_market, S, K, T, r))
                implied_vol[i] = iv
            Df['ImpVol'] = implied_vol
            Df['a'] = KalmanResults[:, 0]
            Df['R'] = KalmanResults[:, 1]
            Df['m'] = KalmanResults[:, 2]
            Df['C'] = KalmanResults[:, 3]
            Df['f'] = KalmanResults[:, 4]
            Df['Q'] = KalmanResults[:, 5]
            Df['e'] = KalmanResults[:, 6]
            AssetDLM[asset]['OptionChain']['call'][strike] = Df
    return AssetDLM


def LoadStockPrice():
    PriceDf = pd.read_excel('Daily_Closing_price.xlsx')
    Columns = PriceDf.columns
    NewCols = []
    NewIndex = ['Last Price', 'Dates']
    ctr = -1
    for s in Columns:
        ctr = ctr + 1
        # keep stock symbol
        NewCols.append(Columns[ctr][0:Columns[ctr].find(' ')])

    PriceDf.columns = NewCols
    PriceDf.index = PriceDf['Unnamed:']
    PriceDf.index.name = 'Date'
    PriceDf = PriceDf.drop('Unnamed:', axis=1)
    OldIndex = PriceDf.index
    for d in range(2, len(OldIndex)):
        NewIndex.append(OldIndex[d].strftime("%Y-%m-%d"))
    PriceDf.index = NewIndex
    # Compute returns
    NumAssets = len(PriceDf.columns)
    Price = PriceDf['SPX'].values[2:]
    Returns = np.zeros((len(Price), NumAssets))*np.nan
    ctr = -1
    for asset in PriceDf.columns:
        ctr = ctr + 1
        Price = PriceDf[asset].values[2:]
        for t1 in range(1, len(Price)):
            Returns[t1, ctr] = np.log(Price[t1]/Price[t1-1])
    ReturnsDf = pd.DataFrame(Returns)
    ReturnsDf.index = NewIndex[2:]
    # ReturnsDf.index.name = 'Date'
    ReturnsDf.columns = PriceDf.columns
    ReturnsDf.drop(index=ReturnsDf.index[0], axis=0, inplace=True)
    return PriceDf, ReturnsDf


def EstimateGarch11(ReturnsDf):
    AssetList = ReturnsDf.columns
    Results = np.zeros((len(AssetList), 3))
    ctr = -1
    for Asset in AssetList:
        ctr = ctr + 1
        Df = ReturnsDf[Asset]
        Df = pd.DataFrame(Df)
        Df.columns = ['log_returns']
        Returns = Df['log_returns'].values*100
        Res = ga.Garch11(Returns)
        Results[ctr, :] = Res.params
    GarchDf = pd.DataFrame(Results, columns=['omega', 'alpha', 'beta'])
    GarchDf.index = AssetList
    GarchDf.index.name = 'Asset'
    return GarchDf


def LoadRiskFreeRate():
    RFDf = pd.read_csv('DTB3.csv')
    RFDf.index = RFDf['observation_date']
    RFDf.index.name = 'Date'
    RFDf = RFDf.drop('observation_date', axis=1)
    RFDf['DTB3'] = RFDf['DTB3'].values / 100
    rates = RFDf.values
    for t1 in range(1, len(rates)):
        if np.isnan(rates[t1][0]):
            rates[t1] = rates[t1-1][0]
    RFDf['DTB3'] = rates
    return RFDf


def GetAssetList():
    cwd = os.getcwd()
    SubDir = os.listdir(cwd + '/' + 'STOCKS' + '/')
    AssetList = []
    for Dir in SubDir:
        posdot = Dir.find('.')
        if posdot == -1:
            Search = cwd + '/Stocks' + '/' + Dir + '/' + '*.csv'
            filelist = glob.glob(Search)
            for f in filelist:
                f1 = getfilename(f)
                try:
                    putpos = f1.index('put')
                    assetpos = f1.index('_')
                    AssetList.append(f1[0:assetpos])
                except:
                    putpos = -1
                try:
                    callpos = f1.index('call')
                    assetpos = f1.index('_')
                    AssetList.append(f1[0:assetpos])
                except:
                    callpos = -1
    AssetList = list(set(AssetList))
    # AssetList = ['ABNB','ADBE']
    return AssetList


def getfilename(f):
    Continue = True
    while Continue:
        try:
            pos = f.index('/')
            f = f[pos+1:]
        except:
            Continue = False
    f = f.lower()
    return f


def LoadOptionChain(Asset, GarchDf):
    AssetsWithReturns = list(GarchDf.T.columns)
    OptionChain = {}  # Option strikes dictionary
    if not (Asset in AssetsWithReturns):
        print(Asset + ' DOES NOT HAVE RETURNS, WILL NOT BE PROCESSED!')
    if Asset in AssetsWithReturns:
        print(Asset + ' has returns')
        cwd = os.getcwd()
        os.listdir(cwd + '/' + 'STOCKS' + '/' + Asset)
        Path = cwd + '/' + 'STOCKS' + '/' + Asset + '/'
        Garch11Param = GarchDf.T[Asset].values
        OptionChain['Asset'] = Asset  # Which Asset is being loaded
        OptionChain['Garch11Param'] = Garch11Param  # Garch(1,1) parameters
        OptionChain['call'] = {}  # Option strikes dictionary for calls
        OptionChain['callKalman'] = {}
        OptionChain['put'] = {}  # Option strikes dictionary for puts
        OptionChain['putKalman'] = {}
        filelist = glob.glob(Path + '*.csv')  # find all matching file
        for f in filelist:
            f1 = getfilename(f)
            Df = pd.read_csv(Path + f1)
            Df.index = Df['timestamp']
            Df.index.name = 'Date'
            Df = Df.drop('timestamp', axis=1)
            Df.rename(
                columns={'time_to_expiration_days': 'Maturity'}, inplace=True)
            # Df['Maturity'] = Df['Maturity'].values/256
            Df['Maturity'] = Df['Maturity'].values/365
            Asset, OptionType, Strike = Extract_Asset_OptionType_Strike(f1)
            # print((Asset,OptionType,Strike))
            if OptionType == 'call':
                OptionChain['call'][Strike] = Df
            if OptionType == 'put':
                OptionChain['put'][Strike] = Df

    return OptionChain


def Extract_Asset_OptionType_Strike(f):
    # print(f)
    Asset = f[0:f.index('_')]
    f2 = f.replace(Asset+'_', '')
    OptionType = f2[0:f2.index('_')].lower()
    f3 = f2.replace(OptionType+'_', '')
    Strike = f3[0:f3.index('_')]
    return Asset, OptionType, Strike


def PreProcessOptionChain(OptionChain, PriceDf, ReturnsDf, RFDf,AssetType='Actual'):
    global DaysPerYear
    Asset = OptionChain['Asset']
    OptionChain['AssetType'] = AssetType
    AssetPriceDf = PriceDf[Asset]
    strike_list_call = list(OptionChain['call'].keys())
    strike_list_put = list(OptionChain['put'].keys())
    for strike in strike_list_call:
        Df = OptionChain['call'][strike]
        AvgOpenClose = np.mean(Df[['open', 'close']].values, 1)
        Df['AvgOpenClose'] = AvgOpenClose  # Average of open and close
        Strike = np.ones((len(AvgOpenClose)))*float(strike)
        Df['Strike'] = Strike    # convert to number
        Df['DaysPerYear'] = DaysPerYear
        if 'vwap' in Df:
            # print('vwap in Df')
            Df['OptionPrice'] = Df['vwap']
        if not 'vwap' in Df:
            # print('vwap NOT in Df')
            Df['OptionPrice'] = Df['AvgOpenClose']
        Df = pd.merge(Df, RFDf, left_index=True, right_index=True)
        Df.rename(columns={'DTB3': 'RiskFree'}, inplace=True)
        Df = pd.merge(Df, AssetPriceDf, left_index=True, right_index=True)
        Df.rename(columns={Asset: 'StockPrice'}, inplace=True)
        RetDf = ReturnsDf[Asset]
        RetDf = pd.DataFrame(RetDf)
        Df = pd.merge(Df, RetDf, left_index=True, right_index=True)
        Df.rename(columns={Asset: 'StockReturn'}, inplace=True)
        Df.drop(['open', 'high', 'low', 'close',
                'volume'], axis=1, inplace=True)
        OptionChain['call'][strike] = Df
        # print(Df.T)
    for strike in strike_list_put:
        Df = OptionChain['put'][strike]
        AvgOpenClose = np.mean(Df[['open', 'close']].values, 1)
        Df['AvgOpenClose'] = AvgOpenClose  # Average of open and close
        Strike = np.ones((len(AvgOpenClose)))*float(strike)
        Df['Strike'] = Strike    # convert to number
        Df['DaysPerYear'] = DaysPerYear
        if 'vwap' in Df:
            Df['OptionPrice'] = Df['vwap']
        if not 'vwap' in Df:
            Df['OptionPrice'] = Df['AvgOpenClose']
        Df = pd.merge(Df, RFDf, left_index=True, right_index=True)
        Df.rename(columns={'DTB3': 'RiskFree'}, inplace=True)
        Df = pd.merge(Df, AssetPriceDf, left_index=True, right_index=True)
        Df.rename(columns={Asset: 'StockPrice'}, inplace=True)
        RetDf = ReturnsDf[Asset]
        RetDf = pd.DataFrame(RetDf)
        Df = pd.merge(Df, RetDf, left_index=True, right_index=True)
        Df.rename(columns={Asset: 'StockReturn'}, inplace=True)
        Df.drop(['open', 'high', 'low', 'close',
                'volume'], axis=1, inplace=True)
        OptionChain['put'][strike] = Df
    return OptionChain


# def Test(Optimize=0):
#     D = GetData('AAPL')
#     D = CreateOptionData(D)
#     D = RunDLM362(D, Optimize)
#     print('Correlation forecast to observed = ' + str(D['Corr_f_y']))
#     return D


# def GetData(ticker='AAPL', start_date='2024-04-30', end_date='2025-03-10'):
#     Data = yf.download(ticker, start=start_date, end=end_date)
#     Data['log_return'] = np.log(Data['Close'] / Data['Close'].shift(1))
#     D = {}
#     D['ticker'] = ticker
#     D['start_date'] = start_date
#     D['end_date'] = end_date
#     D['Data'] = Data
#     D['log_return'] = Data['log_return']
#     return D

def CreateOptionSimData():
    SimAssetDLM    = {}
    NumStrikes  = 5
    PriceDf, ReturnsDf = LoadStockPrice()
    RFDf        = LoadRiskFreeRate()
    AssetList   = ['AAPL','AMAT','GOOG']
    for Asset in AssetList:
        R,Df,D  = CreateOptionData(PriceDf,ReturnsDf,RFDf,NumStrikes,Asset)
        SimAssetDLM[Asset] = {}
        SimAssetDLM[Asset]['OptionChain'] = {}
        SimAssetDLM[Asset]['OptionChain']['Asset'] = Asset
        # SimAssetDLM[Asset]['OptionChain']['AssetType'] = 'Simulated'
        SimAssetDLM[Asset]['OptionChain']['GarchPaam'] = R['GarchParam']
        SimAssetDLM[Asset]['OptionChain']['call']  = {}
        SimAssetDLM[Asset]['OptionChain']['callKalman']  = {}
        StrikeList  = list(R['K'][1,:])
        for strike in StrikeList:
            SimAssetDLM[Asset]['OptionChain']['call'][str(int(strike))] = Df
            D = SetupDLM362(Df,  R['GarchParam'])
            D = RunDLM362(D, Optimize=0)
            SimAssetDLM[Asset]['OptionChain']['callKalman'][str(int(strike))] = D
        SimAssetDLM = Add_ImpVol_DLMresults(SimAssetDLM)
    save(SimAssetDLM,'SimAssetDLM.pkl')
    return SimAssetDLM
def ProcessSimulatedAsset(AssetDLM,PriceDf,ReturnsDf,Asset,OptionChain,RFDf,R,Df,D):
    global DaysPerYear
    DaysPerYear = 365 
    AssetUpper = Asset.upper()
    OptionChain = PreProcessOptionChain(OptionChain, PriceDf, ReturnsDf, RFDf,'Simulated')
    AssetDLM[AssetUpper] = {}
    AssetDLM[AssetUpper]['OptionChain'] = OptionChain
    Garch11Param = AssetDLM[AssetUpper]['OptionChain']['Garch11Param']
    CallStrikeList = list(
        AssetDLM[AssetUpper]['OptionChain']['call'].keys())
    for Strike in CallStrikeList:
        Df = AssetDLM[AssetUpper]['OptionChain']['call'][Strike]
        D = SetupDLM362(Df, Garch11Param)
        D = RunDLM362(D, Optimize=0)
        AssetDLM[AssetUpper]['OptionChain']['callKalman'][Strike] = D
    AssetDLM = Add_ImpVol_DLMresults(AssetDLM)
    save(AssetDLM, 'AssetDLM2.pkl')
    return AssetDLM

def CreateOptionData(PriceDf,ReturnsDf,RFDf,NumStrikes=5,Asset='AAPL'):
    global DaysPerYear
    StockPrice = PriceDf[Asset].values[2:]
    RiskFree   = RFDf['DTB3'].values
    R          = ga.SimulateOption(StockPrice,RiskFree,NumStrikes=5,NumYears=1,DaysPerYear=365)
    #make fake file for processing
    OptionPrice = R['OptionPrice']
    K           = R['K']
    Maturity    = R['Maturity']
    RiskFree    = R['RiskFree']
    Sigma       = R['Sigma']
    SimPrice    = R['SimPrice']
    SimReturn   = R['SimReturn']
    Garch11Param  = R['GarchParam']
    NumStrikes = ['NumStrikes=5']
    for k in range(0,NumStrikes):
        Data = np.vstack((SimReturn,SimPrice,OptionPrice[:,k],K[:,k],Maturity,RiskFree)).T
        cols = ['StockReturn','StockPrice','OptionPrice','Strike','Maturity','RiskFree']
        Df   = pd.DataFrame(Data,columns = cols)
        D    = SetupDLM362(Df, Garch11Param)
        D    = RunDLM362(D, Optimize=0)
    return R,Df,D


# def SetupDLM362(Df, Garch11Param):
#     # print(Df.columns)
#     D = {}
#     D['Garch11Param'] = Garch11Param
#     D['RiskFree'] = Df['RiskFree'].values
#     D['Maturity'] = Df['Maturity'].values
#     D['StockReturn'] = Df['StockReturn'].values
#     D['StockPrice'] = Df['StockPrice'].values
#     D['Strike'] = np.unique(Df['Strike'].values)[0]
#     D['t'] = len(Df['OptionPrice'])
#     D['y'] = np.matrix(Df['OptionPrice'].values).T
#     D['r'] = 1  # Length of state vector
#     D['a'] = np.matrix(np.zeros((D['t'], D['r'])))*np.nan
#     D['R'] = np.zeros((D['t'], D['r']))*np.nan
#     D['m'] = np.matrix(np.ones((D['t'], D['r'])))
#     D['C'] = np.zeros((D['t'], D['r']))
#     D['Wscale'] = .1
#     D['W'] = np.matrix(np.diag(np.ones(D['r'])))*D['Wscale']
#     D['f'] = np.matrix((np.zeros((D['t'], 1))*np.nan))
#     D['Q'] = np.matrix((np.zeros((D['t'], 1))*np.nan))
#     D['e'] = np.matrix((np.zeros((D['t'], 1))*np.nan))
#     D['n'] = np.matrix((np.zeros((D['t'], 1))*np.nan))
#     D['d'] = np.matrix((np.zeros((D['t'], 1))*np.nan))
#     D['S'] = np.matrix((np.zeros((D['t'], 1))*np.nan))
#     D['n'] = np.matrix((np.ones((D['t'], 1))))
#     D['d'] = np.matrix((np.ones((D['t'], 1))))
#     D['d1'] = .9  # Variance discounting
#     D['A'] = np.matrix(np.zeros((D['t'], D['r'])))*np.nan
#     D['LLT'] = np.matrix((np.zeros((D['t'], 1))*np.nan))  # Log likelihood
#     D['m0'] = np.var(Df['StockReturn'])
#     C0      = D['m0']/3
#     D['C0'] = C0
#     return D

def SetupDLM362(Df, Garch11Param, warmup=20, inflate_factor=3.0):
    """
    Build the D dict for Kalman filter.
    Initialize m0,C0 by an intercept-only OLS on the first `warmup` days
    of implied variance^2 (with a percent->decimal safeguard).
    """
    D = {}
    D['Garch11Param'] = Garch11Param
    D['RiskFree']     = Df['RiskFree'].values
    D['Maturity']     = Df['Maturity'].values
    D['StockReturn']  = Df['StockReturn'].values
    D['StockPrice']   = Df['StockPrice'].values
    D['Strike']       = float(Df['Strike'].iloc[0])
    D['t']            = len(Df)
    D['y']            = mat(Df['OptionPrice'].values).T

    # Pre‑allocate all the DLM arrays
    r = 1
    D['r']      = r
    D['a']      = mat(np.zeros((D['t'], r))) * np.nan
    D['R']      = np.zeros((D['t'], r)) * np.nan
    D['m']      = mat(np.ones((D['t'], r)))
    D['C']      = np.zeros((D['t'], r))
    D['Wscale'] = 0.1
    D['W']      = mat(np.eye(r)) * D['Wscale']
    D['f']      = mat(np.zeros((D['t'], 1))) * np.nan
    D['Q']      = mat(np.zeros((D['t'], 1))) * np.nan
    D['e']      = mat(np.zeros((D['t'], 1))) * np.nan
    D['n']      = mat(np.ones((D['t'], 1)))
    D['d']      = mat(np.ones((D['t'], 1)))
    D['d1']     = 0.9     # initial variance‐discount factor
    D['Sv']     = mat(np.zeros((D['t'], 1))) * np.nan
    D['A']      = mat(np.zeros((D['t'], r))) * np.nan
    D['LLT']    = mat(np.zeros((D['t'], 1))) * np.nan

    # --- OLS warm‑up for m0,C0 ---
    N0  = min(warmup, D['t'])
    iv0 = np.zeros(N0)
    for i in range(N0):
        Vm = Df['vwap'].iloc[i] if 'vwap' in Df else Df['OptionPrice'].iloc[i]
        iv_raw = bs.find_vol(
            Vm,
            Df['StockPrice'].iloc[i],
            Df['Strike'].iloc[i],
            Df['Maturity'].iloc[i],
            Df['RiskFree'].iloc[i]
        )
        # if find_vol returns >1 (e.g. percent), convert to decimal
        iv0[i] = (iv_raw/100.0) if iv_raw > 1 else iv_raw

    var0  = iv0**2
    beta0 = var0.mean()
    resid = var0 - beta0
    s2    = resid.var(ddof=1)

    D['m0'] = beta0
    D['C0'] = (s2 / N0) * inflate_factor
    # -----------------------------------

    return D

def RunDLM362(D, Optimize=0):
    """
    Pre‑allocate the arrays and then either:
      - Optimize=0 -> KalmanOptionPricing
      - Optimize=1 or 2 -> your existing hyperparam routines
    This function does NOT overwrite D['m0'],D['C0'].
    """
    # re‑set dims
    T = D['t'] = len(D['y'])
    D['r']     = 1

    # pre‑allocate (same as in Setup)
    D['a']   = mat(np.zeros((T,1))) * np.nan
    D['R']   = np.zeros((T,1)) * np.nan
    D['m']   = mat(np.ones((T,1)))
    D['C']   = np.zeros((T,1))
    D['Wscale'] = D['Wscale']
    D['W']   = mat(np.eye(1)) * D['Wscale']
    D['f']   = mat(np.zeros((T,1))) * np.nan
    D['Q']   = mat(np.zeros((T,1))) * np.nan
    D['e']   = mat(np.zeros((T,1))) * np.nan
    D['n']   = mat(np.ones((T,1)))
    D['d']   = mat(np.ones((T,1)))
    D['Sv']  = mat(np.zeros((T,1))) * np.nan
    D['A']   = mat(np.zeros((T,1))) * np.nan
    D['LLT'] = mat(np.zeros((T,1))) * np.nan

    if Optimize == 0:
        # pass exactly the two hyperparams the function expects:
        D = KalmanOptionPricing(D, D['d1'], D['Wscale'])
    elif Optimize == 1:
        # … your Optimize=1 code …
        pass
    elif Optimize == 2:
        # … your Optimize=2 code …
        pass

    return D


def LLfun(x, D):
    D = KalmanOptionPricing_Opt(x, D)
    LL = D['LL']
    return LL


def KalmanOptionPricing(D, VarianceDiscounting, Wscale):
    """
    Run the Kalman‐filter recursions with
      - D['m0'], D['C0'] as priors,
      - VarianceDiscounting = discount factor d1,
      - Wscale = state‐noise variance.
    """
    p = D['t']
    # unpack
    y  = D['y']
    a, R = D['a'], D['R']
    m, C = D['m'], D['C']
    f, Q = D['f'], D['Q']
    LLT  = D['LLT']
    e    = D['e']
    n, d = D['n'], D['d']
    Sv   = D['Sv']
    A    = D['A']

    # use the priors you set in SetupDLM362
    m0, C0 = D['m0'], D['C0']

    # use the passed‐in hyperparams
    d1 = VarianceDiscounting
    W  = Wscale

    # GARCH(1,1) params
    omega, alpha, beta = D['Garch11Param']
    u     = D['StockReturn']
    S     = D['StockPrice']
    K     = D['Strike']
    Tmat  = D['Maturity']
    rfr   = D['RiskFree']

    # initialize
    n[0]  = 5
    d[0]  = 0.001
    Sv[0] = d[0]/n[0]
    m[0]  = m0
    C[0]  = C0

    phi = np.zeros((p, 1))

    for i in range(1, p):
        # 1) Prior
        a[i] = omega + alpha*(u[i-1]**2) + beta*m[i-1]
        R[i] = beta**2 * mat(C[i-1]) + W

        # 2) Forecast
        f[i] = bs.black_scholes(S[i], K, rfr[i], Tmat[i], np.sqrt(a[i]), option_type="call")
        phi[i] = Calc_phi(S[i], K, rfr[i], a[i], Tmat[i])
        factor = 0.5 * S[i] * np.sqrt(Tmat[i]/a[i]) * phi[i]

        # 3) Forecast‐error variance
        Sv[i] = d[i-1]/n[i-1]
        Q[i]  = (factor**2)*R[i] + Sv[i]

        # 4) Update
        e[i]   = y[i] - f[i]
        n[i]   = d1*n[i-1] + 1
        d[i]   = d1*d[i-1] + Sv[i-1]*( (e[i]**2)/Q[i] )
        A[i]   = (factor * R[i]) / Q[i]
        m[i]   = a[i] + A[i]*e[i]
        C[i]   = R[i] - factor*R[i]*(1/Q[i])*factor*R[i]

        # 5) Log‑likelihood
        density = pdft(y[i], n[i]-1, f[i], Q[i])
        LLT[i] = np.log(density) if density>1e-323 else np.log(1e-323)

    # pack up
    D['a'], D['R'] = a, R
    D['m'], D['C'] = m, C
    D['f'], D['Q'] = f, Q
    D['e'], D['n'], D['d'], D['Sv'], D['A'], D['LLT'] = e, n, d, Sv, A, LLT
    D['LL'] = np.sum(LLT)

    return Analysis(D)



# def LLfun(x, D):
#     D = KalmanOptionPricing_Opt(x, D)
#     LL = D['LL']
#     return LL


# def KalmanOptionPricing(D, VarianceDiscounting, Wscale):
#     p = D['t']  # Number of time periods
#     y = D['y']  # Observed y
#     a = D['a']  # Prior of state vector
#     R = D['R']  # Covariance of prior of state vector
#     m = D['m']  # Posterior of state vector
#     C = D['C']  # Covariance of posterior of state vector
#     W = 0.1  # State error covariance
#     f = D['f']  # One step ahead forecast
#     Q = D['Q']  # Variance of forecast
#     LLT = D['LLT']  # Log likelihood of observation
#     e = D['e']  # Observed forecast error
#     n = D['n']  # Degrees of freedom
#     d = D['d']  # Sum of squared errors
#     Sv = D['Sv']  # Estmate of observation equation error variance
#     d1 = D['d1']  # Variance discount factor
#     A = D['A']
#     m0 = D['m0']
#     C0 = D['C0']
#     n[0] = 5
#     d[0] = .001
#     Sv[0] = d[0]/n[0]
#     m[0] = m0
#     C[0] = C0
#     GarchParam = D['Garch11Param']
#     omega = GarchParam[0]  # Garch parameters
#     alpha = GarchParam[1]
#     beta = GarchParam[2]
#     u = D['StockReturn']  # Stock returns se eq 1.4
#     phi = np.zeros((len(u), 1))
#     S = D['StockPrice']
#     K = D['Strike']
#     Maturity = D['Maturity']
#     RiskFree = D['RiskFree']
#     # Assume posterior values of m, C exist at t = 0
#     # LLT[0] = pdft(y[0],n[0],y[0],Sv[0])
#     for i in range(1, p):  # Kalman (DLM) recursions over time periods
#         S1 = S[i]        # Stock price
#         K1 = K        # Call option strike price
#         M1 = Maturity[i]  # Option maturity in years
#         r1 = RiskFree[i]  # Risk free rate
#         a[i] = (omega + alpha*u[i-1]*u[i-1] + beta * m[i-1]
#                 )[0, 0]  # Prior states, Garch(1,1) dynamics
#         R[i] = (beta*beta*mat(C[i-1]) + W)[0, 0]  # Covariance of prior states
#         # print([S1, K1, r1, M1, np.sqrt(a[i])[0,0]])
#         f[i] = bs.black_scholes(S1, K1, r1, M1, np.sqrt(a[i])[
#                                 0, 0], option_type="call")
#         # print('Forecast at i = ' + str(i) + ' is ' + str(f[i][0,0]))
#         phi[i] = Calc_phi(S1, K1, r1, a[i], M1)[0, 0]
#         factor = .5 * S1 * np.sqrt(M1/a[i]) * phi[i]  # See eq 1.19 in notes
#         Sv[i] = d[i-1]/n[i-1]  # Estimated observational error variance
#         # Variance of forecast, see eq 1.17 in notes
#         Q[i] = (factor*factor)*R[i] + Sv[i]
#         e[i] = y[i] - f[i]
#         n[i] = d1*n[i-1] + 1  # Increment degrees of freedom
#         d[i] = d1*d[i-1] + Sv[i-1]*((e[i]*e[i])/Q[i])
#         A[i] = (factor*R[i] / Q[i])  # See notes eq 1.27
#         # if A[i] >=5:
#         #     A[t] = .01
#         # print('a[i] = ' + str(a[i]) + ' m[i-1] = ' + str(m[i-1]))
#         # print('e[i] = ' + str(e[i]) + ' A[i] = ' + str(A[i]))
#         m[i] = a[i] + A[i]*e[i]  # Posterior states
#         # if m[i] <.001:
#         #     m[i] = np.median(np.ravel(D['m'][0:i-1]))
#         # Covariance of posterior states
#         C[i] = R[i]-factor*R[i]*(1/Q[i])*factor*R[i]  # See eq 1.29
#         density = pdft(y[i], n[i]-1, f[i], Q[i])
#         LLT[i] = np.log(1e-323)
#         if density > 1e-323:
#             LLT[i] = np.log(density)  # Log likelihood of the observation
#     # Put all back on D
#     D['a'] = a
#     D['m'] = m
#     D['A'] = A
#     D['R'] = R
#     D['C'] = C
#     D['f'] = f
#     D['Q'] = Q
#     D['e'] = e
#     D['d'] = d
#     D['n'] = n
#     D['Sv'] = Sv
#     D['LLT'] = LLT
#     D['LL'] = np.sum((D['LLT']))
#     D = Analysis(D)
#     return D


def KalmanOptionPricing_Opt(x, D):
    GarchParam = list(x[0:3])
    VarianceDiscounting = x[3]
    Wscale = x[4]
    D['GarchParam'] = GarchParam
    D['VarianceDiscounting'] = VarianceDiscounting
    D['Wscale'] = Wscale
    p = D['t']  # Number of time periods
    y = D['y']  # Observed y
    a = D['a']  # Prior of state vector
    R = D['R']  # Covariance of prior of state vector
    m = D['m']  # Posterior of state vector
    C = D['C']  # Covariance of posterior of state vector
    W = Wscale  # State error covariance
    f = D['f']  # One step ahead forecast
    Q = D['Q']  # Variance of forecast
    LLT = D['LLT']  # Log likelihood of observation
    e = D['e']  # Observed forecast error
    n = D['n']  # Degrees of freedom
    d = D['d']  # Sum of squared errors
    Sv = D['Sv']  # Estmate of observation equation error variance
    d1 = VarianceDiscounting  # Variance discount factor
    A = D['A']
    m0 = D['m0']
    C0 = D['C0']
    n[0] = 5
    d[0] = .001
    Sv[0] = d[0]/n[0]
    m[0] = m0
    C[0] = C0
    omega = GarchParam[0]  # Garch parameters
    alpha = GarchParam[1]
    beta = GarchParam[2]
    u = D['u']  # Stock returns se eq 1.4
    phi = np.zeros((len(u), 1))
    S = D['S']
    K = D['K']
    Maturity = D['Maturity']
    Riskfreerate = D['Riskfreerate']
    # Assume posterior values of m, C exist at t = 0
    LLT[0] = pdft(y[0], n[0], y[0], Sv[0])
    for i in range(1, p):  # Kalman (DLM) recursions over time periods
        S1 = S[i][0]        # Stock price
        K1 = K[i][0]        # Call option strike price
        M1 = Maturity[i][0]  # Option maturity in years
        r1 = Riskfreerate[i][0]  # Risk free rate
        a[i] = omega + alpha*u[i-1]*u[i-1] + beta * \
            m[i-1]  # Prior states, Garch(1,1) dynamics
        R[i] = beta*beta*mat(C[i-1]) + W  # Covariance of prior states
        f[i] = bs.black_scholes(
            S1, K1, r1, M1, np.sqrt(a[i]), option_type="call")
        phi[i] = Calc_phi(S1, K1, r1, a[i], M1)
        factor = .5 * S1 * np.sqrt(M1/a[i]) * phi[i]  # See eq 1.19 in notes
        Sv[i] = d[i-1]/n[i-1]  # Estimated observational error variance
        # Variance of forecast, see eq 1.17 in notes
        Q[i] = (factor*factor)*R[i] + Sv[i]
        e[i] = y[i] - f[i]
        n[i] = d1*n[i-1] + 1  # Increment degrees of freedom
        d[i] = d1*d[i-1] + Sv[i-1]*((e[i]*e[i])/Q[i])
        A[i] = (factor*R[i] / Q[i])  # See notes eq 1.27
        # if A[i] >=5:
        #     A[t] = .01
        # print('a[i] = ' + str(a[i]) + ' m[i-1] = ' + str(m[i-1]))
        # print('e[i] = ' + str(e[i]) + ' A[i] = ' + str(A[i]))
        m[i] = a[i] + A[i]*e[i]  # Posterior states
        # if m[i] <.001:
        #     m[i] = np.median(np.ravel(D['m'][0:i-1]))
        # Covariance of posterior states
        C[i] = R[i]-factor*R[i]*(1/Q[i])*factor*R[i]  # See eq 1.29
        density = pdft(y[i], n[i]-1, f[i], Q[i])
        LLT[i] = np.log(1e-323)
        if density > 1e-323:
            LLT[i] = np.log(density)  # Log likelihood of the observation
    # Put all back on D
    D['a'] = a
    D['m'] = m
    D['A'] = A
    D['R'] = R
    D['C'] = C
    D['f'] = f
    D['Q'] = Q
    D['e'] = e
    D['d'] = d
    D['n'] = n
    D['Sv'] = Sv
    D['LLT'] = LLT
    D['LL'] = -np.sum((D['LLT']))
    # D = Analysis(D)
    return D


