import pandas as pd
def tech_financial(df, n=12, n_fast=12, n_slow=26):

    # EMA (Exponential Moving Average)
    EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n - 1).mean(), name='EMA_' + str(n))
    df['EMA'] = EMA

    # MOM (Momentum)
    M = pd.Series(df['Close'].diff(n), name = 'Momentum_' + str(n))
    df['MOM'] = M

    # ROC (Rate of Change)
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))
    df['ROC'] = ROC

    # Moving Average
    df['12day MA'] = df['Close'].shift(1).rolling(window = 12).mean()
    df['26day MA'] = df['Close'].shift(1).rolling(window = 26).mean()

    # Standard Deviation
    df['Std_dev']= df['Close'].rolling(5).std()

    # Coppock Curve
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series((ROC1 + ROC2).ewm(span = n, min_periods = n).mean(), name = 'Copp_' + str(n))
    df['Copp'] = Copp

    # Bollinger Bands
    MA = pd.Series(df['Close'].rolling(n).mean())
    MSD = pd.Series(df['Close'].rolling(n).std())
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))
    df['B1'] = B1
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))
    df['B2'] = B2

    # Trix
    EX1 = df['Close'].ewm(span = n, min_periods = n - 1).mean()
    EX2 = EX1.ewm(span = n, min_periods = n - 1).mean()
    EX3 = EX2.ewm(span = n, min_periods = n - 1).mean()
    i = 0
    ROC_l = [0]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))
    df['Trix'] = Trix

    # MACD, MACD Signal and MACD difference
    EMAfast = pd.Series(df['Close'].ewm(span = n_fast, min_periods = n_slow - 1).mean())
    EMAslow = pd.Series(df['Close'].ewm(span = n_slow, min_periods = n_slow - 1).mean())
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span = 9, min_periods = 8).mean(), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df['MACD'] = MACD
    df['MACDsign'] = MACDsign
    df['MACDdiff'] = MACDdiff

    # Return final dataframe
    return df
