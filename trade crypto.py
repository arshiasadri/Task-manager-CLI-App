import argparse
import os
import numpy as np
import X as pd
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
import datetime
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler
from pruject.trade import scaled_data, scaler
def add_technical_indicators(df):
    df = df.copy()
    df['return'] = df['Close'].pct_change()
    df['ma7'] = df['Close'].rolling(7).mean()
    df['ma21'] = df['Close'].rolling(21).mean()
    df['ema12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['Close'].ewm(span=26, adjust=False).mean()
    scaled_data = scaler.fit_transform(df[['Close']].values)
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9,adjust=False).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low,high_close,low_close],axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14).mean()
    df['vol_change'] = df['Volume'].pct_change()
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    df['close_ma7_ratio'] = df['Close'] / df['ma7'] - 1
    df['close_ma21_ratio'] = df['Close'] / df['ma21'] - 1
    df = df.dropna()
    return df

def create_lstm_sequences(values, seq_len):
    X,y = [],[]
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len,0])
    return np.array(X),np.array(y)

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(128,return_sequences=True, input_shape= input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse',metrics=['mae'])
    return model

def backtest_with_risk_management(dates,prices,signals, atrs, initial_capital=10000.0,
                                  risk_per_trade=0.01,sl_atr=3.0,tp_atr=6.0,
                                  fee=0.001,slippage=0.0005):
    cash = initial_capital
    position = 0.0
    entry_price = 0.0
    stop_price = None
    tp_price = None
    pv = []
    trades = []
    for i in range(len(prices)):
        price = prices[i]
        date = dates[i]
        sig = signals[i]
        atr = atrs[i] if not np.isnan(atrs[i]) else 0.0
        if position > 0:
            if price <= stop_price :
                proceeds = position * price * (1 - fee - slippage)
                cash = proceeds
                trades.append((date, 'SELL_STOP',price, position))
                position = 0.0
                stop_price = None
                tp_price = None
            elif price >= tp_price:
                proceeds = position * price * (1 - fee - slippage)
                cash = proceeds
                trades.append((date, 'SELL_TP',price,position))
                position = 0.0
                stop_price = None
                tp_price = None
        if sig == 1 and position == 0:
            if atr <= 0:
                qty = (cash * 0.1) / (price * (1 + slippage))
            else:
                dollar_risk_per_share = atr * sl_atr
                dollar_risk = cash * risk_per_trade
                qty = dollar_risk / dollar_risk_per_share if dollar_risk_per_share>0 else (cash * 0.1) / price
                cost_est = qty * price * (1 + slippage + fee)
                if cost_est > cash:
                    qty = cash / (price * (1 + slippage + fee))
            if qty <= 0:
                pv.append(cash + position * price)
                continue
            entry_price = price * (1 + slippage)
            stop_price = entry_price - sl_atr * atr
            tp_price = entry_price + tp_atr * atr
            position = qty
            cash = cash - qty * entry_price * (1 + fee)
            trades.append((date, 'BUY', entry_price, qty))
        elif sig == 0 and position > 0:
            proceeds = position * price * (1 - fee - slippage)
            cash = proceeds
            trades.append((date,'SELL_SIGNAL',price,position))
            position = 0.0
            stop_price = None
            tp_price = None
        pv.append(cash + position * price)
    final_value = cash + position * prices[-1]
    pv_series = pd.Series(pv, index=dates[:len(pv)])
    returns = pv_series.pct_change().dropna()
    total_return = (pv_series.iloc[-1] / pv_series.iloc[0]) - 1 if len(pv_series)>0 else 0
    days = (dates[len(pv_series)-1] - dates[0]).days if len(pv_series)>0 else 1
    annual_factor = 365.25
    cagr = (pv_series.iloc[-1] / pv_series.iloc[0]) ** (annual_factor / days) - 1 if days>0 and len(pv_series)>0 else 0
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) >1 else 0
    cummax = pv_series.cummax()
    drawdown = (pv_series - cummax) / cummax
    max_dd = drawdown.min()

    metrics = {
        'final_value' : float(pv_series.iloc[-1]) if len(pv_series)>0 else float(initial_capital),
        'total_return' : float(total_return),
        'cagr' : float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown':float(max_dd)
    }
    return pv_series,trades,metrics
def run_pipeline(ticker='BTC-USD',start='2018-01-01',end=None,interval='1d',seq_len=60,
                 test_size=0.2, epochs=25, batch_size=32, save_dir='models',optimize =False,
                 n_trials=50, risk_per_trade=0.01,sl_atr=3.0, tp_atr=6.0):
    os.makedirs(save_dir, exist_ok=True)
    reports_dir = 'reports'
    os.makedirs(reports_dir,exist_ok=True)
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
        print(f"Downloading {ticker} from {start} to {end} interval={interval}...")
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            raise ValueError('No data download. Check ticker/interval or internet connection')
        df = add_technical_indicators(df)
        features = ['Close','return','ma7','ma21','ema12','ema26','macd','macd_signal','rsi','atr','vol_change','vol_ma20','close_ma7_ratio','close_ma21_ratio']
        df_feat = df[features].dropna()
        scaler = MinMaxScaler()
        values = scaler.fit_transform(df_feat.values)
        X_lstm,y_lstm = create_lstm_sequences(values,seq_len)
        X_xgb = []
        y_xgb = []
        for i in range(seq_len, len(values)):
            X_xgb.append(values[i])
            y_xgb.append(values[i,0])
        X_xgb = np.array(X_xgb)
        y_xgb = np.array(y_xgb)
        dates = df_feat.index[seq_len:]
        split_idx = int(len(X_lstm) *(1-test_size))
        X_lstm_train, X_lstm_test = X_lstm[:split_idx],X_lstm[split_idx:]
        y_lstm_train,y_lstm_test = y_lstm[:split_idx], y_lstm[split_idx:]
        X_xgb_train,X_xgb_test = X_xgb[:split_idx], X_xgb[split_idx:]
        y_xgb_train,y_xgb_test = y_xgb[:split_idx], y_xgb[split_idx:]
        dates_test = dates[split_idx:]
        X_lstm_train = X_lstm_train.reshape((X_lstm_train.shape[0],X_lstm_train.shape[1],X_lstm_train.shape[2]))
        X_lstm_test = X_lstm_test.reshape((X_lstm_test.shape[0],X_lstm_test.shape[1],X_lstm_test.shape[2]))
        lstm = build_lstm(input_shape=(seq_len,X_lstm_train.shape[2]))
        callbacks = [EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True),ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=4,min_lr=1e-6)]
        lstm.fit(X_lstm_train,y_lstm_train,validation_data=(X_lstm_test,y_lstm_test),epochs=epochs,batch_size=batch_size,callbacks=callbacks,verbose=0)
        preds_lstm =lstm.predict(X_lstm_test).flatten()
        xgb_model = xgb.XGBRFRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,random_state=42)
        xgb_model.fit(X_xgb_train,y_xgb_train,eval_set=[(X_xgb_test,y_xgb_test)],early_stopping_rounds=20,verbose=False)
        preds_xgb = xgb_model.predict(X_xgb_test)
        def inverse_close(scaled_close_arr):
            filler = np.zeros((len(scaled_close_arr),values.shape[1]))
            filler[:,0] = scaled_close_arr
            inv = scaler.inverse_transform(filler)[:,0]
            return inv
        preds_lstm_price = inverse_close(preds_lstm)
        preds_xgb_price = inverse_close(preds_xgb)
        y_test_price = inverse_close(y_lstm_test)
        today_price = inverse_close(X_lstm_test[:,-1,0])
        atr_test = df['atr'].values[seq_len+split_idx: seq_len + split_idx + len(y_test_price)]
        def objective(trial):
            w_l = trial.suggest_float('w_lstm',0.0,1.0)
            w_x = 1.0 - w_l
            threshold = trial.suggest_float('threshold', -0.05, 0.05)
            preds_ens = w_l * preds_lstm_price + w_x * preds_xgb_price
            signals = (preds_ens > today_price * (1 + threshold)).astype(int)
            pv_series, trades, metrics = backtest_with_risk_management(dates_test,y_test_price, signals, atr_test, initial_capital=10000.0, risk_per_trade=risk_per_trade, sl_atr=sl_atr, tp_atr=tp_atr)
            return -metrics['sharpe']
        best = None
        if optimize:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            best = study.best_params
            print('Optuna best params:', best)
            w_l = best['w_lstm']
            w_x = 1.0 - w_l
            threshold = best['threshold']
        else:
            w_l,w_x = 0.5, 0.5
            threshold = 0.0
        preds_ens = w_l * preds_lstm_price + w_x * preds_xgb_price
        signals = (preds_ens > today_price * (1 + threshold)).astype(int)
        pv_series, trades, metrics = backtest_with_risk_management(dates_test, y_test_price, signals, atr_test, initial_capital=10000.0,risk_per_trade=risk_per_trade, sl_atr=sl_atr, tp_atr=tp_atr)
        lstm_path = os.path.join(save_dir, f'lstm_{ticker.replace("/","-")}.h5')
        xgb_path = os.path.join(save_dir, f'xgb_{ticker.replace("/","-")}.joblib')
        scaler_path = os.path.join(save_dir, f'scaler_{ticker.replace("/","-")}.joblib')
        lstm.save(lstm_path)
        joblib.dump(xgb_model, xgb_path)
        joblib.dump(scaler, scaler_path)
        trades_df = pd.DataFrame(trades, columns=['date','side','price','qty'])
        trades_path = os.path.join(save_dir, f'trades_{ticker.replace("/","-")}.csv')
        trades_df.to_csv(trades_path, index=False)
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(reports_dir, f'report_{ticker.replace("/","-")}_{now}.pdf')
        with PdfPages(report_path) as pdf :
            plt.figure(figsize=(10,6))
            plt.plot(dates_test, y_test_price, label = 'Actual')
            plt.plot(dates_test, preds_ens, label = 'Ensemble Pred')
            plt.title(f'{ticker} Actual vs Predicted')
            plt.legend()
            plt.tight_layout()
            pdf.savefig(); plt.close()
            plt.figure(figsize=(10,6))
            plt.plot(pv_series.index, pv_series.values)
            plt.title('Backtest Portfolio Value')
            plt.tight_layout()
            pdf.savefig(); plt.close()
            fig, ax = plt.subplots(figsize=(10,6))
            ax.axis('off')
            tbl = trades_df.tail(50)
            ax.table(cellText=tbl.values, colLabels=tbl.columns, loc='center')
            plt.title('Recent Trades')
            pdf.savefig(); plt.close()
            fig, ax = plt.subplots(figsize=(8,6))
            ax.axis('off')
            txt = '\n'.join([f"{k}: {v:.4f}" if isinstance(v,float) else f"{k}: {v}" for k,v in metrics.items()])
            ax.text(0.01,0.99,txt, va ='top', fontsize=12)
            plt.title('Backtest Metrics')
            pdf.savefig(); plt.close()
        print('Saved report to', report_path)
        print('Metrics:', metrics)
        if best is not None :
            print('Best optuna params:', best)
        return {
            'dates_test': dates_test,
            'y_test_price': y_test_price,
            'preds_ens': preds_ens,
            'pv_series': pv_series,
            'trades': trades,
            'metrics': metrics,
            'report': report_path,
            'models': {'lstm': lstm_path, 'xgb': xgb_path, 'scaler': scaler_path}
        }
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='BTC-USD')
    parser.add_argument('--start', type=str, default='2018-01-01')
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--interval', type=str, default='1d', choices=['1d','1h','15m'])
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--trials',type=int,default=50)
    parser.add_argument('--risk_per_trade', type=float, default=0.01)
    parser.add_argument('--sl_atr', type=float, default=3.0)
    parser.add_argument('--tp_atr', type=float, default=6.0)
    args = parser.parse_args()
    results = run_pipeline(ticker=args.ticker, start=args.start, end=args.end, interval=args.interval, seq_len=args.seq_len,
                           test_size=args.test_size, epochs=args.epochs, batch_size=args.batch_size, save_dir=args.save_dir,
                           optimize=args.optimize, n_trials=args.trials, risk_per_trade=args.risk_per_trade, sl_atr=args.sl_atr, tp_atr=args.tp_atr)
    print('\nDone. Report and models saved.')