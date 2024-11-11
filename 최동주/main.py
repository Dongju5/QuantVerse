import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrx import stock
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 시작일과 종료일 설정
start_date = "2000-01-01"
end_date = "2024-10-31"

# 사용 방법: python script.py <종목명>
if len(sys.argv) != 2:
    print("Usage: python script.py <종목명>")
    sys.exit(1)

# 종목명 입력받기
target_name = sys.argv[1]

# pykrx로 상장된 종목코드와 이름 가져오기
tickers = stock.get_market_ticker_list(market="KOSPI")
ticker_info = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}

# 특정 종목 코드 가져오기
target_ticker = None
for ticker, name in ticker_info.items():
    if name == target_name:
        target_ticker = ticker
        break

if target_ticker is None:
    print(f"종목명 '{target_name}'을(를) 찾을 수 없습니다. 다시 확인해주세요.")
    sys.exit(1)

try:
    # 특정 종목의 주식 데이터 가져오기
    df_krx = stock.get_market_ohlcv_by_date(start_date, end_date, target_ticker)

    # 필요한 정보 추출 (시가, 고가, 저가, 종가, 거래량)
    df_krx = df_krx[['시가', '고가', '저가', '종가', '거래량']].reset_index()
    df_krx['ticker'] = target_ticker
    df_krx['name'] = target_name
    df_krx.rename(columns={'시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close', '거래량': 'volume', '날짜': 'date'}, inplace=True)

    # 날짜 순서로 정렬
    df_krx.sort_values(by='date', inplace=True)

    # 주요 지표 계산 (RSI, 이동평균선, MACD)
    # RSI 계산
    window_length = 14
    delta = df_krx['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window_length, min_periods=1).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df_krx['RSI'] = 100 - (100 / (1 + rs))

    # 이동평균선 계산
    df_krx['SMA_20'] = df_krx['close'].rolling(window=20).mean()  # 20일 단순 이동평균선 (SMA)
    df_krx['SMA_50'] = df_krx['close'].rolling(window=50).mean()  # 50일 단순 이동평균선 (SMA)

    # MACD 계산
    ema_short = df_krx['close'].ewm(span=12, adjust=False).mean()
    ema_long = df_krx['close'].ewm(span=26, adjust=False).mean()
    df_krx['MACD'] = ema_short - ema_long
    df_krx['MACD_signal'] = df_krx['MACD'].ewm(span=9, adjust=False).mean()

    # 주요 지표 포함 데이터 출력
    print(f"\n{target_name} 주요 지표 포함 데이터:")
    print(df_krx.tail())

    # 레이블 설정
    df_krx['label_avg'] = (df_krx['close'] > df_krx['close'].rolling(window=30).mean() * 1.2).astype(int)

    # 레이블 완료 데이터 출력
    print("\n레이블 완료 데이터:")
    print(df_krx[['date', 'close', 'label_avg']].tail())

    # 학습 및 테스트 데이터 준비
    features = ['RSI', 'SMA_20', 'SMA_50', 'MACD']
    df_combined = df_krx[features + ['label_avg']].dropna()
    X = df_combined[features]
    y = df_combined['label_avg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    # 훈련 및 테스트 데이터 준비 완료 출력
    print("\n훈련 및 테스트 데이터 준비 완료:")
    print(f"훈련 데이터 크기: {X_train.shape}, 테스트 데이터 크기: {X_test.shape}")

    # 랜덤 포레스트 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    print("\n모델 평가 결과:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # 시각화
    plt.figure(figsize=(14, 10))

    # 1. 종가와 이동평균선 그래프
    plt.subplot(3, 1, 1)
    plt.plot(df_krx['date'], df_krx['close'], label='Close Price', color='blue')
    plt.plot(df_krx['date'], df_krx['SMA_20'], label='SMA 20', color='orange')
    plt.plot(df_krx['date'], df_krx['SMA_50'], label='SMA 50', color='green')
    plt.title(f'{target_name} Stock Price with SMA 20 and SMA 50')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # 2. RSI 그래프
    plt.subplot(3, 1, 2)
    plt.plot(df_krx['date'], df_krx['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title('RSI Indicator')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.legend()

    # 3. MACD 그래프
    plt.subplot(3, 1, 3)
    plt.plot(df_krx['date'], df_krx['MACD'], label='MACD', color='blue')
    plt.plot(df_krx['date'], df_krx['MACD_signal'], label='MACD Signal', color='red')
    plt.title('MACD Indicator')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error retrieving data for ticker {target_ticker}: {e}")
