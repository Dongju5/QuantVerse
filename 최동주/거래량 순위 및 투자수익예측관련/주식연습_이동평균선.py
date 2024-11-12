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

    # 주요 지표 계산 (이동평균선)
    # 이동평균선 계산
    df_krx['SMA_20'] = df_krx['close'].rolling(window=20).mean()  # 20일 단순 이동평균선 (SMA)
    df_krx['SMA_50'] = df_krx['close'].rolling(window=50).mean()  # 50일 단순 이동평균선 (SMA)

    # 초기 자본 설정
    initial_capital = 10000000  # 1천만 원
    capital = initial_capital
    shares = 0

    # 가상 주식 실행 시작 날짜 설정
    simulation_start_date = "2023-10-31"
    df_krx = df_krx[df_krx['date'] >= simulation_start_date]

    # 거래 시뮬레이션
    for i in range(len(df_krx)):
        row = df_krx.iloc[i]
        date = row['date']
        close_price = row['close']
        sma_20 = row['SMA_20']
        sma_50 = row['SMA_50']

        # 매수 조건: 단기 이동평균선(SMA 20)이 장기 이동평균선(SMA 50)을 상향 돌파할 때 (골든 크로스)
        if sma_20 > sma_50 and capital > close_price:
            shares_to_buy = capital // close_price
            capital -= shares_to_buy * close_price
            shares += shares_to_buy
            print(f"{date}: 매수 - {shares_to_buy} 주 @ {close_price}원, 남은 자본: {capital}원")

        # 매도 조건: 단기 이동평균선(SMA 20)이 장기 이동평균선(SMA 50)을 하향 돌파할 때 (데드 크로스)
        elif sma_20 < sma_50 and shares > 0:
            capital += shares * close_price
            print(f"{date}: 매도 - {shares} 주 @ {close_price}원, 총 자본: {capital}원")
            shares = 0

    # 마지막 날 모든 주식 매도
    if shares > 0:
        final_close_price = df_krx.iloc[-1]['close']
        capital += shares * final_close_price
        print(f"{df_krx.iloc[-1]['date']}: 최종 매도 - {shares} 주 @ {final_close_price}원, 총 자본: {capital}원")
        shares = 0

    # 최종 수익 계산
    profit = capital - initial_capital
    print(f"최종 자산: {capital}원, 총 수익: {profit}원")

except Exception as e:
    print(f"Error retrieving data for ticker {target_ticker}: {e}")