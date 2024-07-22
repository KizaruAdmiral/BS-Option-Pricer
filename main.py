import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import seaborn as sns

#Page Setting
st.markdown('''<style>.appview-container .main .block-container {{padding-top: {padding_top}rem;
padding-bottom: {padding_bottom}rem;}}'''.format(padding_top=1, padding_bottom=1),unsafe_allow_html=True,)

st.markdown(
    r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)

# Model Parameters Setting
sidebar = st.sidebar.header("Model Parameters")
Spot_Price = st.sidebar.number_input("Spot Price", value = 100.00, format="%.4f")
Exercise_Price = st.sidebar.number_input("Exercise Price", value = 120.00, format="%.4f")
Risk_free = st.sidebar.number_input("Risk-free rate (in numeric value)", value=0.025, format="%.4f")
start = datetime.datetime(2024, 1, 31)
end = datetime.datetime(2024, 12, 31)
Initial_date = st.sidebar.date_input("Initial Date", value = start)
End_date = st.sidebar.date_input("End Date", value = end)
Volatility = st.sidebar.number_input("Volatility (in numeric value)", value = 0.500, format="%.4f")

#YearFrac
import streamlit as st
import datetime

def IsLeapYear(year):
    if year % 4 > 0:
        IsLeapYear = False
    elif year % 100 > 0:
        IsLeapYear = True
    elif year % 400 == 0:
        IsLeapYear = True
    else:
        IsLeapYear = False
    return IsLeapYear

def IsEndOfMonth(day, month, year):
    if month in [1,3,5,7,8,10,12]:
        IsEndOfMonth = (day == 31)
    if month in [4,6,9,11]:
        IsEndOfMonth = (day == 30)
    if month in [2]:
        if IsLeapYear(year):
            IsEndOfMonth = (day == 29)
        else:
            IsEndOfMonth = (day == 28)
    return IsEndOfMonth

def Days360(StartYear, EndYear, StartMonth, EndMonth, StartDay, EndDay):
    Days360 = ((EndYear - StartYear) * 360) + ((EndMonth - StartMonth) * 30) + (EndDay - StartDay)
    return Days360

def TmpDays360Nasd(StartDate, EndDate, Method, UseEom):
    StartDay = StartDate.day
    StartMonth = StartDate.month
    StartYear = StartDate.year
    EndDay = EndDate.day
    EndMonth = EndDate.month
    EndYear = EndDate.year
    if (EndMonth == 2 and IsEndOfMonth(EndDay, EndMonth, EndYear)) and ((StartMonth == 2 and IsEndOfMonth(StartDay, StartMonth, StartYear)) or Method == 3):
        EndDay = 30     
    if EndDay == 31 and (StartDay >= 30 or Method == 3):
        EndDay = 30
    if StartDay == 31:
        StartDay = 30
    if (UseEom and StartMonth == 2 and IsEndOfMonth(StartDay, StartMonth, StartYear)):
        StartDay = 30   
    return Days360(StartYear, EndYear, StartMonth, EndMonth, StartDay, EndDay)

def TmpDays360Euro(StartDate, EndDate):
    StartDay = StartDate.day
    StartMonth = StartDate.month
    StartYear = StartDate.year
    EndDay = EndDate.day
    EndMonth = EndDate.month
    EndYear = EndDate.year
    if StartDay == 31:
        StartDay = 30
    if EndDay == 31:
        EndDay = 30
    return Days360(StartYear, EndYear, StartMonth, EndMonth, StartDay, EndDay)

def DateDiff(StartDate, EndDate):
    return abs((StartDate - EndDate).days)

def TmpDiffDates(StartDate, EndDate, Basis):
    if Basis in [0]:
        TmpDiffDates = TmpDays360Nasd(StartDate, EndDate, 0, True)
    elif Basis in [1,2,3]:
        TmpDiffDates = DateDiff(StartDate, EndDate)
    elif Basis in [4]:
       TmpDiffDates = TmpDays360Euro(StartDate, EndDate)
    return TmpDiffDates

def TmpCalcAnnualBasis(StartDate, EndDate, Basis):
    if Basis in [0,2,4]:
        TmpCalcAnnualBasis = 360
    elif Basis in [3]:
        TmpCalcAnnualBasis = 365
    elif Basis in [1]:
        StartDay = StartDate.day
        StartMonth = StartDate.month
        StartYear = StartDate.year
        EndDay = EndDate.day
        EndMonth = EndDate.month
        EndYear = EndDate.year

        if StartYear == EndYear:
            if IsLeapYear(StartYear):
                TmpCalcAnnualBasis = 366
            else:
                TmpCalcAnnualBasis = 365
        elif (EndYear - 1) == StartYear and (StartMonth > EndMonth or (StartMonth == EndMonth and StartDay >= EndDay)):
            if IsLeapYear(StartYear):        
                if StartMonth < 2 or (StartMonth == 2 and StartDay <= 29):
                    TmpCalcAnnualBasis = 366
                else:
                    TmpCalcAnnualBasis = 365
            elif IsLeapYear(EndYear):
                if EndMonth > 2 or (EndMonth == 2 and EndDay == 29):
                    TmpCalcAnnualBasis = 366
                else:
                    TmpCalcAnnualBasis = 365  
            else:
                TmpCalcAnnualBasis = 365
        else:
            TmpCalcAnnualBasis = 0
            for iYear in range(StartYear, EndYear + 1 ): 
                if IsLeapYear(iYear):
                    TmpCalcAnnualBasis = TmpCalcAnnualBasis + 366
                else:
                    TmpCalcAnnualBasis = TmpCalcAnnualBasis + 365
            TmpCalcAnnualBasis = TmpCalcAnnualBasis / (EndYear - StartYear + 1)
    return TmpCalcAnnualBasis

def YearFrac(StartDate, EndDate, Basis):
    Numerator = TmpDiffDates(StartDate, EndDate, Basis)
    Denom = TmpCalcAnnualBasis(StartDate, EndDate, Basis)
    YearFrac = Numerator/Denom
    return YearFrac

# Paramewter Table
st.title("Black-Scholes Option Pricer")
st.header("Parameters")
x1 = YearFrac(Initial_date, End_date, 1)
Input = pd.DataFrame({"Parameter":["Spot Price", "Exercise Price", "Risk-free rate (%)", "Initial Date", "End Date", "Time to Maturity (in days)", "Volatility (%)"],
"Value":[Spot_Price, Exercise_Price, Risk_free, Initial_date, End_date, x1, Volatility]})
parameter = pd.DataFrame(Input)
st.dataframe(parameter, width = 600, hide_index = True)

# Black-Scholes Option Pricing Model

import math
from scipy.stats import norm

def black_scholes_call(S, sigma, K, r, T):
  d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
  d2 = d1 - sigma * math.sqrt(T)
    
  call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    
  return call_price

def black_scholes_put(S, sigma, K, r, T):
  d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
  d2 = d1 - sigma * math.sqrt(T)
    
  put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

  return put_price


# Displaying result of Black-Scholes Option Pricing Model
call = black_scholes_call(Spot_Price, Volatility, Exercise_Price, Risk_free, x1)
put = black_scholes_put(Spot_Price, Volatility, Exercise_Price, Risk_free, x1)
ca = str(call)
pu = str(put)

from streamlit_extras.metric_cards import style_metric_cards
left, right = st.columns(2)
with left:
  st.metric("CALL Price", call)

with right:
  st.metric("PUT Price", put)

style_metric_cards(background_color="#000000", border_size_px = 5,border_color="#ffffff", border_left_color= "#00ffff")




st.divider()


# Heatmap Parameters Setting
sidebar1 = st.sidebar.header(" ")
sidebar2 = st.sidebar.header(" ")
sidebar4 = st.sidebar.divider()
sidebar3 = st.sidebar.header("Heatmap Parameters")
S_change = st.sidebar.number_input("Spot Price ∆ (in ±%)", value=10.00, format="%.2f", placeholder="Type within 0.00 - 100.00")
V_change = st.sidebar.number_input("Vol ∆ (in ±%)", value=10.00, format="%.2f", placeholder="Type within 0.00 - 100.00")

# Sensitivity Analysis
st.header("Option Price Interactive Heatmap")
st.markdown("Explore how option prices fluctuate with varying ''Spot Prices and Volatility'' levels using interactive heatmap parameters, all other parameters remain constant.")
n = 11
min_spot_price = Spot_Price * (1 - S_change / 100)
max_spot_price = Spot_Price * (1 + S_change / 100)
min_volatility = Volatility * (1 - V_change / 100)
max_volatility = Volatility * (1 + V_change / 100)

x2 = np.linspace(min_spot_price, max_spot_price, n)
y2 = np.linspace(min_volatility, max_volatility, n)
x3 = ['%.2f' % elem for elem in x2]
y3 = ['%.2f' % elem for elem in y2]

x4,y4 = np.meshgrid(x2,y2)

z1 = np.vectorize(black_scholes_call)

z2 = np.vectorize(black_scholes_put)

a = z1(x4, y4, K = Exercise_Price, r = Risk_free, T = x1)

b = z2(x4, y4, K = Exercise_Price, r = Risk_free, T = x1)

left2, right2 = st.columns([2,2])
with left2:
  fig1 = plt.figure(figsize = (20,10))
  cp = sns.heatmap(a, xticklabels=x3, yticklabels=y3, cbar=True, cbar_kws={'format':"%.2f", 'pad':0.03}, square=True, fmt='.3f', annot=True, annot_kws={'size':10}, cmap='plasma')
  cp.invert_yaxis()
  plt.xlabel('Spot Price')
  plt.ylabel('Volatility')
  st.pyplot(fig1)

with right2:
  fig2 = plt.figure(figsize = (20,10))
  pp = sns.heatmap(b, xticklabels=x3, yticklabels=y3, cbar=True, cbar_kws={'format':"%.2f", 'pad':0.03}, square=True, fmt='.3f', annot=True, annot_kws={'size':10}, cmap='viridis')
  pp.invert_yaxis()
  plt.xlabel('Spot Price')
  plt.ylabel('Volatility') 
  st.pyplot(fig2)
