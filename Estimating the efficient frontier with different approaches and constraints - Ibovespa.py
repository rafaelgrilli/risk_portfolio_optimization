#!/usr/bin/env python
# coding: utf-8

# In[3]:


""""
I have a lot of interest for portfolio management and research in general. 
These last days I was wondering what could be the best portfolio of stocks, considering as the
whole potential market the stocks that compound Brazilian Ibovespa. Yesterday I conducted this exercise, 
downloaded and cleaned the data from YahooFinance, and needed to take out from the scope of possible 
assets 5 stocks which IPO was recent and data available limited. If I had used the whole cohort
of companies in Ibovespa, I would have got a result that is not a nonconvex problem, and in this 
case portfolio theory may not be the best fit. I use PyPortofolioOpt, a library that implements portfolio 
optimisation methods. I will present the portfolios that maximize the Sharpe Ratio, with 
different optimisation approaches on the same data
"""


# In[4]:


# I start by estimating the long-only portfolio that maximises the SR (the risk-free default is 2%)


# In[5]:


import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


# In[6]:


df = pd.read_csv("ibov_stocks2.csv", parse_dates=True, index_col="Date")


# In[7]:


mu = expected_returns.mean_historical_return(df)
mu


# In[8]:


mu.plot.barh(figsize=(10,10));


# In[9]:


S = risk_models.sample_cov(df)
S


# In[10]:


from pypfopt import plotting
plotting.plot_covariance(S, plot_correlation=True)


# In[11]:


ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
weights = ef.clean_weights()
weights


# In[12]:


num_small = len([k for k in weights if weights[k] <= 1e-4])
print(f"{num_small}/{len(ef.tickers)} ticker have zero weight")


# In[13]:


ef.portfolio_performance(verbose=True)


# In[14]:


from pypfopt import CLA, plotting
cla = CLA(mu, S)
cla.max_sharpe()
cla.portfolio_performance(verbose=True)
ax = plotting.plot_efficient_frontier(cla, showfig=False)


# In[15]:


"""
We could have a portfolio with annual vol of 11.4% and expected annual return of 22.3%. 
It is possible to obtain a positive SR by defining portfolios with stocks that are part of the ibovespa index.
This output, despite interesting, is not useful in itself. Let's then convert it into an allocation that 
an investor could use to weight her own portfolio
"""


# In[18]:


from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000)
allocation, leftover = da.lp_portfolio()
print("Discrete allocation", allocation)
print("Funds remaining: ${:.2f}".format(leftover))


# In[20]:


import pandas as pd
pd.Series(weights).plot.pie(figsize=(10,10))


# In[21]:


"""
As we may check from the output, only part of the stocks compounding the Ibovespa index were selected to
be part of our SP maximised portfolio. Let's now conduct some exercises, obtaining other portfolios
for different risks. Let's remember that, according to the model for a long-only portfolio, the investor
would maximise the SP by taking 11.4% of expected risk, or volatility. What if the investor
is a risk-lover, and is willing to invest in a riskier portfolio to increase the odds of higher returns?
"""


# In[22]:


ef = EfficientFrontier(mu, S)
ef.efficient_risk(target_volatility=0.15)
weights = ef.clean_weights()
weights


# In[23]:


num_small = len([k for k in weights if weights[k] <= 1e-4])
print(f"{num_small}/{len(ef.tickers)} ticker have zero weight")


# In[24]:


ef.portfolio_performance(verbose=True)


# In[25]:


"""
Comparing this portfolio with 15% of risk, higher than the natural portfolio maximising the SP for the lowest 
level of risk, 11.4%, we may expect a higher return (27.4% vs 22.3%, respectively)
"""


# In[26]:


pd.Series(weights).plot.pie(figsize=(10,10))


# In[27]:


"""
One interesting feature of this new portfolio is that, compared with the first, is being less benefited by 
diversification. Still, from 12 stocks were selected by the algorithm to be part of the portfolio.
We should also remember that this is a long-only portoflio. Let's simulate some other portfolios and check 
new possible outputs.
"""


# In[28]:


ef = EfficientFrontier(mu, S)
ef.efficient_risk(target_volatility=0.2)
weights = ef.clean_weights()
weights


# In[29]:


pd.Series(weights).plot.pie(figsize=(10,10))


# In[30]:


ef.portfolio_performance(verbose=True)


# In[31]:


"""
It is clearer that as we increase the portfolio's risk, odding to increase expected return, we also 
lose the diversification benefits and get more concentration as a result. We may use regression and penalization
to avoid that the optimiser overfits the data. We are likely to obtain better results by enforcing 
more diversification, if possible. L2 regularisation does that, let's simulate: 
"""


# In[32]:


from pypfopt import objective_functions
ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.L2_reg, gamma=1)
ef.efficient_risk(0.2)
weights = ef.clean_weights()
ef.portfolio_performance(verbose=True)
weights


# In[33]:


# let's now simulate the global-minimum variance (GMV) without providing the returns.
# we also apply the Leidoit Wolf method for covariance shrinkage. let's compare if the results will change:


# In[34]:


S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
ef = EfficientFrontier(None, S, weight_bounds=(None, None))
ef.min_volatility()
weights = ef.clean_weights()
weights


# In[35]:


pd.Series(weights).plot.barh(figsize=(10,10))


# In[36]:


ef.portfolio_performance(verbose=True)


# In[37]:


"""
The GMV portfolio provides us with a portfolio with 8.2% of risk. 
Let's simulate the allocation, considering an investor happy with this risk level and with
100000 Brazilian reais to invest in this portfolio.
"""


# In[38]:


latest_prices = get_latest_prices(df)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000, short_ratio=0.1)
alloc, leftover = da.lp_portfolio()
print(f"Discrete allocation performed with R${leftover:.2f} leftover")
alloc


# In[41]:


pd.Series(weights).plot.barh(figsize=(10,10))


# In[42]:


ef.portfolio_performance(verbose=True)


# In[43]:


"""
Applying the GMV portfolio to a total of R$100,000.00, and a short ratio of 10% (R$10,000.00),
we may replicate the GMV portfolio. Comparing with the portfolios obtained with simple returns and covariances,
more assets are included, which may suggest that diversification benefits were achievable. Important to note 
again that we might only have as the output the GMV, we do not have expected returns. Let's proceed 
with the allocation
"""


# In[44]:


"""
Let's now do an exercise to find the Max SR adding sector constrints to our portfolio. 
"""


# In[47]:


sector_mapper = {
    "ABEV3": "Consumer Staples",
    "B3SA3": "Financials",
    "BBDC3": "Financials",
    "BBDC4": "Financials", 
    "BBSE3": "Financials", 
    "BEEF3": "Consumer Staples", 
    "BRML3": "Real Estate", 
    "BTOW3": "Consumer Discretionary",
    "CCRO3": "Industrials", 
    "CIEL3": "Tech",
    "CMIG4": "Utilities",
    "COGN3": "Consumer Discretionary",
    "CSAN3": "Oil & Gas",
    "CSNA3": "Materials",
    "CVCB3": "Consumer Discretionary",
    "CYRE3": "Real Estate",
    "ECOR3": "Industrials",
    "EGIE3": "Utilities",
    "ELET3": "Utilities",
    "ELET6": "Utilities", 
    "EMBR3": "Industrials",
    "ENBR3": "Utilities", 
    "ENGI11": "Utilities",
    "EQTL3": "Utilities", 
    "EZTC3": "Real Estate",
    "FLRY3": "Health Care",
    "GGBR4": "Materials",
    "GOAU4": "Materials", 
    "GOLL4": "Industrials",
    "HGTX3": "Consumer Discretionary",
    "HYPE3": "Pharma",
    "IGTA3": "Real Estate",
    "ITSA4": "Financials", 
    "ITUB4": "Financials", 
    "JBSS3": "Consumer Staples", 
    "KLBN11": "Pulp & Paper", 
    "LAME4": "Consumer Discretionary", 
    "LREN3": "Consumer Discretionary",
    "MRFG3": "Consumer Staples", 
    "MRVE3": "Consumer Discretionary",
    "MULT3": "Real Estate",
    "PETR3": "Oil & Gas", 
    "PETR4": "Oil & Gas", 
    "PRIO3": "Oil & Gas", 
    "QUAL3": "Industrials",
    "RADL3": "Healthcare", 
    "RENT3": "Consumer Discretionary", 
    "SANB11": "Financials", 
    "SBSP3": "Utilities", 
    "SULA11": "Financials", 
    "TAEE11": "Energy", 
    "TOTS3": "Tech", 
    "UGPA3": "Energy", 
    "USIM5": "Materials", 
    "VALE3": "Materials", 
    "VIVT4": "Communications", 
    "WEGE3": "Industrials", 
    "YDUQ3": "Consumer Discretionary",
    "TIMS3": "Communications", 
    "SUZB3": "Pulp & Paper"
}

sector_lower = {
    "Energy": 0.02,
    "Pulp & Paper": 0.02,
}

sector_upper = { 
    "Consumer Discretionary": 0.2,
    "Financials": 0.3,
}


# In[55]:


# we could use the above sector dictionary to bound the weights of sectors in the portfolio, which will not
# be done here. Let's again try to target our volatility at 15%


# In[51]:


ef = EfficientFrontier(mu, S)
ef.efficient_risk(target_volatility=0.15)
weights = ef.clean_weights()
weights


# In[52]:


num_small = len([k for k in weights if weights[k] <= 1.e-4])
print(f"{num_small}/{len(ef.tickers)} tickers have zero weight")


# In[53]:


ef.portfolio_performance(verbose=True)


# In[56]:


"""
Once again, many of the possible assets to that potentially could be part of our portfolio were ignored, and an investor 
may be worried about few diversification. Let's enforce some level of diversification with L2 regularisation. 
"""


# In[57]:


ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.L2_reg, gamma=1)
ef.efficient_risk(0.15)
weights = ef.clean_weights()
weights


# In[58]:


num_small = len([k for k in weights if weights[k] <= 1.e-4])
print(f"{num_small}/{len(ef.tickers)} tickers have zero weight")


# In[59]:


# as expected, more assets were attributed weights != 0, which brings us more diversification. 


# In[60]:


pd.Series(weights).plot.pie(figsize=(10, 10))


# In[61]:


ef.portfolio_performance(verbose=True)


# In[63]:


"""

Let's now work with the assumption that we have a required rate of return, such as an investor to retire, 
or the actuarial level necessary to cover the liabilities of a pension fund or scheme. We also suppose 
our portfolio to be market neutral, equally exposed to long and short positions. 

"""


# In[71]:


ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
ef.add_objective(objective_functions.L2_reg)
ef.efficient_return(target_return=0.15, market_neutral=True)
weights = ef.clean_weights()
weights


# In[72]:


ef.portfolio_performance(verbose=True)


# In[73]:


pd.Series(weights).plot.barh(figsize=(10,10))


# In[74]:


print(f"Net weight: {sum(weights.values()):.2f}")


# In[77]:


# now, simulate yourself and let me know your best theoretical portfolio according to your personal risk tolerance.


# In[ ]:




