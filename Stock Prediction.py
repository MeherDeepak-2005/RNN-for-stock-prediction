import matplotlib.pyplot as plt
import torch
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from LSTM import LSTM
from torch.utils.data import DataLoader
from torch import save
from torch import load
import pprint

pprint = pprint.PrettyPrinter(indent=4)

model = LSTM(1, 10, 50, 1)

# Normalize data
scaler = StandardScaler()

# List all stocks to get
stocks = [
	'MSFT',
	'AAPL',
	'GOOG',
	'TSLA'
]

stock_data = []

# get all data of stocks and append only the 'Close' prices to stock_data
for stock in stocks:
	stock_data.append(yf.Ticker(stock).history('max')['Close'])

# concatenate all stocks along
stock_dataframe = pd.concat(stock_data, axis=1)
stock_dataframe.columns = stocks
stock_dataframe.dropna(inplace=True)

plt.plot(stock_dataframe.values, label=stocks)
plt.legend(loc='upper left')
plt.show()

returns_data = []
series = []

# get each of the closing price from all the Stocks stored in stock_data
for index, stock_price in enumerate(stock_data):
	returns = np.log(stock_price).diff()
	assert returns.shape == stock_price.shape
	returns_data.append(returns)

return_dataframe = pd.concat(returns_data, axis=1)
return_dataframe.columns = stocks
return_dataframe.dropna(inplace=True)

plt.plot(return_dataframe.values, label=stocks)
plt.legend(loc='upper left')
plt.show()

series = scaler.fit_transform(stock_dataframe.values)
print(series.shape)
series = pd.DataFrame(series, columns=stocks)
print(series.tail())


def train(epochs: int, X, y):
	train_loss = np.zeros(epochs)

	for it in range(epochs):
		model.optimizer.zero_grad()

		outputs = model(X)
		loss = model.loss(outputs, y)

		loss.backward()
		model.optimizer.step()

		train_loss[it] = loss.item()

		if (it + 1) % 100 == 0:
			print('Epoch', (it + 1) / epochs, 'Train Loss: ', loss.item())

		if (it + 1) % 1000 == 0:
			print("... saving model ...")
			save(model.state_dict(), 'NewModel.pt')

	return train_loss


for stock in series:
	T = 20
	D = 1
	X = []
	Y = []
	new_series = series[stock].values
	for t in range(len(new_series) - T):
		x = new_series[t:t + T]
		y = new_series[t + T]
		X.append(x)
		Y.append(y)

	X = np.array(X).reshape(-1, T, 1)
	Y = np.array(Y).reshape(-1, 1)

	X = torch.from_numpy(X.astype(np.float32)).view(-1, T, 1)
	Y = torch.from_numpy(Y.astype(np.float32)).view(-1, 1)

	X.to(model.device)
	Y.to(model.device)

	train_loss = train(epochs=10000, X=X, y=Y)

	plt.plot(train_loss, label='train loss')
	plt.legend()
	plt.show()
