import torch
import torch.nn as nn
import torch.optim as optim


class LSTM(nn.Module):
	def __init__(self, n_inputs, n_hidden, n_rnn_layers, n_outputs):
		super(LSTM, self).__init__()
		self.D = n_inputs
		self.M = n_hidden
		self.K = n_outputs
		self.L = n_rnn_layers

		self.lstm = nn.LSTM(
			input_size=self.D,
			hidden_size=self.M,
			num_layers=self.L,
			batch_first=True
		)
		self.fc = nn.Linear(self.M, self.K)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

		self.loss = nn.MSELoss()
		self.optimizer = optim.Adam(self.parameters(), lr=0.1)

	def forward(self, X):
		h0 = torch.zeros(self.L, X.size(0), self.M).to(self.device)
		c0 = torch.zeros(self.L, X.size(0), self.M).to(self.device)

		out, _ = self.lstm(X, (h0, c0))

		out = self.fc(out[:, -1, :])
		return out

