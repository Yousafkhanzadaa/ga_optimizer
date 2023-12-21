# Each trading strategy should provide a uniform interface so that the optimizer can interact with it in a standardized way. This should include:
# A backtesting method that returns metrics (e.g., net profit).
# Metadata for parameters that can be optimized.

class TradingStrategy:
    def __init__(self, **params):
        self.params = params

    def backtest(self, data):
        raise NotImplementedError("Backtesting method must be implemented by strategies")

    @property
    def parameter_ranges(self):
        raise NotImplementedError("Parameter ranges must be defined")
    


class MyStrategy(TradingStrategy):
    
    def __init__(self, lookback=20, z_entry_threshold=1.0, z_exit_threshold=0.0):
        self.lookback = lookback
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold
        
        
    def backtest(self, data):
        
        # Calculate rolling mean and standard deviation
        rolling_mean = data['Close'].rolling(window=self.lookback).mean()
        rolling_std = data['Close'].rolling(window=self.lookback).std()

        # Calculate the Z-scores
        z_scores = (data['Close'] - rolling_mean) / rolling_std

        # Generate buy/sell signals based on Z-scores and thresholds
        data['position'] = 0
        data.loc[z_scores < -self.z_entry_threshold, 'position'] = 1  # Buy signal
        data.loc[z_scores > self.z_exit_threshold, 'position'] = -1   # Sell signal
        data['position'] = data['position'].shift(1)  # To avoid look-ahead bias

        # Assuming we start with a cash balance of 100,000 and invest fully on each trade
        starting_cash = 100000
        cash = starting_cash
        position = 0

        for i in range(1, len(data)):
            if data['position'][i] == 1 and cash > 0:  # Buy signal
                position = cash / data['Close'][i]
                cash -= position * data['Close'][i]
            elif data['position'][i] == -1 and position > 0:  # Sell signal
                cash += position * data['Close'][i]
                position = 0

        # Calculate final portfolio value after the backtesting period
        final_value = cash + (position * data['Close'].iloc[-1])
        net_profit = final_value - starting_cash

        return net_profit

    
    @property
    def parameter_ranges(self):
        return {
            "lookback": (10, 50),  # Lookback period can range from 10 to 50
            "z_entry_threshold": (0.5, 2.0),  # Z-score entry threshold
            "z_exit_threshold": (-0.5, 2.0),  # Z-score exit threshold
        }
