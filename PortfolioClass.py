import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class portfolio:
    INTERVAL_TO_DAYS = {"1m": 1/1440, "1d": 1, "1wk": 7, "1mo": 30, "3mo": 90}
    """
    name = "default portfolio"
    tickers: list[str] = []
    interval: str = "1d"
    interval_days: int = 1
    raw_data : pd.DataFrame = pd.DataFrame()
    data: pd.DataFrame = pd.DataFrame()
    expected_returns_method: str = "historical"
    R"""


    def __init__(self, name_input, tickers="AAPL", interval="1d"):
        self.name = name_input
        self.tickers = tickers
        self.set_interval(interval)
        self.raw_data = None
        self.data  = None
        self.expected_returns_method = "historical"
        self.returns = None
        self.u = None
        self.Sigma = None
        self.dataMatrix = None
        self.returns = None
        self.retInd = "simple"
        self.eXretInd = "historical"
        self.halfLife = 10
        self.INTERVAL_TO_DAYS = {"1m": 1/1440, "1d": 1, "1wk": 7, "1mo": 30, "3mo": 90}
        self.targetRmax = 0



    def changeName(self, newName): 
        self.name = newName
    
    def printName(self):
        print(self.name)
    
    def set_tickers(self, new_tickers):
        """
        Set the tickers list equal to a passed list.
        :param new_tickers: list[str] - The new list of tickers.
        """
        if not all(isinstance(ticker, str) for ticker in new_tickers):
            raise ValueError("All tickers must be strings.")
        self.tickers = new_tickers

    def add_tickers(self, tickers_to_add):
        """
        Add a passed list of string tickers to the current list.
        :param tickers_to_add: list[str] - List of tickers to add.
        """
        if not all(isinstance(ticker, str) for ticker in tickers_to_add):
            raise ValueError("All tickers must be strings.")
        self.tickers.extend(tickers_to_add)

    def show_tickers(self):
        """
        Show the tickers listed next to their index in the list.
        """
        for index, ticker in enumerate(self.tickers):
            print(f"{index}: {ticker}")

    def remove_ticker_by_index(self, index):
        """
        Remove one of the tickers from the list based on the passed index.
        :param index: int - Index of the ticker to remove.
        """
        if not (0 <= index < len(self.tickers)):
            raise IndexError("Invalid index. Please provide a valid index.")
        removed_ticker = self.tickers.pop(index)
        print(f"Removed ticker: {removed_ticker}")

    def contains_ticker(self, ticker):
        """
        Check if a passed string is in the tickers list.
        :param ticker: str - The ticker to check.
        :return: bool - True if the ticker is in the list, False otherwise.
        """
        return ticker in self.tickers

    def add_ticker(self, ticker):
        """
        Add a string as one element to the tickers list.
        :param ticker: str - The ticker to add.
        """
        if not isinstance(ticker, str):
            raise ValueError("Ticker must be a string.")
        self.tickers.append(ticker)

    def set_interval(self, interval):
        """
        Set the interval for the portfolio.
        :param interval: str - Interval string.
        """
        if interval not in self.INTERVAL_TO_DAYS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {list(self.INTERVAL_TO_DAYS.keys())}.")
        self.interval = interval
        self.interval_days = self.INTERVAL_TO_DAYS[interval]  # Set corresponding number of days

    def get_interval_info(self):
        """
        Get the current interval and its corresponding number of days.
        :return: tuple - (interval, number of days).
        """
        return self.interval, self.interval_days
    
    def update_raw_data(self, start_date, end_date):
        """
        Update the instance-specific raw_data variable with data from yfinance.
        :param start_date: str - Start date for the data (e.g., '2022-01-01').
        :param end_date: str - End date for the data (e.g., '2023-01-01').
        """
        if not self.tickers:
            raise ValueError("Tickers list is empty. Add tickers before updating raw data.")

        # Fetch data using yfinance
        data = yf.download(
            tickers=self.tickers,
            start=start_date,
            end=end_date,
            interval=self.interval,
            group_by="ticker"
        )

        # Validate and update raw_data
        if data.empty:
            raise ValueError("No data fetched. Check your parameters.")
        self.raw_data = data
        print(f"Raw data updated for portfolio: {self.name}")
    
    def set_expected_returns_method(self, method):
        """
        Set the method used for calculating expected returns.
        :param method: str - One of 'geometric', 'exponential', 'historical'.
        """
        valid_methods = ["geometric", "exponential", "historical"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}.")
        self.expected_returns_method = method
        print(f"Expected returns method set to: {method}")

    def update_dataMatrix(self):
        close_prices = self.data.loc[:, (slice(None), "Close")]
        close_prices.columns = close_prices.columns.get_level_values(0)
        closing_prices_matrix = close_prices.to_numpy()
        self.dataMatrix = closing_prices_matrix

    def update_data(self):
        """
        Cleans the raw_data by:
        - Replacing missing data points with averages from both sides.
        - Handling consecutive missing data with linear interpolation.
        - Filling missing values at the edges with forward/backward fill.
        Updates the cleaned data to self.data.
        """
        if self.raw_data is None or self.raw_data.empty:
            raise ValueError("Raw data is empty. Fetch raw data before cleaning.")

        # Copy raw_data to avoid modifying it directly
        cleaned_data = self.raw_data.copy()

        # Linear interpolation for missing values (handles consecutive missing values)
        cleaned_data.interpolate(method="linear", axis=0, inplace=True)

        # Forward-fill and backward-fill for edge cases (first/last missing values)
        cleaned_data.fillna(method="bfill", axis=0, inplace=True)  # Fill from next valid value
        cleaned_data.fillna(method="ffill", axis=0, inplace=True)  # Fill from previous valid value

        # Update the data attribute
        self.data = cleaned_data
        print(f"Data cleaned and updated for portfolio: {self.name}")
        self.update_dataMatrix()
    
    def update_simpReturns(self):
        self.returns = (self.dataMatrix[1:,:] - self.dataMatrix[:-1,:]) / self.dataMatrix[:-1,:]        
    def update_logReturns(self):
        self.returns = np.log(self.dataMatrix[1:, :] / self.dataMatrix[:-1, :])
    def update_max_feasible_target(self):
        self.targetRmax = np.max(self.u)
    def update_Returns(self):
        if (self.retInd == "simple"):
            self.update_simpReturns()
        else:
            self.update_logReturns()
    
    def eXReturnsHM(self):
        self.u = self.returns.mean(axis = 0)
    def exReturnsGO(self):
        self.u = np.prod(1 + self.returns, axis=0)**(1 / self.returns.shape[0]) - 1
    def exReturnsEX(self):
        lambda_ = 2**(-1/self.halfLife)
        weights = np.array([lambda_**(self.returns.shape[0] - 1 - t) for t in range(self.returns.shape[0])])
        self.u = (self.returns.T @ weights) / weights.sum()
    def update_eXReturns(self):
        if self.eXretInd == "historical":
            self.eXReturnsHM()
        elif self.eXretInd == "geometric":
            self.exReturnsGO()
        elif self.eXretInd == "exponential":
            self.exReturnsEX()
        self.update_max_feasible_target()
    
    def set_retInd(self, method):
        """
        Sets the preferred method for calculating returns.
        :param method: str - "simple" or "log".
        """
        valid_methods = ["simple", "log"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method for retInd: {method}. Valid options are: {valid_methods}.")
        self.retInd = method
        print(f"Return calculation method updated to: {self.retInd}")

    def set_eXretInd(self, method):
        """
        Sets the preferred method for calculating expected returns.
        :param method: str - "historical", "geometric", or "exponential".
        """
        valid_methods = ["historical", "geometric", "exponential"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method for eXretInd: {method}. Valid options are: {valid_methods}.")
        self.eXretInd = method
        print(f"Expected return calculation method updated to: {self.eXretInd}")  
    
    def update_Sigma(self):
        centered_returns = self.returns - np.mean(self.returns, axis=0)
        self.Sigma = np.dot(centered_returns.T, centered_returns) / (self.returns.shape[0] - 1)
    def udpateAll(self):
        self.update_raw_data()
        self.update_data()
        self.update_dataMatrix()
        self.update_Returns()
        self.update_eXReturns()
        self.update_Sigma()
    def calcOptW(self, targetR):
        self.UpdateAll()
        ones = np.ones(len(self.u))
        inv_Sigma = np.linalg.inv(self.Sigma)
        A = self.u.T @ inv_Sigma @ self.u
        B = self.u.T @ inv_Sigma @ ones
        C = ones.T @ inv_Sigma @ ones

        Delta = A * C - B**2

        w = inv_Sigma @ (
        ((C * targetR - B) / Delta) * self.u +
        ((A - B * targetR) / Delta) * ones
        )

        return(w)
    
    def calculate_expected_return(self, w):
        self.UpdateAll()
        return np.dot(w, self.u)
    def calculate_portfolio_risk(self, w):
        return np.sqrt(w.T @ self.Sigma @ w)

    def efficient_frontier_points(self, num_points=50):
        """
        Calculates the Efficient Frontier points for expected return and risk.

        Parameters:
            num_points (int): Number of target returns to calculate along the frontier.

        Returns:
            pandas.DataFrame: A DataFrame containing portfolio risks and returns with labeled columns.
        """
        R_min = np.min(self.mu)
        R_max = np.max(self.mu)
        target_returns = np.linspace(R_min, R_max, num_points)

        risks = []
        returns = []

        for R_star in target_returns:
            w = self.calcOptW(R_star)
            portfolio_risk = self.calculate_portfolio_risk(w)
            risks.append(portfolio_risk)
            returns.append(R_star)

        frontier_df = pd.DataFrame({
            "Portfolio Risk (Standard Deviation)": risks,
            "Portfolio Return": returns
        })

        return frontier_df

    def plot_efficient_frontier(frontier_df):
        """
        Plots the Efficient Frontier from a DataFrame containing risk and return.

        Parameters:
            frontier_df (pandas.DataFrame): DataFrame with "Portfolio Risk (Standard Deviation)" and "Portfolio Return" columns.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(frontier_df["Portfolio Risk (Standard Deviation)"], 
                frontier_df["Portfolio Return"], label="Efficient Frontier", marker="o")
        plt.xlabel("Portfolio Risk (Standard Deviation)")
        plt.ylabel("Portfolio Return")
        plt.title("Efficient Frontier")
        plt.legend()
        plt.grid()
        plt.show()