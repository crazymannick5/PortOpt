import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import sys

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


    def __init__(self, startdate, enddate, name_input,  tickers="AAPL", interval="1d"):
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
        self.start_date = startdate
        self.end_date = enddate



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
    def UpdateAll(self):
        self.update_raw_data(self.start_date, self.end_date)
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
        R_min = np.min(self.u)
        R_max = np.max(self.u)
        target_returns = np.linspace(R_min, R_max, num_points)

        risks = []
        returns = []

        for R_star in target_returns:
            w = self.calcOptW(R_star)
            portfolio_risk = self.calculate_portfolio_risk(w)
            risks.append(portfolio_risk)
            returns.append(R_star)
            

        #returns = returns[:-1]
        frontier_df = pd.DataFrame({
            "Portfolio Risk (Standard Deviation)": risks,
            "Portfolio Return": returns
        })

        return frontier_df

    def plot_efficient_frontier(self, frontier_df):
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

    def fetch_risk_free_rate(self, start_date, end_date, symbol="^IRX"):
        """
        Fetches the relevant risk-free rate for a given time period from Yahoo Finance.

        Parameters:
            start_date (str): Start date of the period (in 'YYYY-MM-DD' format).
            end_date (str): End date of the period (in 'YYYY-MM-DD' format).
            symbol (str): Yahoo Finance symbol for Treasury yield (^IRX for 3-month T-bill, ^TNX for 10-year bond).

        Returns:
            float: The average risk-free rate over the given period as a decimal.
        """
        # Fetch risk-free rate data from Yahoo Finance
        risk_free_data = yf.download(symbol, start=start_date, end=end_date)
        if risk_free_data.empty:
            raise ValueError(f"No data fetched for symbol: {symbol} between {start_date} and {end_date}")

        # Use the 'Close' column for the risk-free rate and convert percentage to decimal
        risk_free_data = risk_free_data["Close"] / 100  # Convert from percentage to decimal

        # Calculate the average risk-free rate over the period
        average_risk_free_rate = risk_free_data.mean()

        return average_risk_free_rate
    
    def sharpe_ratio_optimization(self, rfr):
        self.UpdateAll()
        ones = np.ones(len(self.u))
        eReturns = self.u - np.full_like(self.u, rfr)#self.u - rfr
        inv_Sigma = np.linalg.inv(self.Sigma)

        w_opt = inv_Sigma @ eReturns / (ones.T @ inv_Sigma @ eReturns)
        expected_return = np.dot(w_opt, self.u)
        risk = np.sqrt(w_opt.T @ self.Sigma @ w_opt)
        sharpe_ratio = (expected_return - rfr) / risk

        return {
            "weights": w_opt,
            "expected_return": expected_return,
            "risk": risk,
            "sharpe_ratio": sharpe_ratio
        }
    def calculate_cml_params(self, sharpe_ratio, portfolio_risk, portfolio_return):
        """
        Calculates the slope and intercept of the Capital Market Line (CML) given portfolio metrics.

        Parameters:
            sharpe_ratio (float): The Sharpe ratio of the portfolio.
            portfolio_risk (float): The risk (standard deviation) of the portfolio.
            portfolio_return (float): The expected return of the portfolio.

        Returns:
            dict: A dictionary containing the slope (m) and intercept (b) of the CML.
        """
        # Intercept (risk-free rate)
        risk_free_rate = portfolio_return - sharpe_ratio * portfolio_risk

        # Slope is the Sharpe ratio
        slope = sharpe_ratio

        return slope, risk_free_rate

    def plot_efficient_frontier_with_cml(self, frontier_df, m, b, tangent_risk, tangent_return):
        """
        Plots the Efficient Frontier and the Capital Market Line (CML) and marks the tangent portfolio.

        Parameters:
            frontier_df (pandas.DataFrame): DataFrame with "Portfolio Risk (Standard Deviation)" and "Portfolio Return" columns.
            m (float): Slope of the CML.
            b (float): Intercept of the CML.
            tangent_risk (float): Risk (standard deviation) of the tangent portfolio.
            tangent_return (float): Expected return of the tangent portfolio.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))

        # Plot the Efficient Frontier
        plt.plot(frontier_df["Portfolio Risk (Standard Deviation)"], 
                frontier_df["Portfolio Return"], label="Efficient Frontier", marker="o")

        # Generate the CML line
        cml_risks = frontier_df["Portfolio Risk (Standard Deviation)"].values
        cml_returns = np.array([m * risk + b for risk in cml_risks])
        plt.plot(cml_risks, cml_returns, label="Capital Market Line (CML)", linestyle="--", color="red")

        # Mark the tangent portfolio
        plt.scatter(tangent_risk, tangent_return, color="blue", label="Tangency Portfolio", zorder=5)
        plt.annotate("Tangency Portfolio", 
                    (tangent_risk, tangent_return), 
                    textcoords="offset points", xytext=(10, -10), ha='center', color="blue")

        # Add labels and legend
        plt.xlabel("Portfolio Risk (Standard Deviation)")
        plt.ylabel("Portfolio Return")
        plt.title("Efficient Frontier and Capital Market Line")
        plt.legend()
        plt.grid()
        

        min_risk = 0
        max_risk = frontier_df["Portfolio Risk (Standard Deviation)"].max()
        plt.xlim(min_risk, max_risk * 1.1)  # Extend by 10%

        # Set Y-axis limits
        min_return = min(float(frontier_df["Portfolio Return"].min()), float(b))
        max_return = max(frontier_df["Portfolio Return"].max(), max(cml_returns))
        padding = (max_return - min_return) * 0.5  # Add 10% padding
        plt.ylim(min_return - padding, max_return + padding)
        plt.show()

"""
# Define the date range
start_date = "2023-01-01"
end_date = "2023-12-31"

my_portfolio = portfolio(start_date, end_date, name_input="Test Portfolio", tickers=["AAPL", "MSFT", "GOOGL","TSLA","NVDA","AMZN","WMT","JPM"], interval="1d")



# Fetch and prepare the data
my_portfolio.update_raw_data(start_date=start_date, end_date=end_date)
my_portfolio.update_data()
my_portfolio.update_Returns()
my_portfolio.update_eXReturns()
my_portfolio.update_Sigma()

# Fetch the risk-free rate
risk_free_rate = my_portfolio.fetch_risk_free_rate(start_date=start_date, end_date=end_date)

# Perform Sharpe Ratio optimization to find the tangent portfolio
tangent_portfolio = my_portfolio.sharpe_ratio_optimization(rfr=risk_free_rate)
tangent_risk = tangent_portfolio["risk"]
tangent_return = tangent_portfolio["expected_return"]
sharpe_ratio = tangent_portfolio["sharpe_ratio"]

# Calculate the Capital Market Line (CML) parameters
m, b = my_portfolio.calculate_cml_params(sharpe_ratio, tangent_risk, tangent_return)

# Generate the Efficient Frontier points
frontier_df = my_portfolio.efficient_frontier_points(num_points=50)

# Plot the Efficient Frontier, CML, and Tangency Portfolio
my_portfolio.plot_efficient_frontier_with_cml(frontier_df, m, b, tangent_risk, tangent_return)"""


def portfolio_cli():
    """
    Command-line interface for managing and interacting with portfolios.
    """
    portfolios = {}

    def display_menu():
        print("\nPortfolio Optimization CLI")
        print("1. Create a new portfolio")
        print("2. View existing portfolios")
        print("3. Switch to a portfolio")
        print("4. Exit")
        print("-" * 40)

    def portfolio_menu(portfolio):
        while True:
            print(f"\nManaging Portfolio: {portfolio.name}")
            print("1. Add tickers")
            print("2. Remove a ticker")
            print("3. View tickers")
            print("4. Change return calculation method (simple/log)")
            print("5. Change expected return method (historical/geometric/exponential)")
            print("6. Update data and perform analysis")
            print("7. Plot Efficient Frontier with CML")
            print("8. Print tangent portfolio weights")
            print("9. Print portfolio weights for a target return")
            print("10. Back to main menu")
            print("-" * 40)

            choice = input("Enter your choice: ").strip()

            if choice == "1":
                tickers = input("Enter tickers to add (comma-separated): ").strip().split(",")
                portfolio.add_tickers([ticker.strip().upper() for ticker in tickers])
            elif choice == "2":
                portfolio.show_tickers()
                try:
                    index = int(input("Enter the index of the ticker to remove: ").strip())
                    portfolio.remove_ticker_by_index(index)
                except ValueError:
                    print("Invalid index.")
            elif choice == "3":
                portfolio.show_tickers()
            elif choice == "4":
                method = input("Enter return method (simple/log): ").strip().lower()
                portfolio.set_retInd(method)
            elif choice == "5":
                method = input("Enter expected return method (historical/geometric/exponential): ").strip().lower()
                portfolio.set_eXretInd(method)
            elif choice == "6":
                try:
                    portfolio.UpdateAll()
                    print("Data updated and portfolio analysis performed.")
                except Exception as e:
                    print(f"Error during update: {e}")
            elif choice == "7":
                try:
                    # Perform analysis
                    portfolio.UpdateAll()
                    risk_free_rate = portfolio.fetch_risk_free_rate(portfolio.start_date, portfolio.end_date)
                    tangent_portfolio = portfolio.sharpe_ratio_optimization(rfr=risk_free_rate)
                    tangent_risk = tangent_portfolio["risk"]
                    tangent_return = tangent_portfolio["expected_return"]
                    sharpe_ratio = tangent_portfolio["sharpe_ratio"]
                    m, b = portfolio.calculate_cml_params(sharpe_ratio, tangent_risk, tangent_return)
                    frontier_df = portfolio.efficient_frontier_points(num_points=50)

                    # Plot
                    portfolio.plot_efficient_frontier_with_cml(frontier_df, m, b, tangent_risk, tangent_return)
                except Exception as e:
                    print(f"Error during plotting: {e}")
            elif choice == "8":
                try:
                    risk_free_rate = portfolio.fetch_risk_free_rate(portfolio.start_date, portfolio.end_date)
                    tangent_portfolio = portfolio.sharpe_ratio_optimization(rfr=risk_free_rate)
                    print("Tangent Portfolio Weights:")
                    for ticker, weight in zip(portfolio.tickers, tangent_portfolio["weights"]):
                        print(f"{ticker}: {weight:.4f}")
                except Exception as e:
                    print(f"Error fetching tangent portfolio weights: {e}")
            elif choice == "9":
                try:
                    target_return = float(input("Enter the target return: ").strip())
                    weights = portfolio.calcOptW(target_return)
                    print("Portfolio Weights for Target Return:")
                    for ticker, weight in zip(portfolio.tickers, weights):
                        print(f"{ticker}: {weight:.4f}")
                except ValueError:
                    print("Invalid target return.")
                except Exception as e:
                    print(f"Error fetching portfolio weights: {e}")
            elif choice == "10":
                break
            else:
                print("Invalid choice. Please try again.")

    while True:
        display_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            # Create a new portfolio
            name = input("Enter portfolio name: ").strip()
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD): ").strip()
            tickers = input("Enter tickers (comma-separated): ").strip().split(",")
            interval = input("Enter interval (e.g., 1d, 1wk, 1mo): ").strip()
            portfolio_instance = portfolio(start_date, end_date, name, [ticker.strip().upper() for ticker in tickers], interval)
            portfolios[name] = portfolio_instance
            print(f"Portfolio '{name}' created.")

        elif choice == "2":
            # View existing portfolios
            if portfolios:
                print("\nExisting Portfolios:")
                for idx, p in enumerate(portfolios.values(), 1):
                    print(f"{idx}. {p.name} (Tickers: {', '.join(p.tickers)})")
            else:
                print("\nNo portfolios available. Create one first.")

        elif choice == "3":
            # Switch to a portfolio
            if not portfolios:
                print("\nNo portfolios available. Create one first.")
                continue

            print("\nAvailable Portfolios:")
            portfolio_names = list(portfolios.keys())
            for idx, name in enumerate(portfolio_names, 1):
                print(f"{idx}. {name}")

            try:
                portfolio_idx = int(input("Enter the number of the portfolio to manage: ").strip()) - 1
                if 0 <= portfolio_idx < len(portfolio_names):
                    selected_portfolio = portfolios[portfolio_names[portfolio_idx]]
                    portfolio_menu(selected_portfolio)
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == "4":
            print("Exiting Portfolio Optimization CLI. Goodbye!")
            sys.exit(0)

        else:
            print("Invalid choice. Please try again.")
portfolio_cli()