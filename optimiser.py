import re
import yfinance as yf
from datetime import date, timedelta
import argparse
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Tuple
import sys


def main():
    # Stock input, validation and data retrieval.
    RetrieveData.year()
    Input.collect_portfolio()
    Input.risk_free_rate()
    Input.display()
    port_name = Input.portfolio_name()
    stock_price = RetrieveData.adj_price()
    valid_stocks, invalid_stocks = RetrieveData.stock_verification(stock_price)
    valid_port = RetrieveData.portfolio_verification(valid_stocks)

    # Basic Statistics
    daily_return = Analysis.daily_return(stock_price)
    cum_return = Analysis.cumulative_return(daily_return)
    annual_return, benchmark_return = Analysis.annualised_return(
        daily_return, valid_port
    )
    stock_vol, benchmark_vol = Analysis.stock_volatility(daily_return, valid_port)
    stock_sharpe = Analysis.stock_sharpe(valid_port)
    benchmark_sharpe = Analysis.portfolio_sharpe(benchmark_return, benchmark_vol)
    Analysis.key_assignment_weight(valid_port)
    port_vol, cov_matrix = Analysis.portfolio_volatility(daily_return, valid_port)
    var_benchmark, sim_returns_benchmark = VaR.calculate(
        benchmark_return, benchmark_vol
    )

    # Basic Statistical output
    print(f"\nValid stocks: {valid_stocks}")
    print(f"Invalid stocks: {invalid_stocks}\n")
    print(f"Annualised Returns\n{annual_return}\n")
    print(f"Volatility\n{stock_vol}\n")
    print(f"Sharpe Ratio\n{stock_sharpe}\n")
    print(f"Covariance Matrix:\n{cov_matrix}\n")

    # Benchmark Portfolio
    print(f"[BENCHMARK] Metrics")
    print(f"Return: {benchmark_return}")
    print(f"Volatility: {benchmark_vol}")
    print(f"Sharpe Ratio: {benchmark_sharpe}")
    print(
        f"Value at Risk at {VaR.CONFIDENCE}% confidence level over {VaR.MONTH} months: {var_benchmark:.4f}\n"
    )

    # Portfolio Analysis calculation
    port_weights = Analysis.display_portfolio_weight(valid_port)
    _, expected_return_np = Input.expected_return(valid_port)
    Analysis.key_assignment_stock_sharpe(valid_port)
    port_return = Analysis.portfolio_historical_return(valid_port)
    port_exp_return = Analysis.portfolio_expected_return(valid_port)
    port_sharpe = Analysis.portfolio_sharpe(port_exp_return, port_vol)
    diverse_port = Analysis.diversification_benefit(valid_port, port_vol)
    var_port, sim_returns_port = VaR.calculate(port_return, port_vol)

    # Portfolio Analysis output
    print(f"\n[{port_name.upper()}] Metrics")
    print(f"{port_weights}")
    print(f"\nReturn: {port_return}")
    print(f"Volatility: {port_vol}")
    print(f"Sharpe ratio: {port_sharpe}")
    print(f"Diversification Benefit: {diverse_port}")
    print(
        f"Value at Risk at {VaR.CONFIDENCE}% confidence level over {VaR.MONTH} months: {var_port:.4f}"
    )

    # Optimal portfolio calculation
    min_weight, max_weight = Optimisation.min_max_weight(valid_port)
    opt_weights = Optimisation.maximise_sharpe(
        valid_port, Input.rf_rate, cov_matrix, min_weight, max_weight
    )
    opt_port = Optimisation.optimal_portfolio_list(valid_port, opt_weights)
    display_opt_weight = Analysis.display_portfolio_weight(opt_port)
    opt_return, opt_vol, opt_sharpe = Optimisation.portfolio_metrics(
        opt_port, cov_matrix
    )
    diverse_opt = Analysis.diversification_benefit(opt_port, opt_vol)
    var_opt, sim_returns_opt = VaR.calculate(opt_return, opt_vol)

    # Optimisation outputs
    print(f"\n[OPTIMISED] Metrics")
    print((display_opt_weight).round(5))
    print(
        f"Minimum weight on each stock: {min_weight}\nMaximum weight on each stock: {max_weight}\n"
    )
    print(f"Return:{opt_return}")
    print(f"Volatility:{opt_vol}")
    print(f"Sharpe Ratio: {opt_sharpe}")
    print(f"Diversification Benefit: {diverse_opt}")
    print(
        f"Value at Risk at {VaR.CONFIDENCE}% confidence level over {VaR.MONTH} months: {var_opt:.4f}\n"
    )

    user_input = input("Would you like to output the graphs? (yes/otherwise): ")
    if user_input.lower() == "yes" or user_input.lower() == "y":
        FileOutput.csv(f"{port_name}_price.csv", stock_price)
        FileOutput.csv(f"{port_name}_return.csv", daily_return)
        FileOutput.pdf(
            f"{port_name}_cumulative_returns.pdf",
            lambda: Plot.cumulative_returns(cum_return),
        )
        FileOutput.pdf(
            f"{port_name}_var.pdf", lambda: Plot.simulation(var_port, sim_returns_port)
        )
        FileOutput.pdf(
            f"{port_name}_optimisation_var.pdf",
            lambda: Plot.simulation(var_opt, sim_returns_opt),
        )
        FileOutput.pdf(
            f"{port_name}_benchmark_var.pdf",
            lambda: Plot.simulation(var_benchmark, sim_returns_benchmark),
        )
        FileOutput.pdf(
            f"{port_name}_sharpe.pdf",
            lambda: Plot.sharpe_ratio(
                valid_port,
                port_name,
                port_return,
                port_vol,
                opt_return,
                opt_vol,
                benchmark_return,
                benchmark_vol,
            ),
        )
    else:
        sys.exit()


# The values of "benchmark", "rf_rate_ticker", "rf_rate", "region", and "regex" can be adjusted based on user requirements.
class Input: 
    benchmark = "VTI" # The benchmark portfolio (market portfolio) to compare with the performance of your own portfolio.
    rf_rate_ticker = "^TNX" # Ticker symbol for the current risk-free rate.
    rf_rate = None # Specify the exact value of the risk-free rate of your choice.
    region = "US" # The stock's region.
    regex = r"^([A-Z]{1,5}|[A-Z]{1,3}\.[A-Z])$" # Regular expression to validate user's input.

    @classmethod
    def portfolio_name(cls):
        name = (
            input(
                "Provide a name for this portfolio (or press Enter for the default name): "
            ).lower()
            or "portfolio"
        )
        return name

    @classmethod
    def collect_portfolio(cls):
        """
        Collects and validates US stock inputs, appends the stock information into the `portfolio` class variable.
        """
        cls.portfolio = []
        COUNT = 1 
        instruction = f"Enter the {cls.region} stock ticker and the corresponding amount, separated by a space."
        done = "** Trigger E0FError when complete (Ctrl+D) **"
        print(instruction)
        print(done)
        while True:
            try:
                user_input = input(f"Stock {COUNT}: ").upper().split()
                if not len(user_input) == 2:
                    print(instruction)
                    continue
                ticker, amount = user_input
                try:
                    amount = float(amount)
                except ValueError:
                    print("Invalid amount")
                    continue
                if not re.search(cls.regex, ticker):
                    print("Invalid ticker format.")
                    continue
                if any(stock["ticker"] == ticker for stock in cls.portfolio):
                    for stock in cls.portfolio:
                        if stock["ticker"] == ticker:
                            stock["amount"] += amount
                else:
                    stock = {"ticker": ticker, "amount": amount}
                    cls.portfolio.append(stock)
                COUNT += 1
            except EOFError:
                break

        if len(cls.portfolio) < 2:
            raise ValueError("Portfolio must have at least 2 stocks. Please try again.")

    @classmethod
    def risk_free_rate(cls):
        if cls.rf_rate_ticker is not None:
            risk_free_rate = yf.download(cls.rf_rate_ticker, end=date.today())
            cls.rf_rate = float(risk_free_rate["Close"].iloc[-1] / 100)
            return cls.rf_rate
        else:
            return cls.rf_rate

    @classmethod
    def display(cls):
        print(
            f"\nBenchmark Portfolio: {cls.benchmark}\nRisk free rate: {cls.rf_rate}\nStocks Entered:"
        )
        display_string = ""
        SET = 1
        for stock in cls.portfolio:
            if stock["amount"] > 0:
                display_string += f"Set {SET}: {stock}\n"
                SET += 1
        print(display_string)

    @classmethod
    def tickers(cls) -> list:
        # list comprehension
        tickers = [stock["ticker"] for stock in cls.portfolio if stock["amount"] > 0]
        tickers.append(cls.benchmark)
        return tickers

    @classmethod
    def expected_return(cls, portfolio: list) -> Tuple[list, pd.Series]:
        """
        Prompts the user to specify the expected return of each stock, the default value is the historical return.
        Then appends the expected return into the `portfolio` class variable.
        """
        expected_return_list = []
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        for stock in portfolio:
            while True:
                user_input = input(
                    f"Please enter the expected annual return for [{stock['ticker']}] in decimals "
                    f"(or press Enter to use the historical return): "
                )
                if not user_input:
                    stock["expected_return"] = stock["hist_return"]
                    expected_return_list.append((stock["ticker"], stock["hist_return"]))
                    break
                try:
                    expected_return = float(user_input)
                    stock["expected_return"] = expected_return
                    expected_return_list.append((stock["ticker"], expected_return))
                    break
                except ValueError:
                    print("Invalid decimal format")
        expected_return_pd = pd.Series(dict(expected_return_list)).sort_index()
        return portfolio, expected_return_pd


class RetrieveData:
    DEFAULT_YEARS = 5

    @classmethod
    def year(cls):
        parser = argparse.ArgumentParser(
            description="This program statistically analysis stock performance and optimises portfolio returns,"
            "exclusively allowing for long positions."
        )
        parser.add_argument(
            "--years",
            type=int,
            default=cls.DEFAULT_YEARS,
            help="Specify the number of years to trace back",
        )
        return parser.parse_args()

    @classmethod
    def adj_price(cls) -> pd.DataFrame: # adjusted close price
        args = cls.year()
        years_back = date.today() - timedelta(days=args.years * 365)
        stock_prices = yf.download(Input.tickers(), start=years_back)["Adj Close"] 
        stock_prices.dropna(axis=1, how="all", inplace=True)
        return stock_prices

    @classmethod
    def stock_verification(cls, data: pd.DataFrame) -> Tuple[list, list]:
        ticker_input = data.columns
        valid_stocks = [stock for stock in ticker_input if stock != Input.benchmark]
        invalid_stocks = [
            stock for stock in Input.tickers() if stock not in ticker_input
        ]
        if not invalid_stocks:
            return valid_stocks, None
        else:
            return valid_stocks, invalid_stocks

    @classmethod
    def portfolio_verification(cls, valid_stocks):
        valid_portfolio = [
            stock for stock in Input.portfolio if stock["ticker"] in valid_stocks
        ]
        return valid_portfolio


class Analysis:
    @classmethod
    def daily_return(cls, stock_price: pd.DataFrame) -> pd.DataFrame: # logarithmic return
        stock_prices_shifted = stock_price.shift(1)
        log_returns = pd.DataFrame()
        for column in stock_price.columns:
            log_returns[column] = np.log(
                stock_price[column] / stock_prices_shifted[column]
            )
        return log_returns

    @classmethod
    def cumulative_return(cls, daily_log_return):
        return np.exp(daily_log_return.cumsum()) - 1

    @classmethod
    def annualised_return(
        cls, daily_log_return: pd.DataFrame, portfolio: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        annualised_return = np.exp(daily_log_return.mean(skipna=True) * 252) - 1
        benchmark_return = annualised_return[Input.benchmark]
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        for i, stock in enumerate(portfolio):
            stock["hist_return"] = annualised_return.iloc[i]
        return annualised_return, benchmark_return

    @classmethod
    def stock_volatility(cls, daily_return: pd.DataFrame, portfolio: list): # sample standard deviation
        volatility = np.std(daily_return, axis=0, ddof=1) * np.sqrt(252)
        benchmark_volatility = volatility[Input.benchmark]
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        for i, stock in enumerate(portfolio):
            stock["volatility"] = volatility.iloc[i]
        return volatility, benchmark_volatility

    @classmethod
    def stock_sharpe(cls, portfolio):
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        tickers = [stock["ticker"] for stock in portfolio]
        hist_return = [stock["hist_return"] for stock in portfolio]
        volatility = [stock["volatility"] for stock in portfolio]
        sharpe_ratio = [
            (historical - Input.rf_rate) / risk
            for historical, risk in zip(hist_return, volatility)
        ]
        data = {ticker: ratio for ticker, ratio in zip(tickers, sharpe_ratio)}
        result = pd.Series(data, dtype=float, name="Sharpe Ratio")
        return result

    @classmethod
    def key_assignment_weight(cls, portfolio: list):
        total_sum = 0
        
        for stock in portfolio:
            ticker_symbol = stock["ticker"]
            stock_data = yf.Ticker(ticker_symbol)
            recent_price = stock_data.history(period="1d")["Close"].iloc[0]
            stock["price"] = recent_price
            total_sum += stock["amount"] * recent_price
        
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        weights = [stock["amount"] * stock["price"] / total_sum for stock in portfolio]
        
        for i, stock in enumerate(portfolio):
            stock["weight"] = weights[i]

    @classmethod
    def key_assignment_stock_sharpe(cls, portfolio):
        for stock in portfolio:
            expected_return = stock["expected_return"]
            volatility = stock["volatility"]
            sharpe = (expected_return - Input.rf_rate) / volatility
            stock["sharpe"] = sharpe

    @classmethod
    def portfolio_historical_return(cls, portfolio):
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        hist_returns = np.array([stock["hist_return"] for stock in portfolio])
        weights = np.array([stock["weight"] for stock in portfolio])
        return np.dot(hist_returns, weights)

    @classmethod
    def portfolio_expected_return(cls, portfolio):
        exp_returns = np.array([stock["expected_return"] for stock in portfolio])
        weights = np.array([stock["weight"] for stock in portfolio])
        return np.dot(exp_returns, weights)

    @classmethod
    def portfolio_volatility(cls, daily_return: pd.DataFrame, portfolio):
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        daily_return_no_benchmark = daily_return.dropna().drop(
            columns=[Input.benchmark]
        )
        covariance_matrix = np.cov(daily_return_no_benchmark.T)
        covariance_matrix_df = pd.DataFrame(
            covariance_matrix,
            columns=daily_return_no_benchmark.columns,
            index=daily_return_no_benchmark.columns,
        )
        weights = np.array([stock["weight"] for stock in portfolio])
        portfolio_volatility = np.sqrt(
            np.dot(np.dot(weights, covariance_matrix), np.transpose(weights)) * 252
        )
        return portfolio_volatility, covariance_matrix_df

    @classmethod
    def portfolio_sharpe(cls, portfolio_return, portfolio_volatility):
        sharpe_ratio = (portfolio_return - Input.rf_rate) / portfolio_volatility
        return sharpe_ratio

    @classmethod
    def display_portfolio_weight(cls, portfolio: list) -> pd.Series:
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        tickers = [stock["ticker"] for stock in portfolio]
        weights = [stock["weight"] for stock in portfolio]
        data = {ticker: weight for ticker, weight in zip(tickers, weights)}
        result = pd.Series(data, dtype=float, name="Weight")
        return result

    @classmethod
    def diversification_benefit(cls, portfolio, portfolio_volatility):
        volatility = np.array([stock["volatility"] for stock in portfolio])
        weights = np.array([stock["weight"] for stock in portfolio])
        weighted_average_volatility = np.dot(volatility, weights)
        diversification_benefit = weighted_average_volatility - portfolio_volatility
        return diversification_benefit


class Optimisation:
    @classmethod
    def min_max_weight(cls, portfolio: list):
        num_tickers = len([stock["ticker"] for stock in portfolio])
        min_weight_input = input(
            f"\nPlease specify a minimum weight between 0 and {round(1 / num_tickers, 2)}"
            f" for portfolio optimisation (or press Enter for 0): "
        ).strip()
        min_weight = float(min_weight_input) if min_weight_input else 0
        while not (0 <= min_weight <= 1 / num_tickers):
            print("Invalid input. Please enter a valid minimum weight.")
            min_weight_input = input(
                f"Enter the minimum weight (between 0 and {1 / num_tickers}): "
            ).strip()
            min_weight = float(min_weight_input) if min_weight_input else 0
        max_weight_input = input(
            f"Please specify a maximum weight between {round(1 / num_tickers, 2)}"
            f" and 1 for portfolio optimisation (or press Enter for 1): "
        ).strip()
        max_weight = float(max_weight_input) if max_weight_input else 1
        while not (1 / num_tickers <= max_weight <= 1):
            print("Invalid input. Please enter a valid maximum weight.")
            max_weight_input = input(
                f"Enter the maximum weight (between {1 / num_tickers} and 1): "
            ).strip()
            max_weight = float(max_weight_input) if max_weight_input else 1
        return min_weight, max_weight

    @classmethod
    def objective(cls, weights, rf_rate, cov_matrix, returns):
        port_return = np.dot(returns, weights)
        port_volatility = np.sqrt(
            np.dot(np.dot(weights, cov_matrix), np.transpose(weights)) * 252
        )
        sharpe_ratio = (port_return - rf_rate) / port_volatility
        return -sharpe_ratio

    @classmethod
    def maximise_sharpe(cls, portfolio, rf_rate, cov_matrix, min_weight, max_weight):
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        bounds = tuple((min_weight, max_weight) for _ in range(len(portfolio)))
        initial_weights = np.array([stock["weight"] for stock in portfolio])
        result = minimize(
            Optimisation.objective,
            initial_weights,
            args=(rf_rate, cov_matrix, [stock["hist_return"] for stock in portfolio]),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result.x

    @classmethod
    def optimal_portfolio_list(cls, portfolio, optimal_weights):
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        return [
            {
                "ticker": stock["ticker"],
                "expected_return": stock["expected_return"],
                "volatility": stock["volatility"],
                "weight": weight,
            }
            for stock, weight in zip(portfolio, optimal_weights)
        ]

    @classmethod
    def portfolio_metrics(cls, portfolio, cov_matrix):
        portfolio = sorted(portfolio, key=lambda x: x["ticker"])
        weights = np.array([stock["weight"] for stock in portfolio])
        expected_returns = np.array([stock["expected_return"] for stock in portfolio])
        port_return = np.dot(expected_returns, weights)
        port_volatility = np.sqrt(
            np.dot(np.dot(weights, cov_matrix), np.transpose(weights)) * 252
        )
        sharpe_ratio = (port_return - Input.rf_rate) / port_volatility
        return port_return, port_volatility, sharpe_ratio


class VaR:
    # Not recommended to modify the VaR values.
    MONTH = 3
    CONFIDENCE = 95
    SIMULATIONS = 10000

    @classmethod
    def calculate(cls, portfolio_return, portfolio_volatility):
        np.random.seed(13)
        daily_return = np.random.normal(
            portfolio_return / 252,
            portfolio_volatility / np.sqrt(252),
            (cls.SIMULATIONS, int(252 * cls.MONTH / 12)),
        )
        sim_value = np.exp(np.cumsum(daily_return, axis=1))
        sim_returns = sim_value[:, -1] - 1
        var = np.percentile(sim_returns, 100 - cls.CONFIDENCE)
        return var, sim_returns


class Plot:
    @classmethod
    def cumulative_returns(cls, cumulative_return):
        plt.figure(figsize=(10, 6))
        for column in cumulative_return.columns:
            plt.plot(cumulative_return.index, cumulative_return[column], label=column)
        plt.title("Cumulative Returns Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: "{:.0%}".format(y))
        )
        plt.legend(loc="upper left")

    @classmethod
    def simulation(cls, var, sim_returns):
        plt.figure(figsize=(10, 6))
        plt.hist(
            sim_returns, bins=50, density=True, alpha=0.75, color="b", edgecolor="black"
        )
        plt.axvline(
            x=var,
            color="r",
            linestyle="dashed",
            linewidth=2,
            label=f"VaR at {VaR.CONFIDENCE}%",
        )
        plt.title(f"VaR over {VaR.MONTH} months")
        plt.xlabel("Portfolio Returns")
        plt.ylabel("Frequency")
        plt.legend()
        plt.xlim(-1, 1)

    @classmethod
    def sharpe_ratio(
        cls,
        portfolio,
        port_name,
        portfolio_return,
        portfolio_volatility,
        opt_return,
        opt_vol,
        benchmark_return,
        benchmark_vol,
    ):
        expected_returns = np.array([stock["expected_return"] for stock in portfolio])
        standard_deviations = np.array([stock["volatility"] for stock in portfolio])
        tickers = np.array([stock["ticker"] for stock in portfolio])
        expected_returns = np.append(
            expected_returns, [portfolio_return, opt_return, benchmark_return]
        )
        standard_deviations = np.append(
            standard_deviations, [portfolio_volatility, opt_vol, benchmark_vol]
        )
        tickers = np.append(
            tickers, [f"[{port_name.upper()}]", "[OPTIMISED]", "[BENCHMARK]"]
        )
        scatter = plt.scatter(
            standard_deviations,
            expected_returns,
            c=(expected_returns - Input.rf_rate) / standard_deviations,
            cmap="viridis",
            marker="o",
        )
        cbar = plt.colorbar(scatter, label="Sharpe Ratio")
        cbar.set_label("Value", rotation=270, labelpad=15)
        plt.title("Sharpe Ratio")
        plt.xlabel("Standard Deviation")
        plt.ylabel("Expected Return")
        rounded_y_ticks = np.arange(0, max(expected_returns) + 0.05, 0.1)
        rounded_y_ticks = np.append(rounded_y_ticks, Input.rf_rate)
        plt.yticks(rounded_y_ticks, [f"{int(tick * 100)}%" for tick in rounded_y_ticks])
        plt.xticks(
            np.arange(0, max(standard_deviations) + 0.05, 0.1),
            [
                f"{int(tick * 100)}%"
                for tick in np.arange(0, max(standard_deviations) + 0.05, 0.1)
            ],
        )
        for i in range(len(expected_returns)):
            plt.plot(
                [standard_deviations[i], 0],
                [expected_returns[i], Input.rf_rate],
                linestyle="-",
                color="gray",
                linewidth=0.5,
            )

        for i, ticker in enumerate(tickers):
            plt.annotate(
                ticker,
                (standard_deviations[i], expected_returns[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
            )

        plt.xlim(0, max(standard_deviations) + 0.05)
        plt.ylim(0, max(expected_returns) + 0.05)


class FileOutput:
    @classmethod
    def csv(cls, file_name, data):
        data.to_csv(file_name)

    @classmethod
    def pdf(cls, file_name, plot):
        plot()
        plt.savefig(file_name, format="pdf")
        plt.close()


if __name__ == "__main__":
    main()
