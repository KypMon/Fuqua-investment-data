class BacktestInputError(Exception):
    """Raised when user input for the backtest fails validation."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or [message]