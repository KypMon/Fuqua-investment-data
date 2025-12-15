
class RegressionInputError(Exception):
    """Raised when regression input fails validation."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or [message]