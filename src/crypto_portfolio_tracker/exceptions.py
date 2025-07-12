class NetworkOperationError(Exception):
    """Raised when a network operation fails after all retries."""
    pass

class DatabaseOperationError(Exception):
    """Raised when a database operation fails."""
    pass

class UserInputError(Exception):
    """Raised for invalid user input."""
    pass

class CriticalAppError(Exception):
    """Raised for unrecoverable, app-wide failures."""
    pass 