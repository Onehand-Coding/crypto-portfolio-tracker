class NetworkOperationError(Exception):
    """
    Raised when a network operation fails after all retries.
    
    This exception is used to indicate that a network operation has failed
    despite retry attempts, and typically results in a graceful fallback
    or error handling in the application.
    """
    pass


class NetworkUnavailableError(Exception):
    """
    Raised when network is unavailable and offline mode should be triggered.
    
    This exception is specifically used to signal that the network is not
    accessible, and the application should enter offline mode to continue
    functioning with local data only.
    """
    pass


class DatabaseOperationError(Exception):
    """
    Raised when a database operation fails.
    
    This exception indicates that a database operation (query, insert, update, etc.)
    has failed, possibly due to connectivity issues, constraint violations, or
    other database-related problems.
    """
    pass


class UserInputError(Exception):
    """
    Raised for invalid user input.
    
    This exception is used when user input does not meet validation requirements
    or is otherwise invalid for the requested operation.
    """
    pass


class CriticalAppError(Exception):
    """
    Raised for unrecoverable, app-wide failures.
    
    This exception indicates a critical failure that prevents the application
    from continuing normal operation and typically requires immediate attention
    or application restart.
    """
    pass
