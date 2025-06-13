"""https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call"""

import signal
import functools
import multiprocessing
import time
import os


class TimeoutError(Exception):
    """Exception raised when a function times out."""
    pass


def _timeout_target(func, args, kwargs, result_queue, error_queue):
    """Module-level target function for timeout multiprocessing (must be picklable)."""
    try:
        result = func(*args, **kwargs)
        result_queue.put(result)
    except Exception as e:
        error_queue.put(e)


def timeout(seconds):
    """Decorator that raises a TimeoutError if the function takes too long to execute.
    Uses multiprocessing instead of threading to better handle CUDA operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a queue to get the result from the process
            result_queue = multiprocessing.Queue()
            error_queue = multiprocessing.Queue()
            
            # Create and start the process using module-level function
            process = multiprocessing.Process(
                target=_timeout_target, 
                args=(func, args, kwargs, result_queue, error_queue)
            )
            process.start()
            
            # Wait for the process to complete or timeout
            process.join(seconds)
            
            # If the process is still running, terminate it
            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Check for errors
            if not error_queue.empty():
                raise error_queue.get()
            
            # Get the result
            if not result_queue.empty():
                return result_queue.get()
            
            return None
            
        return wrapper
    return decorator
