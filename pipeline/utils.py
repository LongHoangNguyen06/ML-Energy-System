import random
import socket
from datetime import datetime


def current_time_str():
    # Get the current time
    now = datetime.now()
    # Get the hostname
    hostname = socket.gethostname()
    # Format the time and date to include milliseconds in the specified format
    current_time = now.strftime("%H:%M:%S:%f %d-%m-%Y")[
        :-3
    ]  # Remove the last three digits of microseconds to get milliseconds
    # Generate a random number between 0 and 1000
    random_number = random.randint(0, 1000)
    # Combine the hostname, current time string, and random number
    return f"{hostname}_{current_time}_{random_number}"
