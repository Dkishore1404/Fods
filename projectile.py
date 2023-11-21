import numpy as np

# Assuming you have two NumPy arrays representing time intervals and vertical positions
time_intervals = np.array([t1, t2, t3, ...])  # Replace t1, t2, t3, ... with your actual time values
vertical_positions = np.array([h1, h2, h3, ...])  # Replace h1, h2, h3, ... with your actual height values

# Step 1: Calculate vertical displacement
vertical_displacement = np.diff(vertical_positions)

# Step 2: Calculate time intervals
time_displacement = np.diff(time_intervals)

# Step 3: Calculate average velocity
average_velocity = vertical_displacement / time_displacement

# The result will be a NumPy array containing the average velocities at each time interval
