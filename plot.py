import json
import matplotlib.pyplot as plt

# Load the JSON file
with open('1754553587_1.json', 'r') as file:
    data = json.load(file)

# Extract sensor data and timestamps
sensor_data = {}
timestamps = []

for entry in data:
    if 'timestamp' in entry:
        timestamps = entry['timestamp']
    else:
        for key, value in entry.items():
            if key.startswith('value'):
                sensor_data[key] = value

# Plotting
plt.figure(figsize=(14, 8))

for sensor, values in sensor_data.items():
    if len(values) == len(timestamps):
        plt.plot(timestamps, values, label=sensor)
    else:
        print(f"Sensor {sensor} has mismatched data length and will not be plotted.")

plt.xlabel('Timestamp (Nanoseconds)')
plt.ylabel('Analog Sensor Value')
plt.title('Analog Sensor Data vs. Timestamp (Nanoseconds)')
plt.legend()
plt.grid(True)
plt.ticklabel_format(style='plain', axis='x')  # Disable scientific notation for x-axis
plt.show()