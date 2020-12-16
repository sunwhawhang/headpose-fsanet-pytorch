from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt


# Number of samplepoints
N = 600

# Sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
# plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
plt.plot(range(len(yf)), np.abs(yf))
plt.grid()
plt.figure()
plt.plot(range(len(y)), y)
plt.show()



from scipy.fft import fft, ifft
import pandas as pd
import matplotlib.pyplot as plt


# Specify the data we want to use
head_pose = "shake"
data_index = "5"

# Load the data
df = pd.read_csv(f"collected_data/{head_pose}_{data_index}")
yaw = df.yaw.values
pitch = df.pitch.values
roll = df.roll.values

yaw_f = fft(yaw)
pitch_f = fft(pitch)
roll_f = fft(roll)

plt.figure()
plt.plot(range(len(yaw)), yaw, 'r-', label='yaw')
plt.plot(range(len(pitch)), pitch, 'b-', label='pitch')
plt.plot(range(len(roll)), roll, 'g-', label='roll')
plt.legend()


plt.figure()
plt.plot(range(len(yaw_f)), yaw_f, 'r-', label='yaw_f')
plt.plot(range(len(pitch_f)), pitch_f, 'b-', label='pitch_f')
plt.plot(range(len(roll_f)), roll_f, 'g-', label='roll_f')
plt.legend()

plt.show()
