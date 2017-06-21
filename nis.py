import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

with open('laser_nis.txt') as laser_file:
    laser = [float(line.strip()) for line in laser_file]
with open('radar_nis.txt') as radar_file:
    radar = [float(line.strip()) for line in radar_file]

radar_line = 7.815
laser_line = 5.991

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(laser)
plt.plot((0, 250), (laser_line, laser_line), label=str(laser_line))
plt.title('laser NIS')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(radar)
plt.plot((0, 250), (radar_line, radar_line), label=str(radar_line))
plt.title('radar NIS')
plt.legend()

plt.savefig('nis.png')