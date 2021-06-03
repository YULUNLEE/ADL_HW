import matplotlib.pyplot as plt
import numpy as np

rouge1=[12.2477, 14.9803, 16.7324, 17.547, 18.6295, 19.4984, 20.2891, 21.2723, 22.4324, 23.0355, 23.5687, 24.2178, 24.9658, 23.9425, 24.3596, 25.5854, 26.3314, 25.2384, 24.9654, 24.572]
rouge2=[5.6871, 6.1587, 6.5217, 7.5896, 7.7526, 7.6128, 8.5478, 9.0125, 8.8325, 9.2148, 9.4632, 9.5621, 9.6325, 9.9524, 9.4326, 10.2398, 10.95, 9.8523, 9.5214, 9.241]
rougeL=[9.3245, 11.5703, 13.0384, 14.847, 15.795, 16.1484, 17.0091, 18.2723, 19.5147, 20.1455, 20.4682, 21.1278, 21.8618, 20.8425, 21.2506, 22.9854, 23.5714, 22.7384, 22.3644, 21.8572]
print(len(rouge2))
print(np.linspace(1, 20, 20))
plt.title('Text2Text Summarization Training Process')
plt.plot(np.linspace(1, 20, 20), rouge1, label='rouge1')
plt.plot(np.linspace(1, 20, 20), rouge2, '--', label="rouge2")
plt.plot(np.linspace(1, 20, 20), rougeL, linestyle='dotted', label="rougeL")
plt.xticks(np.linspace(1, 20, 20))
plt.xlabel('epochs')
plt.ylabel('f1-score(%)')
plt.legend()
plt.show()