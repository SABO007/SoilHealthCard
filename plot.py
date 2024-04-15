import matplotlib.pyplot as plt

def plot(dict):
    for key in dict:
        X = ['Low', 'Actual', 'High']
        plt.figure(figsize=(10, 6))
        plt.plot(X, dict[key], color='red', label=key)
        plt.xlabel('Range')
        plt.ylabel('Values')
        plt.title(f'{key} Level')
        plt.legend()
        plt.savefig(f'static/{key}.png')