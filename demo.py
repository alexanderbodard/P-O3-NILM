import pickle
import matplotlib.pyplot as plt
import os
from metalearning import Population, Rectangles, NeuralNetwork

def plot_best_evolution(meta_folder="Rectangles/Rectangles", n=10):
    n_generations = len([name for name in os.listdir(meta_folder) if os.path.isfile(os.path.join(meta_folder, name))])
    average = []
    for gen in range(n_generations):
        with open(meta_folder + f"/generation_{gen}.pickle", "rb") as file:
            p = pickle.load(file)
            
        a= []
        for i in enumerate(p.fits):
            a.append(i)
        a = sorted(a, key=lambda x: x[1])
        b =  sum(list(map(lambda x: x[1], a[:n])))/n*10
        average.append(b)
        
    plt.scatter(list(range(1, n_generations+1)), average)
    plt.plot(list(range(1, n_generations+1)), average)  
    plt.hlines(1.56235, xmin=0, xmax=n_generations+1)

    plt.xlim(left=0)
    plt.xlim(right=n_generations+1)
    plt.ylim(bottom = min(average) * 0.9)
    plt.ylim(top= max(average) * 1.1)
    plt.show()
