import numpy as np
import random
import pickle # Model to save objects directly to a file
import os

from training import create_model, prep_data, train
import rectangles


SIZE    = 30  # Number of networks in each generation
RETAIN  = .25 # Fraction of best networks that will be retained after each generation
RANDOMS = .1  # Fraction of the total amount of networks that will be added as randomly generated
              #  networks after each generation (doubled in first 5 generations)


class NeuralNetwork:
    """
    Superclass for AutoEncoder and Rectangles objects; contains the methods
     and attributes that are identical for both
    """
    param_mutation_chance = .2      # See the mutate method
    
    def __init__(self, params=None):
        """ No documentation necessary """
        self.params        = params # Dict with as keys parameter names and as values the current value of that parameter
        self.param_options = None   # Same as above, but the values are lists of all possible values
        self.fit           = None   # Fitness value (loss) of the current network if calculated
        self.model         = None   # Created model of the current network if it exists
        self.trained_model = None   # Trained model of the current network if it exists
        
        if self.params is None: self.set_random_params() # Upon initialisation without parameters, randomly set them
        
    
    def set_random_params(self):
        """
        Randomly set the instance's parameters
        """
        self.params = {key: random.choice(values) for key, values in self.param_options.items()}
        return self
        
    def make_child(self, other):
        """
        Make a "child" (random combination of two "parent" networks)
        """
        return self.__class__({key: [self.params[key], other.params[key]][random.random() < .5] for key in self.params})
        
    def mutate(self, r_weight=1, r_power=1.5):
        """
        # Arne vult nog wel aan. -N-i---h-o-p-e-n---x-
        0 < r_weight < 1
        
        AANVULLING ARNE:
            Mutate the value of each parameter of a network with a chance of NeuralNetwork.param_mutation_chance
            The r_weight and r_power parameters ensure values closer to the current one are more likely to be mutated into
        """
        for param in self.params:
            if random.random() > NeuralNetwork.param_mutation_chance:
                continue
            
            index = self.param_options[param].index(self.params[param])
            
            self.params[param] = self.param_options[param][index + int(round(r_weight * random.random() ** r_power * random.choice([-index, len(self.param_options[param]) - index - 1]), 0))]

        self.model = self.trained_model = self.fit = None

    def make_model(self):
        """ Separately defined in subclasses """
        pass
    
    def train_model(self):
        """ Separately defined in subclasses """
        pass

    def fitness(self):
        """ Separately defined in subclasses """
        pass


class AutoEncoder (NeuralNetwork):
    """
    Class of denoising autoencoders. The class's primary target is not to hold an actual autoencoder, but to hold the parameters used in initialising and training one
    """
    epochs  = 25
    lengths = list(range(32, 129, 16)) # Possible network input sizes
    
    # Prepare training and testing data for each of the possible input sizes
    # [*zip(*lst)] is essentially "lst.transpose()" (which doesn't exist)
    #xss_train, yss_train, xss_test, yss_test = [*zip(*[prep_data(((2, 9),), length=length, stride=24) for length in lengths])]
    
    def __init__(self, params=None):
        """ No documentation necessary (see superclass for meaning of attributes) """
        self.params = params
        self.param_options = {
                "num_filters"     : list(range(4,16)),
                "encoding"        : [n/128 for n in range(24,72)],
                "length"          : AutoEncoder.lengths,
                "dense_activation": ["relu", "softmax", "linear", "tanh", "elu"],
                "conv_activation1": ["relu", "softmax", "linear", "tanh", "elu"],
                "conv_activation2": ["relu", "softmax", "linear", "tanh", "elu"],
                "optimizer"       : ["rmsprop", "adam", "adadelta", "sgd", "adagrad"],
                }
        self.fit           = None
        self.model         = None
        self.trained_model = None
        
    def make_model(self):
        """ Create an untrained model with the instance's parameters """
        if self.model is None:
            self.model = create_model(**self.params)
        return self.model
    
    def train_model(self):
        """ Fairly self-explanatory """
        if self.trained_model is not None: return self.trained_model
        model = self.make_model()
        
        # The index of the network's length in the list of possible lengths will be
        #  the same as the index of the prepped data for that length in the list of prepped data
        i = self.param_options["length"].index(self.params["length"])
        train(model, AutoEncoder.xss_train[i], AutoEncoder.yss_train[i], epochs=AutoEncoder.epochs, verbose=0)
        self.trained_model = model
        
        return self.trained_model
    
    def fitness(self):
        """ Find the value of the loss function, using the testing data """
        print(f"   {self.params}")
        
        if self.fit is None:
            model = self.train_model()
            
            # The index of the network's length in the list of possible lengths will be
            #  the same as the index of the prepped data for that length in the list of prepped data
            i = self.param_options["length"].index(self.params["length"])
            self.fit = model.evaluate(np.expand_dims(AutoEncoder.xss_test[i], axis=2), AutoEncoder.yss_test[i], verbose=0)
        
        print(f"   {round(self.fit, 6)}")
        print()
        return self.fit
        
    
class Rectangles (NeuralNetwork):
    """
    Analogous to AutoEncoder
    """
    epochs  = 20
    lengths = list(range(48, 305, 16))
    
    # [*zip(*lst)] is essentially "lst.transpose()" (which doesn't exist)
    if __name__ == "__main__":
        xss_train, yss_train, xss_test, yss_test = [*zip(*[rectangles.get_cross_validation([(1, 5), (2,9), (3,7)], length=length) for length in lengths])]
    else:
        xss_train = yss_train = xss_test = yss_test = None
    
    def __init__(self, params=None):
        self.params = params
        self.param_options = {
                "n_conv":     [i for i in range(3)],
                "dense_size": [i for i in range(100, 1000)],
                "n_dense":    [i for i in range(2)],
                "length":     Rectangles.lengths,
                "optimizer":  ["rmsprop", "adam", "adadelta", "sgd", "adagrad"], 
                "dense_activation":  ["relu", "softmax", "linear", "tanh", "elu"],
                "conv_activation":   ["relu", "softmax", "linear", "tanh", "elu"]
                }
        self.fit           = None
        self.model         = None
        self.trained_model = None
        
    def make_model(self):
        if self.model is None:
            self.model = rectangles.create_model(**self.params)
        return self.model
    
    def train_model(self):
        if self.trained_model is not None: return self.trained_model
        
        i = self.param_options["length"].index(self.params["length"])
        
        model = self.make_model()
        model = rectangles.train_model(Rectangles.xss_train[i], Rectangles.yss_train[i], save=False, model=model, eps=Rectangles.epochs, verbose=1)
        
        self.trained_model = model
        return self.trained_model
    
    def fitness(self):
        if self.fit is None:
            
            model = self.train_model()
            
            i = self.param_options["length"].index(self.params["length"])
            if not self.params["n_conv"]:
                self.fit = model.evaluate(Rectangles.xss_test[i], Rectangles.yss_test[i], verbose=0)
            else:
                self.fit = model.evaluate(np.expand_dims(Rectangles.xss_test[i], axis=2), Rectangles.yss_test[i], verbose=0)
            
        print(f"   {round(self.fit, 6)}")
        return self.fit
    
    
MODEL_CLASS = Rectangles # Default model class


class Population:
    """
    Class representing a single generation (population) of models
    """
    def __init__(self, history=None, models=None, generation=0, model_class=MODEL_CLASS, size=SIZE, retain=RETAIN, randoms=RANDOMS):
        """ No documentation necessary """
        self.model_class = model_class # Model class of the population
        self.size        = size        # }
        self.retain      = retain      #  } Analogous to the globally defined defaults
        self.randoms     = randoms     # }
        
        self.fits        = None        # List of fitness (loss) values of each of the population's models, if calculated
        
        self.history     = [] if history is None else history   # "History" of preceding populations
        self.models      = models                               # NeuralNetwork subclass objects of the population
        self.generation  = generation                           # Current generation number
        if self.models is None: self.create_random_population() # If no models are given, randomly create them

    def create_random_population(self, model_class=MODEL_CLASS):
        """ See method name """
        self.models = [model_class().set_random_params() for _ in range(self.size)]
        return self.models

    def get_fitnesses(self):
        """ See method name """
        if self.fits is None:
            self.fits = [m.fitness() for m in self.models]
            
        return self.fits

    def evolve(self):
        """
        Evolves a generation into a new one
        """
        # Sort the generation's models by their fitnesses
        by_fitness = sorted(zip(self.get_fitnesses(), self.models), key=lambda t: t[0])
        
        # Separate the tuples into two lists
        fitnesses, sorted_models = map(list, zip(*by_fitness))
        
        # Print some stuff, because why not
        print(", ".join(map(str, [round(f, 6) for f in fitnesses])))
        print(round(sum(fitnesses) / self.size, 6))
        
        # Calculate actual numbers of models from the fractions
        retain_length  = int(self.retain * self.size + .5)
        randoms_length = int(self.randoms * self.size + .5) * (1 + (self.generation < 5)) # Add more randoms if we're at the early stages of training
        
        # Only keep the best models, and generate some random ones
        models       = sorted_models[:retain_length]
        randoms      = [self.model_class().set_random_params() for _ in range(randoms_length)]
        
        # Add as many children as needed to get back to full size
        new_models = []
        for _ in range(self.size - len(models) - randoms_length):
            child = NeuralNetwork.make_child(*random.sample(models, 2))
            child.mutate()
            new_models.append(child)
            
        # Compose the new list of models
        models += randoms + new_models

        # Keras models can't be pickled (see later), so delete them once we don't need them anymore
        # Also delete the fitnesses of the models, to force recalculating them, making it less likely that
        #  a randomly above average training will influence the results
        for model in self.models:
            model.model = model.trained_model = model.fit = None
            
        # Create a new population object
        new_population = Population(history=self.history + [self], 
                                    models=models,
                                    generation=self.generation + 1,
                                    model_class=self.model_class,
                                    size=self.size,
                                    retain=self.retain,
                                    randoms=self.randoms)
        return new_population
    

def train_population(generations, pop=None, pickle_folder=None, size=SIZE, model_class=MODEL_CLASS, retain=RETAIN, randoms=RANDOMS):
    """
    Train a certain number of generations
    
    Optional arguments include an initial population, and a folder to save ("pickle") each generation after being trained
    """
    # Make the pickle folder if it doesn't yet exist
    if pickle_folder is not None and not os.path.exists(pickle_folder):
        os.mkdir(pickle_folder)
    
    # Initialise a new (random) population if none was given
    if pop is None:
        pop = Population(model_class=model_class, size=size, retain=retain, randoms=randoms)
    
    start_gen = pop.generation
    
    # Train each generation
    for gen in range(start_gen, start_gen + generations):
        print(f"Generation {gen}:")
        
        new_pop = pop.evolve()
        # Pickle if a pickle folder was given
        if pickle_folder is not None:
            with open(f"{pickle_folder}/generation_{gen}.pickle", "wb+") as file:
                pickle.dump(pop, file)
        
        pop = new_pop
        print(end="\n\n\n")
        
    return pop


"""
Script to plot some results after training

import matplotlib.pyplot as plt
for i in range(16):
    with open(f"weekend_run2/generation_{i}.pickle", "rb") as file:
        p = pickle.load(file)
    
    ls = [t.params["length"] for t in p.models]
    cs = {v: ls.count(v) for v in ls}
    plt.bar(cs.keys(), cs.values())
    plt.show()
"""