import math
import activations


runs=100
hyperparams = {
            "max_generations" :100, # how many generations we run before aborting
            "max_fitness" :3.9/4, # target fitness for early stopping
            "default_activation" : activations.sigmoid, # 1/(1+math.exp(-x)) 
            "breed_probabilities" : {'sexual' : 0.75,'asexual' : 0.25},
            "mutation_probabilities" : { #note: default a = 2
                                            'node' : 0.03,# TODO w1= old_weight, w2= 1
                                            'edge' : 0.05,# add adge TODO w range [-a,a] 
                                            'weight_perturb': 0.8*0.9, # shift weight TODO range alpha in range[0,a]
                                            'weight_set': 0.8*0.1,  # TODO  new_w range[-a,a]
                                            'bias_perturb': 0.12*0.3, # shift bias TODO range alpha in range[0,a]
                                            'bias_set': 0.12*0.7 # TODO new_b range[-a,a]
                                        },
            "distance_weights" :{
                                    'edge': 1.0, #c1
                                    'weight': 1.0, #c2
                                    'bias': 0.4 #c3
                                },
            "interspecies_mating_rate" : 0.001, 
            "patience" : 15, # how many generations to wait for fitness to improve before eliminating a species
            "champion_population_threshold" : 5 # min number of members in a species to clone champion to next generation
        }

def fitness(expected, output):
    """Calculates the similarity score between expected and output."""
    s = 0
    for i in range(4):
        s += (expected[i] - output[i])**2
    return 1/(1 + math.sqrt(s))


def evaluate(genome):
    """Evaluates the current genome."""
    f1 = genome.forward([0.0, 0.0])[0]
    f2 = genome.forward([1.0, 0.0])[0]
    f3 = genome.forward([0.0, 1.0])[0]
    f4 = genome.forward([1.0, 1.0])[0]
    return fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])
