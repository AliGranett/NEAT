from gene import Gene
from chromosome import Chromosome
if __name__ == '__main__':
    print('This is the main program')
    # Create a gene
    gene = Gene([Chromosome(1, 2, 0.5, True, 1), Chromosome(2, 3, 0.5, True, 1)])
    print(gene)