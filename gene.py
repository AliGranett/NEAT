from chromosome import Chromosome

class Gene:
    def __init__(self, connections: list[Chromosome]):
        """
        Gene class for the connection between two nodes
        Parameters:
        connections (list[Connection]): List of connections
        """
        self.connections = connections
        self.fitness = 0
        self.adjusted_fitness = 0

    def __str__(self):
        connections = '\n'.join([str(connection) for connection in self.connections])
        return f'Gene:\n{connections}'

    def __repr__(self):
        return self.__str__()
