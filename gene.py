class Gene:
    def __init__(self,
                 in_node: int,
                 out_node: int,
                 weight: float, active: bool,
                 innov: int):
        """
        A connection between two nodes in a neural network.
        Parameters:
        ===========
        - in_node (int): The index of the input node.
        - out_node (int): The index of the output node.
        - weight (float): The weight of the connection.
        - active (bool): Whether the connection is active or not.
        - innov (int): The innovation number of the connection.
        """

        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.active = active
        self.innov = innov

    def __str__(self):
        return f"Connection{'' if self.active else '(X)'}: {self.in_node} -> {self.out_node} ({self.weight=} {self.innov=})"

    def __repr__(self):
        return str(self)
