import torch.nn as nn


class Gene:
    def __init__(self,
                 in_node: int,
                 out_node: int,
                 weight: float, active: bool,
                 innov: int,
                 out_activation: nn.Module = nn.Identity()):
        """
        A connection between two nodes in a neural network.
        Parameters:
        ===========
        - in_node (int): The index of the input node.
        - out_node (int): The index of the output node.
        - weight (float): The weight of the connection.
        - active (bool): Whether the connection is active or not.
        - innov (int): The innovation number of the connection.
        - out_activation (nn.Module): The activation function of the output node.
            Default is nn.Identity().
        """

        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.active = active
        self.innov = innov
        self.out_activation = out_activation

    def __str__(self):
        return "Connection: {} {} --- w = {:.2f}  --> {}({}) (innov={})" \
            .format('' if self.active else '(X)',
                    self.in_node,
                    self.weight,
                    self.out_activation.__class__.__name__,
                    self.out_node,
                    self.innov)

    def __repr__(self):
        return str(self)
