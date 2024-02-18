import torch.nn as nn
import numpy as np


class Node:
    def __init__(self, node_id: int, node_type: str, activation: nn.Module = nn.Identity()):
        """
        A node in a neural network.
        Parameters:
        ===========
        - node_id (int): The index of the node.
        - node_type (str): The type of the node. Can be 'input', 'output', or 'hidden'.
        - activation (nn.Module): The activation function of the node.
            Default is nn.Identity().
        """

        self.node_id = node_id
        self.type = node_type
        self.activation = activation

    def __str__(self):
        return "({} ({})".format(
            self.node_id,
            self.type,
            # self.activation.__class__.__name__
        )

    def __repr__(self):
        return str(self)


class Connection:
    def __init__(self, in_node: int, out_node: int, active: bool, weight: float = None):
        """
        A connection between two nodes in a neural network.
        Parameters:
        ===========
        - in_node (int): The index of the input node.
        - out_node (int): The index of the output node.
        - active (bool): Whether the connection is active or not.
        - weight (float): The weight of the connection.
            If None, a random weight is assigned.
        """

        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight if weight is not None else np.random.normal(0, 1)
        self.active = active
        self.innov = self.__get_innov()

    def __get_innov(self):
        """
        Get the innovation number of the connection.
        """
        # TODO: Implement this method. For now, return 0.
        return 0

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
