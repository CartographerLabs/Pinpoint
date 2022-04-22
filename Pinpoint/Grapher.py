import networkx as nx


class grapher():
    """
    A wrapper class used for generating a graph for interactions between users
    """
    graph = None

    def __init__(self):
        """
        Constructor.
        """
        self.graph = nx.DiGraph()

    def add_edge_wrapper(self, node_1_name, node_2_name, weight, relationship):
        """
        A wrapper function used to add an edge connection or node.
        :param node_1_name: from
        :param node_2_name: to
        :param weight:
        :param relationship:
        :return:
        """
        self.graph.add_edge(node_1_name, node_2_name, weight=weight, relation=relationship)

    def add_node(self, node_name):
        """
        A wrapper function that adds a node with no edges to the graph
        :param node_name:
        """
        self.graph.add_node(node_name)

    def get_info(self):
        """
        Retrieves information about the graph
        :return:
        """
        return nx.info(self.graph)

    def show_graph(self):
        """
        Displays the graph
        :return:
        """
        nx.spring_layout(self.graph)

    def get_degree_centrality_for_user(self, user_name):
        """
        Returns the Degree of Centrality for a given user present in the graph
        :param user_name:
        :return: the Degree of Centrality for a given user present in the graph
        """
        centrality = nx.degree_centrality(self.graph)
        return centrality[user_name]

    # todo implement
    # def get_eigenvector_centrality_for_user(self, user_name):
    #    centrality = nx.eigenvector_centrality(self.graph)
    #    return centrality[user_name]
