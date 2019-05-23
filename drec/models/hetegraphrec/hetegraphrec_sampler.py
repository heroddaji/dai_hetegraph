import random
import numpy as np
import torch


class HeteGraphRecNodeSampler():
    def __init__(self, dataset, **kwargs):
        """
        :param
        - path: a dict of node label and the path for each label
            eG: {'user':['movie','user','movie']}
        - edge_weight: the edge attribute that acts as weight value
        - edge_choice: the strategy to choose which node to go next based on edge_weight
            default is "similar", node with si
        """
        self.ds = dataset
        self.path = kwargs.get('path', None)
        if not self.path:
            raise Exception("need \"path\" parameter")
        self.edge_weight_attr = kwargs.get("edge_weight_attr", 'score')
        self.edge_weight_choice = kwargs.get("edge_weight_choice", 'highest')  # highest, lowest or a number (score)
        self.nbs_per_node = kwargs.get("nbs_per_node", 10)  # amount of nbs to sample per ndoe
        self.edge_scores = kwargs.get("edge_scores", [1, 2, 3, 4, 5])
        self._process_score_priority()

    def _process_score_priority(self):
        self.score_priorty = self.edge_scores
        if self.edge_weight_choice == 'higest':
            self.score_priorty.sort(reverse=True)
        if self.edge_weight_choice == 'lowest':
            self.score_priorty.sort()

    def sample_neighborhood(self, nodes):
        print('sampling nbs...')
        nbs_sampling = {}
        g = self.ds.graph
        for node_id in nodes:
            node = g.nodes[node_id]
            node_label = node['label']
            node_label_path = self.path[node_label]
            nbs_sampling[node_id] = []
            for i in range(self.nbs_per_node):
                for idx, nb_label in enumerate(node_label_path):
                    if idx == 0:
                        random_node_id = node_id
                        nbs_sampling[node_id].append([])
                        random_node_id = self._select_random_node_of_label(g, random_node_id, nb_label)
                    else:
                        random_node_id = self._select_random_node_of_label(g, random_node_id, nb_label)

                    nbs_sampling[node_id][-1].append(random_node_id)
        return nbs_sampling

    def sample_negative_nodes(self, nodes, size=200, types={'user': 'item', 'item': 'user'}):
        g = self.ds.graph
        # find negative nodes for each key pair (if user type, find item negative node, or if item type, find user negative nodes)
        # todo: random nodes for now, find a better method later on
        neg_nodes_dict = {k: [] for k, v in types.items()}
        r_nodes = random.choices(list(g.nodes), k=size)
        for n in r_nodes:
            neg_nodes_dict[g.nodes[n]['label']].append(n)

        return neg_nodes_dict

    def _select_random_node_of_label(self, g, node_id, label):
        # retrieve nb and sort the scores
        nb_nodes = g[node_id]
        is_reverse = False
        if self.edge_weight_choice == "highest":
            is_reverse = True
        sorted_nb_nodes = sorted(nb_nodes.items(), key=lambda kv: kv[1]['score'], reverse=is_reverse)
        selected_nb_nodes = []
        for nb_node in sorted_nb_nodes:
            if g.nodes[nb_node[0]]['label'] == label:
                selected_nb_nodes.append(nb_node[0])

        if len(selected_nb_nodes) == 0:
            return node_id

        # cut down the size
        new_size = len(selected_nb_nodes)
        if new_size > 50:
            new_size = 20
        elif new_size > 30:
            new_size = 15
        else:
            new_size = int(new_size / 2)

        if new_size <= 1:
            new_size = 1

        selected_nb_nodes = selected_nb_nodes[:new_size]

        # new select a random node in this list
        selected_random_nb_node = random.choice(selected_nb_nodes)
        return selected_random_nb_node

    def get_nbs_attr(self, node, node_walks):
        '''
        only get the neighborhood node attribute who has the same label with target node
        '''
        g = self.ds.graph
        node_label = g.nodes[node]['label']
        index_to_pick = self.path[node_label].index(node_label)
        real_nb = []
        for nb in node_walks:
            nb_node = nb[index_to_pick]
            real_nb.append(g.nodes[nb_node]['feature'])
        return np.stack(real_nb)

    def get_nbs_attr_from_embeddings(self, node, node_walks, unique_nodes_sorted, nodes_embeds_sorted):
        g = self.ds.graph
        node_label = g.nodes[node]['label']
        index_to_pick = self.path[node_label].index(node_label)
        real_nb = []
        for nb in node_walks:
            nb_node = nb[index_to_pick]
            real_nb.append(nodes_embeds_sorted[unique_nodes_sorted.index(nb_node), :])
        return torch.stack(real_nb)


class HeteGraphRecConcatNodeSampler(HeteGraphRecNodeSampler):
    '''
    this class get neighborhood attribute will concat all nodes each path path
    '''

    def __init__(self, dataset, **kwargs):
        super(HeteGraphRecConcatNodeSampler, self).__init__(dataset, **kwargs)

    def get_nbs_attr(self, node, node_walks):
        '''
        concat all node attribute in the path
        '''
        g = self.ds.graph
        real_nb = []
        for nb in node_walks:
            nb_attrs = tuple([g.nodes[nb_node]['feature'] for nb_node in nb])
            real_nb.append(np.hstack(nb_attrs))
        return np.stack(real_nb)

    def get_nbs_attr_from_embeddings(self, node, node_walks, unique_nodes_sorted, nodes_embeds_sorted):
        real_nb = []
        for nb in node_walks:
            nb_attrs = [nodes_embeds_sorted[unique_nodes_sorted.index(nb_node), :] for nb_node in nb]
            real_nb.append(torch.cat(nb_attrs))
        return torch.stack(real_nb)