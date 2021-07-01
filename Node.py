import torch
import training_node
import training_leaf
import testing_node
import testing_leaf
import CNN1


class Node:
    def __init__(self, data_set, test_set, left_child, right_child, PATH, leaf):
        self.data_set = data_set
        self.test_set = test_set
        self.left_child = left_child
        self.right_child = right_child
        self.PATH = PATH
        self.leaf = leaf

    def train(self, num_epochs):
        if not self.leaf:
            training_node.train(self, num_epochs)
        elif self.leaf:
            training_leaf.train(self.data_set, self.PATH, num_epochs)

    def test(self):
        if not self.leaf:
            return testing_node.test(self)
        elif self.leaf:
            return testing_leaf.test(self.test_set, self.PATH)

    def get_predicted(self, image):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        node_net = CNN1.Net().to(device)
        node_net.load_state_dict(torch.load(self.PATH))
        node_output = node_net(image)
        _, node_predicted = torch.max(node_output.data, 1)

        if self.leaf:
            node_predicted[0] = self.data_set.groups[node_predicted[0]][0]
            return node_predicted
        elif node_predicted[0] == 0:
            return self.left_child.get_predicted(image)
        elif node_predicted[0] == 1:
            return self.right_child.get_predicted(image)
