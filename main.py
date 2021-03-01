import Node
import testing_node
import testing_leaf
import divide_dataset
import torch
import Subset


def tree():
    train_set, test_set = divide_dataset.divide_dataset([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    #PATHS = ['./node_net.pth', './left_net.pth', './right_net.pth']
    #node = Node.Node(train_set, PATHS[1], PATHS[2], PATHS[0])
    print(train_set.get_item(4))
    print(train_set.__getitem__(4))
    left_dataset1 = torch.utils.data.Subset(train_set, [0, 3, 4])
    left_dataset = Subset.Subset(left_dataset1, train_set, [0, 3, 4])

    print(left_dataset.get_item(2))
    print(left_dataset.__getitem__(2))


if __name__ == '__main__':
    tree()
