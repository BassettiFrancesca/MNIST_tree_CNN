import torchvision
import torchvision.transforms as transforms
import prepare_dataset
import prepare_leaf_dataset
import Node


def tree():

    leaf_groups = [[[2], [4]], [[2], [4]], [[2], [7]], [[2], [7]], [[1], [4]], [[1], [4]], [[4], [7]], [[4], [7]],
                   [[3], [5]], [[3], [5]], [[3], [8]], [[8], [9]], [[0], [9]], [[0], [9]], [[0], [6]], [[0], [6]]]

    leaf_PATHS = ['./leaf1_net.pth', './leaf2_net.pth', './leaf3_net.pth', './leaf4_net.pth',
                  './leaf5_net.pth', './leaf6_net.pth', './leaf7_net.pth', './leaf8_net.pth',
                  './leaf9_net.pth', './leaf10_net.pth', './leaf11_net.pth', './leaf12_net.pth',
                  './leaf13_net.pth', './leaf14_net.pth', './leaf15_net.pth', './leaf16_net.pth']

    leaves = []

    for i, leaf_group in enumerate(leaf_groups):
        (leaf_train_set, leaf_test_set) = prepare_leaf_dataset.prepare_leaf_dataset(leaf_group)
        leaves.append(Node.Node(leaf_train_set, leaf_test_set, None, None, leaf_PATHS[i], True))

    for leaf in leaves:
        leaf.train(2)
        leaf.test()

    train_sets = []
    test_sets = []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_sets.append(torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform))
    test_sets.append(torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform))

    groups = [[1, 2, 4, 7], [0, 3, 5, 6, 8, 9], [2, 4, 7], [1, 4, 7], [3, 5, 8, 9], [0, 6, 9], [2, 4], [2, 7],
              [1, 4], [4, 7], [3, 5], [3, 8, 9], [0, 9], [0, 6]]

    for group in groups:
        (train_set, test_set) = prepare_dataset.prepare_dataset(group)
        train_sets.append(train_set)
        test_sets.append(test_set)

    PATHS = ['./node1_net.pth', './node2_net.pth', './node3_net.pth', './node4_net.pth', './node5_net.pth',
             './node6_net.pth', './node7_net.pth', './node8_net.pth', './node9_net.pth', './node10_net.pth',
             './node11_net.pth', './node12_net.pth', './node13_net.pth', './node14_net.pth', './node15_net.pth']

    train_sets.reverse()
    test_sets.reverse()
    PATHS.reverse()
    leaves.reverse()

    nodes = []

    i = 0

    for j in range(0, len(leaves), 2):
        nodes.append(Node.Node(train_sets[i], test_sets[i], leaves[j + 1], leaves[j], PATHS[i], False))
        i += 1

    l = 0

    for k in range(i, len(train_sets), 1):
        nodes.append(Node.Node(train_sets[k], test_sets[k], nodes[l + 1], nodes[l], PATHS[k], False))
        l += 2

    for node in nodes:
        node.train(4)
        node.test()


if __name__ == '__main__':
    tree()
