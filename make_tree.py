import torchvision
import torchvision.transforms as transforms
import prepare_dataset
import prepare_leaf_dataset
import Node
import time


def make_tree(num_epochs_l, num_epochs_n, leaf_groups, node_groups):

    start = time.time()

    leaf_PATHS = []

    for i in range(len(leaf_groups)):
        leaf_PATHS.append('./PATHS/leaf' + str(i + 1) + '_net.pth')

    leaves = []

    for i, leaf_group in enumerate(leaf_groups):
        (leaf_train_set, leaf_test_set) = prepare_leaf_dataset.prepare_leaf_dataset(leaf_group)
        leaves.append(Node.Node(leaf_train_set, leaf_test_set, None, None, leaf_PATHS[i], True))

    acc_leaves = []

    for leaf in leaves:
        leaf.train(num_epochs_l)
        acc_leaves.append(leaf.test())

    train_sets = []
    test_sets = []
    #  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    #  train_sets.append(torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform))
    #  test_sets.append(torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform))

    for group in node_groups:
        (train_set, test_set) = prepare_dataset.prepare_dataset(group)
        train_sets.append(train_set)
        test_sets.append(test_set)

    node_PATHS = []

    for i in range(len(node_groups)):
        node_PATHS.append('./PATHS/node' + str(i + 1) + '_net.pth')

    train_sets.reverse()
    test_sets.reverse()
    node_PATHS.reverse()
    leaves.reverse()

    nodes = []

    i = 0

    for j in range(0, len(leaves), 2):
        nodes.append(Node.Node(train_sets[i], test_sets[i], leaves[j + 1], leaves[j], node_PATHS[i], False))
        i += 1

    l = 0

    for k in range(i, len(train_sets), 1):
        nodes.append(Node.Node(train_sets[k], test_sets[k], nodes[l + 1], nodes[l], node_PATHS[k], False))
        l += 2

    acc_nodes = []

    for node in nodes:
        node.train(num_epochs_n)
        acc_nodes.append(node.test())

    end = time.time()

    print(f'Seconds passed: {end - start}')
    print(f'Minutes passed: {(end - start) / 60}')
    print(f'Hours passed: {(end - start) / 3600}\n')

    return acc_leaves, acc_nodes
