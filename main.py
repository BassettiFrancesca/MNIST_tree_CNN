import torchvision
import torchvision.transforms as transforms
import prepare_dataset
import prepare_leaf_dataset
import Node


def tree():
    train_sets = []
    test_sets = []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    for i in range(3):
        train_sets.append(torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform))
        test_sets.append(torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform))
    group1 = [0, 1, 2, 3, 4]
    group2 = [5, 6, 7, 8, 9]
    group3 = [0, 2, 4, 6, 8]
    group4 = [1, 3, 5, 7, 9]
    group5 = [0, 1]
    group6 = [2, 3, 4]
    group7 = [5, 6]
    group8 = [7, 8, 9]
    group9 = [0, 2]
    group10 = [4, 6, 8]
    group11 = [1, 3]
    group12 = [5, 7, 9]
    groups = [group1, group2, group3, group4, group5, group6, group7, group8, group9, group10, group11, group12]
    groups1 = [[0], [1]]
    groups1b = [[0], [1]]
    groups2 = [[2], [3]]
    groups3 = [[3], [4]]
    groups4 = [[5], [6]]
    groups4b = [[5], [6]]
    groups5 = [[7], [8]]
    groups6 = [[8], [9]]
    groups7 = [[0], [2]]
    groups7b = [[0], [2]]
    groups8 = [[4], [6]]
    groups9 = [[6], [8]]
    groups10 = [[1], [3]]
    groups10b = [[1], [3]]
    groups11 = [[5], [7]]
    groups12 = [[7], [9]]
    leaf_groups = [groups1, groups1b, groups2, groups3, groups4, groups4b, groups5, groups6, groups7, groups7b, groups8,
                   groups9, groups10, groups10b, groups11, groups12]
    for group in groups:
        (train_set, test_set) = prepare_dataset.prepare_dataset(group)
        train_sets.append(train_set)
        test_sets.append(test_set)

    leafs = []
    leaf_PATHS = ['./groups1_net.pth', './groups1b_net.pth', './groups2_net.pth', './groups3_net.pth',
                  './groups4_net.pth', './groups4b_net.pth', './groups5_net.pth', './groups6_net.pth',
                  './groups7_net.pth', './groups7b_net.pth', './groups8_net.pth', './groups9_net.pth',
                  './groups10_net.pth', './groups10b_net.pth', './groups11_net.pth', './groups12_net.pth']
    PATHS = ['./group0_net.pth', './group0a_net.pth', './group0b_net.pth', './group1_net.pth', './group2_net.pth',
             './group3_net.pth', './group4_net.pth', './group5_net.pth', './group6_net.pth', './group7_net.pth',
             './group8_net.pth', './group9_net.pth', './group10_net.pth', './group11_net.pth', './group12_net.pth']
    for i, leaf_group in enumerate(leaf_groups):
        (leaf_train_set, leaf_test_set) = prepare_leaf_dataset.prepare_leaf_dataset(leaf_group)
        leafs.append(Node.Node(leaf_train_set, leaf_test_set, None, None, leaf_PATHS[i], True))
    nodes = []

    nodes.append(Node.Node(train_sets[14], test_sets[14], leafs[14], leafs[15], PATHS[14], False))
    nodes.append(Node.Node(train_sets[13], test_sets[13], leafs[12], leafs[13], PATHS[13], False))
    nodes.append(Node.Node(train_sets[12], test_sets[12], leafs[10], leafs[11], PATHS[12], False))
    nodes.append(Node.Node(train_sets[11], test_sets[11], leafs[8], leafs[9], PATHS[11], False))
    nodes.append(Node.Node(train_sets[10], test_sets[10], leafs[6], leafs[7], PATHS[10], False))
    nodes.append(Node.Node(train_sets[9], test_sets[9], leafs[4], leafs[5], PATHS[9], False))
    nodes.append(Node.Node(train_sets[8], test_sets[8], leafs[2], leafs[3], PATHS[8], False))
    nodes.append(Node.Node(train_sets[7], test_sets[7], leafs[0], leafs[1], PATHS[7], False))
    nodes.append(Node.Node(train_sets[6], test_sets[6], nodes[len(nodes)-2], nodes[len(nodes)-1], PATHS[6], False))
    nodes.append(Node.Node(train_sets[5], test_sets[5], nodes[len(nodes)-4], nodes[len(nodes)-3], PATHS[5], False))
    nodes.append(Node.Node(train_sets[4], test_sets[4], nodes[len(nodes)-6], nodes[len(nodes)-5], PATHS[4], False))
    nodes.append(Node.Node(train_sets[3], test_sets[3], nodes[len(nodes)-8], nodes[len(nodes)-7], PATHS[3], False))
    nodes.append(Node.Node(train_sets[2], test_sets[2], nodes[len(nodes)-10], nodes[len(nodes)-9], PATHS[2], False))
    nodes.append(Node.Node(train_sets[1], test_sets[1], nodes[len(nodes)-12], nodes[len(nodes)-11], PATHS[1], False))
    nodes.append(Node.Node(train_sets[0], test_sets[0], nodes[len(nodes)-14], nodes[len(nodes)-13], PATHS[0], False))

    for leaf in leafs:
        leaf.train(1)
        leaf.test()

    for node in nodes:
        node.train(1)
        node.test()


if __name__ == '__main__':
    tree()
