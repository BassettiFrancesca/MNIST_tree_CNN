import torchvision
import torchvision.transforms as transforms
import prepare_dataset
import prepare_leaf_dataset
import Node


def tree():
    train_sets = []
    test_sets = []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_sets.append(torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform))
    test_sets.append(torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform))
    group0a = [1, 2, 4, 7]
    group0b = [0, 3, 5, 6, 8, 9]
    group1 = [2, 4, 7]
    group2 = [1, 4, 7]
    group3 = [3, 5, 8, 9]
    group4 = [0, 6, 9]
    group5 = [2, 4]
    group6 = [2, 7]
    group7 = [1, 4]
    group8 = [4, 7]
    group9 = [3, 5]
    group10 = [3, 8, 9]
    group11 = [0, 9]
    group12 = [0, 6]
    groups = [group0a, group0b, group1, group2, group3, group4, group5, group6, group7, group8, group9, group10,
              group11, group12]
    groups1 = [[2], [4]]
    groups1b = [[2], [4]]
    groups2 = [[2], [7]]
    groups3 = [[2], [7]]
    groups4 = [[1], [4]]
    groups4b = [[1], [4]]
    groups5 = [[4], [7]]
    groups6 = [[4], [7]]
    groups7 = [[3], [5]]
    groups7b = [[3], [5]]
    groups8 = [[3], [8]]
    groups9 = [[8], [9]]
    groups10 = [[0], [9]]
    groups10b = [[0], [9]]
    groups11 = [[0], [6]]
    groups12 = [[0], [6]]
    leaf_groups = [groups1, groups1b, groups2, groups3, groups4, groups4b, groups5, groups6, groups7, groups7b, groups8,
                   groups9, groups10, groups10b, groups11, groups12]
    for group in groups:
        (train_set, test_set) = prepare_dataset.prepare_dataset(group)
        train_sets.append(train_set)
        test_sets.append(test_set)

    leaf_PATHS = ['./groups1_net.pth', './groups1b_net.pth', './groups2_net.pth', './groups3_net.pth',
                  './groups4_net.pth', './groups4b_net.pth', './groups5_net.pth', './groups6_net.pth',
                  './groups7_net.pth', './groups7b_net.pth', './groups8_net.pth', './groups9_net.pth',
                  './groups10_net.pth', './groups10b_net.pth', './groups11_net.pth', './groups12_net.pth']
    PATHS = ['./group0_net.pth', './group0a_net.pth', './group0b_net.pth', './group1_net.pth', './group2_net.pth',
             './group3_net.pth', './group4_net.pth', './group5_net.pth', './group6_net.pth', './group7_net.pth',
             './group8_net.pth', './group9_net.pth', './group10_net.pth', './group11_net.pth', './group12_net.pth']

    leafs = []

    for i, leaf_group in enumerate(leaf_groups):
        (leaf_train_set, leaf_test_set) = prepare_leaf_dataset.prepare_leaf_dataset(leaf_group)
        leafs.append(Node.Node(leaf_train_set, leaf_test_set, None, None, leaf_PATHS[i], True))

    for leaf in leafs:
        leaf.train(2)
        leaf.test()

    node12 = Node.Node(train_sets[14], test_sets[14], leafs[14], leafs[15], PATHS[14], False)
    node12.train(4)
    node12.test()
    node11 = Node.Node(train_sets[13], test_sets[13], leafs[12], leafs[13], PATHS[13], False)
    node11.train(4)
    node11.test()
    node10 = Node.Node(train_sets[12], test_sets[12], leafs[10], leafs[11], PATHS[12], False)
    node10.train(4)
    node10.test()
    node9 = Node.Node(train_sets[11], test_sets[11], leafs[8], leafs[9], PATHS[11], False)
    node9.train(4)
    node9.test()
    node8 = Node.Node(train_sets[10], test_sets[10], leafs[6], leafs[7], PATHS[10], False)
    node8.train(4)
    node8.test()
    node7 = Node.Node(train_sets[9], test_sets[9], leafs[4], leafs[5], PATHS[9], False)
    node7.train(4)
    node7.test()
    node6 = Node.Node(train_sets[8], test_sets[8], leafs[2], leafs[3], PATHS[8], False)
    node6.train(4)
    node6.test()
    node5 = Node.Node(train_sets[7], test_sets[7], leafs[0], leafs[1], PATHS[7], False)
    node5.train(4)
    node5.test()
    node4 = Node.Node(train_sets[6], test_sets[6], node11, node12, PATHS[6], False)
    node4.train(4)
    node4.test()
    node3 = Node.Node(train_sets[5], test_sets[5], node9, node10, PATHS[5], False)
    node3.train(4)
    node3.test()
    node2 = Node.Node(train_sets[4], test_sets[4], node7, node8, PATHS[4], False)
    node2.train(4)
    node2.test()
    node1 = Node.Node(train_sets[3], test_sets[3], node5, node6, PATHS[3], False)
    node1.train(4)
    node1.test()
    node0b = Node.Node(train_sets[2], test_sets[2], node3, node4, PATHS[2], False)
    node0b.train(4)
    node0b.test()
    node0a = Node.Node(train_sets[1], test_sets[1], node1, node2, PATHS[1], False)
    node0a.train(4)
    node0a.test()
    node0 = Node.Node(train_sets[0], test_sets[0], node0a, node0b, PATHS[0], False)
    node0.train(4)
    node0.test()


if __name__ == '__main__':
    tree()
