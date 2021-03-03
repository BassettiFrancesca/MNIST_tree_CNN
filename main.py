import torchvision
import torchvision.transforms as transforms
import prepare_dataset
import prepare_leaf_dataset
import Node


def tree():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set1 = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set1 = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    groups2 = [0, 1, 2, 3, 4]
    (train_set2, test_set2) = prepare_dataset.prepare_dataset(groups2)
    groups3 = [5, 6, 7, 8, 9]
    (train_set3, test_set3) = prepare_dataset.prepare_dataset(groups3)
    groups4 = [[0], [1]]
    (train_set4, test_set4) = prepare_leaf_dataset.prepare_leaf_dataset(groups4)
    groups5 = [[2], [3]]
    (train_set5, test_set5) = prepare_leaf_dataset.prepare_leaf_dataset(groups5)
    groups6 = [[3], [4]]
    (train_set6, test_set6) = prepare_leaf_dataset.prepare_leaf_dataset(groups6)
    groups7 = [[5], [6]]
    (train_set7, test_set7) = prepare_leaf_dataset.prepare_leaf_dataset(groups7)
    groups8 = [[7], [8]]
    (train_set8, test_set8) = prepare_leaf_dataset.prepare_leaf_dataset(groups8)
    groups9 = [[8], [9]]
    (train_set9, test_set9) = prepare_leaf_dataset.prepare_leaf_dataset(groups9)
    groups10 = [2, 3, 4]
    (train_set10, test_set10) = prepare_dataset.prepare_dataset(groups10)
    groups11 = [7, 8, 9]
    (train_set11, test_set11) = prepare_dataset.prepare_dataset(groups11)
    groups12 = [0, 1]
    (train_set12, test_set12) = prepare_dataset.prepare_dataset(groups12)
    groups13 = [5, 6]
    (train_set13, test_set13) = prepare_dataset.prepare_dataset(groups13)
    leaf01a = Node.Node(train_set4, test_set4, 0, 0, './leaf01a_net.pth', True)
    leaf01b = Node.Node(train_set4, test_set4, 0, 0, './leaf01b_net.pth', True)
    leaf23 = Node.Node(train_set5, test_set5, 0, 0, './leaf23_net.pth', True)
    leaf34 = Node.Node(train_set6, test_set6, 0, 0, './leaf34_net.pth', True)
    leaf56a = Node.Node(train_set7, test_set7, 0, 0, './leaf56a_net.pth', True)
    leaf56b = Node.Node(train_set7, test_set7, 0, 0, './leaf56b_net.pth', True)
    leaf78 = Node.Node(train_set8, test_set8, 0, 0, './leaf78_net.pth', True)
    leaf89 = Node.Node(train_set9, test_set9, 0, 0, './leaf89_net.pth', True)
    leaf01a.train(1)
    leaf01b.train(1)
    leaf23.train(1)
    leaf34.train(1)
    leaf56a.train(1)
    leaf56b.train(1)
    leaf78.train(1)
    leaf89.train(1)
    node7 = Node.Node(train_set11, test_set11, leaf78, leaf89, './node7_net.pth', False)
    node7.train(1)
    node6 = Node.Node(train_set13, test_set13, leaf56a, leaf56b, './node6_net.pth', False)
    node6.train(1)
    node5 = Node.Node(train_set10, test_set10, leaf23, leaf34, './node5_net.pth', False)
    node5.train(1)
    node4 = Node.Node(train_set12, test_set12, leaf01a, leaf01b, './node4_net.pth', False)
    node4.train(1)
    node3 = Node.Node(train_set3, test_set3, node6, node7, './node3_net.pth', False)
    node3.train(1)
    node2 = Node.Node(train_set2, test_set2, node4, node5, './node2_net.pth', False)
    node2.train(1)
    node1 = Node.Node(train_set1, test_set1, node2, node3, './node1_net.pth', False)
    node1.train(1)
    node1.test()


if __name__ == '__main__':
    tree()
