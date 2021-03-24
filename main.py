import make_tree


def main():
    num_epochs_l = 4

    num_epochs_n = 4

    leaf_groups = [[[2], [3]], [[2], [3]]]

    node_groups = [[2, 3]]

    acc_leaves = []
    acc_nodes = []
    acc_trees = []

    for i in range(10):
        (a_l, a_n) = make_tree.make_tree(num_epochs_l, num_epochs_n, leaf_groups, node_groups)
        acc_trees.append(a_n[len(a_n) - 1])
        for j in a_n:
            acc_nodes.append(j)
        for k in a_l:
            acc_leaves.append(k)

    print(f'Accuracy of the leaves with {num_epochs_l} epochs: {acc_leaves}')
    total = 0
    for i in acc_leaves:
        total += i
    mean = total / len(acc_leaves)
    print(f'Mean: {mean}\n')

    print(f'Accuracy of the nodes with {num_epochs_n} epochs: {acc_nodes}')
    total = 0
    for i in acc_nodes:
        total += i
    mean = total / len(acc_nodes)
    print(f'Mean: {mean}\n')

    print(f'Accuracy of the trees: {acc_trees}')
    total = 0
    for i in acc_trees:
        total += i
    mean = total / len(acc_trees)
    print(f'Mean: {mean}\n')


if __name__ == '__main__':
    main()
