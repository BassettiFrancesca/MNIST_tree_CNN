import make_tree


def main():
    num_epochs_l = 2

    num_epochs_n = 4

    leaf_groups = [[[1], [2]], [[0], [2]], [[0], [8]], [[0], [9]], [[3], [4]], [[4], [5]], [[6], [7]], [[4], [6]]]

    node_groups = [[0, 1, 2, 8, 9], [3, 4, 5, 6, 7], [0, 1, 2], [0, 8, 9], [3, 4, 5], [4, 6, 7]]

    acc_leaves = []
    acc_nodes_h = []
    acc_nodes_l = []
    acc_trees = []

    for i in range(3):
        (a_l, a_n) = make_tree.make_tree(num_epochs_l, num_epochs_n, leaf_groups, node_groups)
        acc_trees.append(a_n[len(a_n) - 1])
        for j in range(4, 6, 1):
            acc_nodes_h.append(a_n[j])
        for k in range(4):
            acc_nodes_l.append(a_n[k])
        for l in a_l:
            acc_leaves.append(l)

    print(f'Accuracy of the leaves with {num_epochs_l} epochs: {acc_leaves}')
    total = 0
    for i in acc_leaves:
        total += i
    mean_l = total / len(acc_leaves)
    print(f'Mean: %.2f\n' % mean_l)

    print(f'Accuracy of the lower nodes with {num_epochs_n} epochs: {acc_nodes_l}')
    total = 0
    for i in acc_nodes_l:
        total += i
    mean_n = total / len(acc_nodes_l)
    print(f'Mean: %.2f\n' % mean_n)

    print(f'Accuracy of the higher nodes with {num_epochs_n} epochs: {acc_nodes_h}')
    total = 0
    for i in acc_nodes_h:
        total += i
    mean_n = total / len(acc_nodes_h)
    print(f'Mean: %.2f\n' % mean_n)

    print(f'Accuracy of the trees: {acc_trees}')
    total = 0
    for i in acc_trees:
        total += i
    mean_t = total / len(acc_trees)
    print(f'Mean: %.2f\n' % mean_t)


if __name__ == '__main__':
    main()
