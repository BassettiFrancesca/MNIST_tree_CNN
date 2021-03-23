import make_tree


def main():
    num_epochs = 2

    leaf_groups = [[[0], [5]], [[4], [6]], [[1], [2]], [[3], [7]], [[0], [6]], [[1], [4]], [[2], [7]], [[3], [5]]]

    node_groups = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 4, 5, 6], [1, 2, 3, 7], [0, 1, 4, 6], [2, 3, 5, 7]]

    make_tree.make_tree(num_epochs, leaf_groups, node_groups)


if __name__ == '__main__':
    main()
