import torch
import torch.optim as optim
import torch.nn as nn
import CNN


def train(node, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 0.001
    momentum = 0.9

    net = CNN.Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    left_net = CNN.Net().to(device)
    left_net.load_state_dict(torch.load(node.left_child.PATH))

    right_net = CNN.Net().to(device)
    right_net.load_state_dict(torch.load(node.right_child.PATH))

    left = 0
    right = 0
    dcw = 0
    dcr = 0

    indices = [i for i in range(len(node.data_set))]

    for epoch in range(num_epochs):
        l_r_indices = []

        for i in indices:
            (image, label) = node.data_set.get_item(i)
            image = image.to(device)  # devo usare train loader e modificare __getitem__
            label = label.to(device)
            l_r_label = label.to(device)

            left_predicted = node.left_child.get_predicted(image)

            right_predicted = node.right_child.get_predicted(image)

            if right_predicted[0] != left_predicted[0]:

                if label[0] == left_predicted[0]:
                    l_r_label[0] = 0  # left
                    l_r_indices.append(i)
                    if epoch == 0:
                        left += 1
                elif label[0] == right_predicted[0]:
                    l_r_label[0] = 1  # right
                    l_r_indices.append(i)
                    if epoch == 0:
                        right += 1

                optimizer.zero_grad()
                output = net(image)
                loss = criterion(output, l_r_label)
                loss.backward()
                optimizer.step()
            elif epoch == 0:
                if label[0] == left_predicted[0] and label[0] == right_predicted[0]:
                    dcr += 1
                elif label[0] != left_predicted[0] and label[0] != right_predicted[0]:
                    dcw += 1
        indices = l_r_indices
        print(f'N° indices: {len(indices)}')
    print(f'N° indices: {len(indices)}')
    print(f'Finished Training {node.PATH}')
    print(f'Right: {right}')
    print(f'Left: {left}')
    print(f'DontCareWrong: {dcw}')
    print(f'DontCareRight: {dcr}')

    torch.save(net.state_dict(), node.PATH)
