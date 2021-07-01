import torch
import torch.optim as optim
import torch.nn as nn
import CNN1


def train(node, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 0.001
    momentum = 0.9

    net = CNN1.Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    left = 0
    right = 0
    dcw = 0
    dcr = 0

    for epoch in range(num_epochs):
        l_r_indices = []

        train_loader = torch.utils.data.DataLoader(node.data_set, shuffle=False, num_workers=2)

        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            left_predicted = node.left_child.get_predicted(image)

            right_predicted = node.right_child.get_predicted(image)

            if label[0] == left_predicted[0] or label[0] == right_predicted[0]:

                if right_predicted[0] != left_predicted[0]:

                    if label[0] == left_predicted[0]:
                        l_r_label = 0  # left
                        l_r_indices.append(i)
                        if epoch == 0:
                            left += 1
                    elif label[0] == right_predicted[0]:
                        l_r_label = 1  # right
                        l_r_indices.append(i)
                        if epoch == 0:
                            right += 1

                    label[0] = l_r_label

                    optimizer.zero_grad()
                    output = net(image)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()

                elif epoch == 0:
                    dcr += 1
            elif epoch == 0:
                if label[0] != left_predicted[0] and label[0] != right_predicted[0]:
                    dcw += 1
        node.data_set = torch.utils.data.Subset(node.data_set, l_r_indices)

    print(f'Finished Training {node.PATH}')
    print(f'N° indices: {len(l_r_indices)}')
    print(f'Right: {right}')
    print(f'Left: {left}')
    print(f'DontCareWrong: {dcw}')
    print(f'DontCareRight: {dcr}\n')

    torch.save(net.state_dict(), node.PATH)
