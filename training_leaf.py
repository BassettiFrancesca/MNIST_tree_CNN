import torch
import torch.optim as optim
import torch.nn as nn
import CNN1


def train(train_set, PATH, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 4
    num_workers = 2
    learning_rate = 0.001
    momentum = 0.9

    net = CNN1.Net().to(device)

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    print(f'Finished Training {PATH}\n')

    torch.save(net.state_dict(), PATH)

