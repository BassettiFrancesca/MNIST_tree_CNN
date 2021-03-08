import torch
import torchvision
import torchvision.transforms as transforms


def prepare_dataset(group):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=False, num_workers=2)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, num_workers=2)

    indices = []

    for i, (image, label) in enumerate(train_loader):
        if label[0] in group:
            indices.append(i)

    train_dataset = torch.utils.data.Subset(train_set, indices)

    print(f'Size node_dataset {group}: {len(train_dataset)}')

    test_indices = []

    for i, (image, label) in enumerate(test_loader):
        if label[0] in group:
            test_indices.append(i)

    test_dataset = torch.utils.data.Subset(test_set, test_indices)

    print(f'Size test_node_dataset {group}: {len(test_dataset)}\n')

    return train_dataset, test_dataset
