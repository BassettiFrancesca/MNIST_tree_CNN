import torchvision
import torchvision.transforms as transforms
import GroupDataset


def divide_dataset(groups):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_d_set = GroupDataset.GroupDataset(train_set, groups)
    test_d_set = GroupDataset.GroupDataset(test_set, groups)

    return train_d_set, test_d_set
