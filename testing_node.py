import torch
import CNN


def test(node):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = torch.utils.data.DataLoader(node.test_set, shuffle=False, num_workers=2)

    net = CNN.Net().to(device)
    net.load_state_dict(torch.load(node.PATH))

    left_net = CNN.Net().to(device)
    left_net.load_state_dict(torch.load(node.left_child.PATH))

    right_net = CNN.Net().to(device)
    right_net.load_state_dict(torch.load(node.right_child.PATH))

    correct = 0
    total = 0

    with torch.no_grad():
        for (image, label) in test_loader:
            image = image.to(device)
            label = label.to(device)
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)
            if predicted[0] == 0:
                leaf_predicted = node.left_child.get_predicted(image)
            elif predicted[0] == 1:
                leaf_predicted = node.right_child.get_predicted(image)
            total += label.size(0)
            correct += (leaf_predicted == label).sum().item()

    print('Accuracy of the network on the test images: %.3f %%' % (100 * correct / total))
