import torch
import CNN


def test(test_set, PATHS):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = ('0', '1')

    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, num_workers=2)

    net = CNN.Net().to(device)
    net.load_state_dict(torch.load(PATHS[0]))

    left_net = CNN.Net().to(device)
    left_net.load_state_dict(torch.load(PATHS[1]))

    right_net = CNN.Net().to(device)
    right_net.load_state_dict(torch.load(PATHS[2]))

    correct = 0
    total = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
    with torch.no_grad():
        for (image, label) in test_loader:
            image = image.to(device)
            label = label.to(device)
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)
            if predicted[0] == 0:
                outputs = left_net(image)
                _, predicted = torch.max(outputs.data, 1)
            elif predicted[0] == 1:
                outputs = right_net(image)
                _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            if label[0] == predicted[0]:
                n_class_correct[label[0]] += 1
            n_class_samples[label[0]] += 1

    for i in range(2):
        acc = 100 * n_class_correct[i] / n_class_samples[i]
        print('Accuracy of %s: %.3f %%' % (classes[i], acc))

    print('Accuracy of the network on the test images: %.3f %%' % (100 * correct / total))
