import torch
import CNN1


def test(test_set, PATH):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = (test_set.groups[0][0], test_set.groups[1][0])

    batch_size = 4
    num_workers = 2

    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    net = CNN1.Net().to(device)
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(images)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

    for i in range(2):
        acc = 100 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]} in {PATH}: %.3f %%' % acc)

    print(f'Accuracy of {PATH} on the test images: %.3f %%\n' % (100 * correct / total))
    return 100 * correct / total
