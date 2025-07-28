if __name__ == "__main__":
    print("Loading CIFAR10 dataset...")
    full_trainset = CIFAR10(root='./data', train=True, download=True)
    paritions = partition(full_trainset, 10, 0.1)

    client_trainloader, client_testloader = load_dataset(paritions, 0, 64, 100, './data')

    print("CIFAR10 dataset loaded and partitioned.")