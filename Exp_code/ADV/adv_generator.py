import argparse
import torch
import torchvision
from torchvision import transforms
from train_model import *
import attack_generator as attack

def adv_generator(
    perturb_steps=20,
    epsilon=8./255,
    step_size=8./255 / 10,
    loss_fn="cent",
    category="Madry",
    net="resnet18",
    dataset="cifar10",
    model_path="./Res18_model/net_150.pth",
    device="cuda",
):
    
    transform_test = transforms.Compose([transforms.ToTensor(),])
    print('==> Load Test Data')

    if dataset == "cifar10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    if dataset == "svhn":
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    print('==> Load Model')
    if net == "resnet18":
        model = ResNet18().to(device)
    elif net == "resnet34":
        model = ResNet34().to(device)
    else:
        raise ValueError("net must be either 'resnet18' or 'resnet34'.")

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)

    print(net)

    model.eval()
    
    print('==> Generate adversarial sample')
    X_adv = attack.adv_generate(model, test_loader, perturb_steps, epsilon, step_size,loss_fn, category, rand_init=True)
    return X_adv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
    parser.add_argument('--net', type=str, default="resnet18", help="choose from resnet18, resnet34")
    parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10, svhn")
    parser.add_argument('--model_path', default='./Res18_model/net_150.pth', help='model for white-box attack evaluation')
    parser.add_argument('--perturb_steps', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=8./255)
    parser.add_argument('--step_size', type=float, default=8./255 / 10)
    parser.add_argument('--loss_fn', default="cent")
    parser.add_argument('--category', default="Madry")
    parser.add_argument('--device', default="cuda")
    args = parser.parse_args()

    adv_generator(
        perturb_steps=args.perturb_steps,
        epsilon=args.epsilon,
        step_size=args.step_size,
        loss_fn=args.loss_fn,
        category=args.category,
        net=args.net,
        dataset=args.dataset,
        model_path=args.model_path,
        device=args.device,
    )
