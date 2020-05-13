import torch
from torch import optim, nn
import torchvision
from torch.utils.data import DataLoader

from torchvision.models import resnet18

from utils import Flatten
from pokemon import Pokemon

batch_sz = 32
lr = 1e-3
epochs = 10

torch.manual_seed(128)

train_db = Pokemon('./data/pokemon', 224, mode='train')
val_db = Pokemon('./data/pokemon', 224, mode='val')
test_db = Pokemon('./data/pokemon', 224, mode='test')

train_loader = DataLoader(train_db, batch_size=batch_sz, shuffle=True)
val_loader = DataLoader(val_db, batch_size=batch_sz)
test_loader = DataLoader(test_db, batch_size=batch_sz)


def evaluate(model, loader):

    model.eval()

    correct = 0
    total_num = len(loader.dataset)

    for x, y in loader:
        with torch.no_grad():
            logtis = model(x)
            pred = logtis.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total_num


def main():

    trained_model = resnet18(pretrained=True)

    model = nn.Sequential(*list(trained_model.childen())[:-1], Flatten(),
                          nn.Linear(512, 5))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntroyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0

    for epoch in epochs:

        for step, (x, y) in enumerate(train_loader):

            model.trian()
            logtis = model(x)
            loss = criteon(logtis, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        if epoch % 1 == 0:
            val_acc = evaluate(model, val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch

                torch.save(mode.state_dict(), './data/best.mdl')

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded form ckpt!')

    test_acc = evaluate(model, test_loader)
    print('test acc:', test_acc)


if __name__ == "__main__":
    main()
