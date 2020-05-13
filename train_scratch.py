import torch
from torch import optim, nn
import torchvision
from torch.utils.data import DataLoader
# import visdom

from pokemon import Pokemon
from ResNet import resnet18

batchsz = 32
lr = 1e-3
epochs = 10

torch.manual_seed(128)

train_db = Pokemon('./data/pokemon', 224, mode='train')
val_db = Pokemon('./data/pokemon', 224, mode='val')
test_db = Pokemon('./data/pokemon', 224, mode='test')

train_loader = DataLoader(train_db, batch_size=batchsz)
val_loader = DataLoader(val_db, batch_size=batchsz)
test_loader = DataLoader(test_db, batch_size=batchsz)

# viz = visdom.Visdom()


def evalute(model, loader):
    model.eval()

    correct = 0
    total_num = len(loader.dataset)

    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().itme()

    return correct / total_num


def main():
    model = resnet18()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0

    # viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    # viz.line([0], [-1], win='acc', opts=dict(title='val_acc'))

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):

            print('step:', step)
            print()

            model.train()

            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # viz.line([loss.item()], [global_step], win='loss', update='append')
            print('step:', step, 'loss:', loss.item())
            global_step += 1

        if epoch % 1 == 0:

            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch

                torch.save(model.state_dict(), './model/best.mdl')

                # viz.line([val_acc], [global_step],
                #          win='val_acc',
                #          update='append')
            print('val_acc', val_acc, 'best_acc', best_acc)

    print('best acc:', best_acc, 'best epoch', best_epoch)

    model.load_state_dict(torch.load('./model/best.mdl'))
    print('loaded from ckpt')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)


if __name__ == "__main__":
    main()