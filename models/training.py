from torch import save, no_grad
from torch.nn.functional import mse_loss
import math


def train(num_epochs, model, loaders, loss_func, optimizer, path):
    model.train()
    for epoch in range(num_epochs):
        for x,y in loaders['train']:
            loss = loss_func(model(y.cuda()).squeeze(), x.cuda())   
            optimizer.zero_grad()           
            loss.backward()             
            optimizer.step()
            
        if epoch % 5 == 0:
            save(model.state_dict(), path + model.name)
            train_loss = test(loaders['train'], model, False)
            test_loss = test(loaders['val'], model, False)
            print('[Epoch [{}/{}], Train MSE (log): {:.2f}, Val MSE (log): {:.2f}'
                  .format(epoch + 1, num_epochs, math.log10(train_loss), math.log10(test_loss)))

def test(loader, model, light = True):
    model.eval()
    loss = 0
    for i, (x,y) in enumerate(loader):
        with no_grad():
            loss += mse_loss(model(y.cuda()).squeeze(), x.cuda())
            if light and i > 100: break
    return loss.item()/(i+1) if light else loss.item()/len(loader) 