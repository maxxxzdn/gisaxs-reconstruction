from torch import save, no_grad, clip
from torch.nn import Module, L1Loss
from torch.nn.functional import mse_loss
from numpy import mean
from math import log10
from torchmetrics.functional import structural_similarity_index_measure as ssim


class L1SSIM(Module):
    def __init__(self, kernel_size, weight = 0.5):
        super().__init__()
        self.l1 = L1Loss()
        self.weight = weight # 0 -> SSIM loss, 1 -> L1 loss, between -> both
        self.kernel_size = kernel_size
        
    def forward(self, img1, img2):
        loss = 0
        if self.weight < 1.:
            loss += (1-self.weight)*(1 - ssim(img1, img2, kernel_size = self.kernel_size))
        if self.weight > 0.:
            loss += self.weight*self.l1(img1,img2)
        return loss

def train(num_epochs, model, loaders, loss_func, optimizer, path):
    model.train()
    for epoch in range(num_epochs):
        for x,y in loaders['train']:
            loss = loss_func(model(y.cuda()), x.cuda().unsqueeze(1))   
            optimizer.zero_grad()           
            loss.backward()             
            optimizer.step()
            
        if epoch % 5 == 0:
            save(model.state_dict(), path + model.name)
            train_loss = test(loaders['train'], model, False)
            test_loss = test(loaders['val'], model, False)
            print('[Epoch [{}/{}], Train MSE (log): {:.2f}, Val MSE (log): {:.2f}'
                  .format(epoch + 1, num_epochs, log10(train_loss), log10(test_loss)))

def test(loader, model, light = True):
    model.eval()
    loss = 0
    for i, (x,y) in enumerate(loader):
        with no_grad():
            loss += mse_loss(model(y.cuda()).squeeze(), x.cuda())
            if light and i > 100: break
    return loss.item()/(i+1) if light else loss.item()/len(loader) 

def surr_loss(x, x_enc, flow, surrmod, n_samples, eps):
    bs = len(x_enc)
    y_sampled = flow.sample(n_samples, x_enc)
    x = x.unsqueeze(1).repeat(1,n_samples,1,1).reshape(bs,n_samples,-1)
    x_sm = surrmod(y_sampled.reshape(bs*n_samples,-1)).reshape(bs,n_samples,-1)
    return clip(((x - x_sm)**2).mean((1,2)) - eps, 1e-5)

def train_nf(num_epochs, eps, n_samples, alpha, flow, enc, surrmod, loaders, optimizer, path):
    for epoch in range(num_epochs):
        train_loss_nf, test_loss_nf, train_loss_sm, test_loss_sm = [], [], [], []
        for x,y in loaders['train']:
            x = x.cuda()
            y = y.cuda()
            x_enc = enc(x.unsqueeze(1))
            loss_sm = surr_loss(x, x_enc, flow, surrmod, n_samples, eps).mean()
            loss_nf = -flow.log_prob(inputs=y, context=x_enc).mean()
            loss = loss_nf + alpha*loss_sm
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_nf.append(loss_nf.item())
            train_loss_sm.append(loss_sm.item())
            
        for x,y in loaders['val']:
            x = x.cuda()
            y = y.cuda()
            x_enc = enc(x.unsqueeze(1))
            loss_sm = surr_loss(x, x_enc, flow, surrmod, n_samples, eps).mean()
            loss_nf = -flow.log_prob(inputs=y, context=x_enc).mean()
            
            test_loss_nf.append(loss_nf.item())
            test_loss_sm.append(loss_sm.item())
        
        print('Epoch: {}, NF Train Loss: {:.2f}, SM Train Loss (log): {:.2f}, '
              'NF Test Loss: {:.2f}, SM Test Loss (log): {:.2f}'.format(epoch+1, 
                                                                   mean(train_loss_nf), 
                                                                   log10(mean(train_loss_sm)),
                                                                   mean(test_loss_nf), 
                                                                   log10(mean(test_loss_sm))
                                                                  ))   
        save(flow.state_dict(), path + 'flow')
        save(enc.state_dict(), path + 'enc')