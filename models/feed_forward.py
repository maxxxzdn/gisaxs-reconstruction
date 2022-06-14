import torch.distributions as dist

class ClassifierUnc(nn.Module):
    def __init__(self, n_layers):
        super().__init__()    
        self.encoder = models.resnet18(pretrained = False)
        self.loc = nn.Linear(1000, 6*n_layers)
        self.logvar = nn.Linear(1000, 6*n_layers)

    def forward(self, x):
        x = self.encoder(x)
        return self.loc(x), self.logvar(x)
    
def aleatoric_loss(y_loc, y_logvar, y_true):
    se = torch.pow((y_true - y_loc), 2)
    inv_var = torch.exp(-y_logvar)
    return 0.5*(se*inv_var + y_logvar).mean()

def train(num_epochs, classifier, loaders, loss_func):   
    classifier.train()      
    for epoch in range(num_epochs):
        for i, (x,y) in enumerate(loaders['train']):
            b_x, b_y = transformation(x), y.cuda()
            loc, logvar = classifier(b_x)
            loss = loss_func(loc, logvar, b_y)
            
            optimizer.zero_grad()           
            loss.backward()             
            optimizer.step()
            
            if i % 25 == 0:
                train_loss = test(classifier, loaders['train'], True)
                test_loss = test(classifier, loaders['val'], True)
                print('Epoch [{}/{}], Iteration [{}/{}], Train MAE: {:.4f}, Val MAE: {:.4f}'
                      .format(epoch + 1, num_epochs, i, len(loaders['train']), train_loss, test_loss))

def test(classifier, loader, light = True, loss_func = nn.L1Loss()):
    classifier.eval()
    loss = 0
    for i, (x,y) in enumerate(loader):
        with torch.no_grad():
            b_x, b_y = transformation(x), y.cuda()
            loc, logvar = classifier(b_x)
            loss += loss_func(loc, b_y)
        if light:
            break
            
    return loss.item() if light else loss.item()/len(loader) 

#encoder = ClassifierUnc(3).cuda()
#optimizer = optim.Adam(encoder.parameters(), lr = 1e-4)  
#loss_func = aleatoric_loss
#print("# parameters: {}".format(sum(p.numel() for p in encoder.parameters())))
#train(20, encoder, loaders, loss_func)

#x, y = next(iter(loaders['val']))
#b_x, b_y = transformation(x), y.cuda()
#with torch.no_grad():
#    loc, logvar = encoder(b_x)