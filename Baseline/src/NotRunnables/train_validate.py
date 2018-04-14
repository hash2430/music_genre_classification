import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler

# train / eval
def fit(model,train_loader,valid_loader,criterion,learning_rate,num_epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        acc = []

        for i, data in enumerate(train_loader):
            audio = data['mel']
            label = data['label']
            # have to convert to an autograd.Variable type in order to keep track of the gradient...
            audio = Variable(audio).type(torch.FloatTensor)
            label = Variable(label).type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(audio)

            #print(outputs,label)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print ("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (epoch+1, num_epochs, i+1, len(train_loader), loss.data[0]))
        # After learning each epoch, try evaluating the model with validation set
        # if evaluation accuracy suggests too small learning rate for next epoch, stop learning
        eval_loss, _ , _ = eval(model, valid_loader, criterion)
        scheduler.step(eval_loss) # use the learning rate scheduler
        curr_lr = optimizer.param_groups[0]['lr']
        print('Learning rate : {}'.format(curr_lr))
        if curr_lr < 1e-5:
            print ("Early stopping\n\n")
            break

def eval(model,valid_loader,criterion):

    eval_loss = 0.0
    output_all = []
    label_all = []

    for i, data in enumerate(valid_loader):
        model.eval()
        audio = data['mel']
        label = data['label']
        # have to convert to an autograd.Variable type in order to keep track of the gradient...
        audio = Variable(audio).type(torch.FloatTensor)
        label = Variable(label).type(torch.LongTensor)
        outputs = model(audio)
        loss = criterion(outputs, label)

        eval_loss += loss.data[0]

        output_all.append(outputs.data.numpy())
        label_all.append(label.data.numpy())

    avg_loss = eval_loss/len(valid_loader)
    print ('Average loss: {:.4f} \n'. format(avg_loss))

    return avg_loss, output_all, label_all