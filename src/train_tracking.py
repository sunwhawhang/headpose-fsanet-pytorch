import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

def train(model, loader_train, optimizer, loss_fn, scheduler=None, epochs=1, device='cuda:0', loader_val=None):
    """ Train the model """

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    loss_list = []
    loss_val_list = []
    val_count = []

    iter_count = 0

    for e in range(epochs):
        print('======= Epoch {:} / {:} ======='.format(e + 1, epochs))
        total_loss = 0

        for c, (inputs, label) in enumerate(loader_train):

            model.train()  # put model to training mode

            inputs = inputs.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            out = model(inputs)

            loss = loss_fn(out, label)
            total_loss += loss.item()
            loss_list.append(loss.item())
            loss.backward()

            # Prevent gradient from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            print_step = 5
            if c % print_step == 0:
                loss_val = eval(model, loader_val, loss_fn) if loader_val else None
                if loss_val:
                    loss_val_list.append(loss_val)
                    val_count.append(iter_count)

                print('Iteration {}, loss = {:.4f}, lr = {:.7f}, loss_val = {}'.format(c, loss.item(), optimizer.param_groups[0]["lr"], loss_val))
                # if e != 0 and c %100 == 0:
                if False:
                    # Plot the loss and pearson r
                    plt.close('all')
                    plt.figure()
                    plt.plot(range(len(loss_list)), loss_list, label='train loss')
                    plt.legend()
                    plt.show()
                    # plt.savefig('/loss_graphs/epoch_%.0f_iter_%.0f.pdf' % (epoch + 1, c))

            if scheduler is not None: # Step the scheduler at each iteration
                scheduler.step()
            iter_count += 1

        av_loss = total_loss / len(loader_train)
        print("     Average training loss: {0:.2f}".format(av_loss))

    plt.close('all')
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list, label='train loss')
    plt.plot(val_count, loss_val_list, label='val loss')
    plt.legend()
    plt.show()

def eval(model, loader_val, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = []
        labels = []
        for inputs, label in loader_val:
            logits.append(model(inputs).detach().cpu())
            labels.append(label.cpu())
        logits = torch.cat(logits)
        labels = torch.cat(labels)
        return loss_fn(logits, labels)

def check_accuracy(model, loader_test, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        softmax = []
        labels = []
        for inputs, label in loader_test:
            inputs = inputs.to(device)
            softmax.append(model(inputs, softmax=True).detach().cpu())
            labels.append(label.cpu())
        
        softmax = torch.cat(softmax)
        labels = torch.cat(labels)
        acc = accuracy(softmax, labels)
        print("The accuracy is %s" %acc)
        return acc

def accuracy(softmax, labels):
    print("Test data stats: ", Counter(labels.tolist()))
    preds = softmax.argmax(dim=-1)
    print(confusion_matrix(labels.tolist(), preds.tolist()))

    corrects = (preds == labels)
    return corrects.sum().float() / float(labels.size(0))