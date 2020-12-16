import torch
import matplotlib.pyplot as plt

def train(model, loader_train, optimizer, loss_fn, scheduler=None, epochs=1, device='cuda:0'):
    """ Train the model """

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    loss_list = []

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

            if c % 5 == 0:
                print('Iteration {}, loss = {:.4f}, lr = {:.7f}'.format(c, loss.item(), optimizer.param_groups[0]["lr"]))
                if e != 0:
                    # Plot the loss and pearson r
                    plt.close('all')
                    plt.figure()
                    plt.plot(range(len(loss_list)), loss_list, label='train loss')
                    plt.legend()
                    plt.show()
                    # plt.savefig('/loss_graphs/epoch_%.0f_iter_%.0f.pdf' % (epoch + 1, c))

            if scheduler is not None: # Step the scheduler at each iteration
                scheduler.step()

        av_loss = total_loss / len(loader_train)
        print("     Average training loss: {0:.2f}".format(av_loss))