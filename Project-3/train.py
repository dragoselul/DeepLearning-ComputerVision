import torch


def train(model, train_loader, loss_fn, opt, device, save_name, epochs=20):
    # Training loop
  
    model.train()  # train mode
    for epoch in range(epochs):
        print(f'* Epoch {epoch+1}/{epochs}')

        avg_loss = 0
        for X_batch, y_true in train_loader:
            X_batch = X_batch.to(device)
            y_true = y_true.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            y_pred = model(X_batch)
            # IMPORTANT NOTE: Check whether y_pred is normalized or unnormalized
            # and whether it makes sense to apply sigmoid or softmax.
            loss = loss_fn(y_pred, y_true)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)

        # IMPORTANT NOTE: It is a good practice to check performance on a
        # validation set after each epoch.
        #model.eval()  # testing mode
        #Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
        print(f' - loss: {avg_loss}')

    # Save the model
    torch.save(model, save_name)
    print("Training has finished!")
    return save_name
