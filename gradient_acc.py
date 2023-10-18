for i, (data, label) in enumerate(train_loader):
    pred = model(data)
    loss = criterion(pred, label)

    loss = loss / accumulation_steps  
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()    

        optimizer.zero_grad()
