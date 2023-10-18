# thih is a exmple of a neural network work
for x, y in train_loader:
    pred = model(x)
    loss = criterion(pred, y)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (idx+1) % eval_steps == 0:
        eval()
