for i, (data, label) in enumerate(train_loader):
    pred = model(data)
    loss = criterion(pred, label)

    loss = loss / accumulation_steps  
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()    

        optimizer.zero_grad()
# 可能会有人对这里为什么需要除以 accumulation_steps 有疑问
# 实际上是这样的，如果我们有一个大的batch的话，直接计算梯度进行更新，
