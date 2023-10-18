for i, (data, label) in enumerate(train_loader):
    pred = model(data)
    loss = criterion(pred, label)

    loss = loss / accumulation_steps  # 损失标准化，为了梯度稳定
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()    

        optimizer.zero_grad()
# 需要注意，需要适当扩大学习率
