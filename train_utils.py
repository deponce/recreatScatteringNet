import time
import sys

def TrainOneEpoch(model, criterion, train_data,optimizer, device, monitor, steps=60):
    for idx, (batch, target) in enumerate(train_data):
        start = time.time()
        batch, target = batch.to(device), target.to(device)
        outputs = model(batch)
        loss = criterion(outputs, target)
        loss.backward()
        monitor.update(model_output=outputs,
                        target=target,
                        loss=loss)
        if (idx+1)%steps==0:
            optimizer.step()
            optimizer.zero_grad()
        else:
            optimizer.virtual_step()
        end = time.time()
        if idx%1000==0:
            sys.stdout.write(str(idx+1)+' batch take '+str(end-start)+'s\n')
            sys.stdout.flush()
    optimizer.zero_grad()
    return None
