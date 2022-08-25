import sys
import math
import matplotlib.pyplot as plt

last_val_value = None

x = []
y1 = []
y2 = []

first = 160

for line in open(sys.argv[1], 'r').readlines():
    if 'Validation loss' in line:
        continue
    if 'EarlyStopping' in line:
        continue
    
    fields = line.split()
    if len(fields)<6:
        continue

    try:
        train_value = float(fields[3])
        val_value = float(fields[5])
    except Exception as e:
        print(f'error: {e}\n{line}')

    if last_val_value is None:
        last_val_value = val_value

    if math.isnan(val_value) or val_value==0:
        if math.isnan(last_val_value) or last_val_value==0: 
            val_value = train_value
        else:
            val_value = last_val_value
    last_val_value = val_value
    
    x.append(len(x)+1)
    y1.append(train_value)
    y2.append(val_value)

line2, = plt.plot(x[first:], y2[first:], label='validation loss')
line1, = plt.plot(x[first:], y1[first:], label='train loss')
plt.legend(handles=[line1, line2])
plt.ylabel('loss functions\' values')
plt.xlabel('epoch')

plt.show()
    
            
