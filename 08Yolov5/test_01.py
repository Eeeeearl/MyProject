import os

root = './runs/detect/images/exp'
print(len(root))
print(root[:24])

i = 1
while os.path.exists(root):
    print('已存在', root)
    root = root[:24] + str(i)
    i = i + 1

os.mkdir(root)
print('进行创建', root)








