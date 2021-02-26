import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch
from ADL_HW.HW0 import data_process
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import csv

def model():
    EPOCH=3
    BATCH_SIZE=128
    label_list=[]
    x, y, lengh = data_process.data_processing()


    x=torch.from_numpy(x).type(torch.FloatTensor)
    y=torch.from_numpy(y).type(torch.FloatTensor)
    print(y.shape)
    torch_dataset=Data.TensorDataset(x, y)
    loader=Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)


    #model
    class Net(nn.Module): # Model 繼承 nn.Module
        def __init__(self, n_feature, n1_hidden, n2_hidden, n3_hidden, n_output):  # override __init__
            super(Net, self).__init__() # 使用父class的__init__()初始化網路
            self.hidden1 = nn.Linear(n_feature,n1_hidden) # layer 1
            self.hidden2 = nn.Linear(n1_hidden, n2_hidden)  # layer 1
            self.hidden3 = nn.Linear(n2_hidden, n3_hidden)  # layer 1
            self.predict= nn.Linear(n3_hidden, n_output) # output layer

            # self.relu = nn.ReLU()
        def forward(self, x):
            x = F.relu(self.hidden1(x))      # activation function for hidden layer
            x=F.relu(self.hidden2(x))
            x = F.relu(self.hidden3(x))
            x = self.predict(x)         # linear output
            out = F.sigmoid(x)
            return out
    net=Net(lengh, 20000, 1000, 100, 1).cuda()
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    loss_function = torch.nn.BCELoss()


    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(loader):
            output = net(b_x.to('cuda'))
            loss = loss_function(output, b_y.to('cuda'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                count=0

                threshold = torch.tensor([0.5]).cuda()
                pred_y = (output > threshold).float() * 1
                # print(prediction)
                # pred_y = torch.max(output, 1)[1].data.cpu().numpy()
                for i in range(BATCH_SIZE):
                    # print(pred_y[i].data.cpu().numpy())
                    # print(b_y[i].data.cpu().numpy())
                    if pred_y[i].data.cpu().numpy()==b_y[i].data.cpu().numpy():
                        count+=1
                accuracy=count/BATCH_SIZE*100
                # accuracy=float((pred_y==b_y.data.cpu().numpy()).astype(int).sum())/float(b_y.size(0))

                print(f'ephoch={epoch+1}, step={step}, loss={loss.data.item()}, acc={accuracy}%')

    # # Specify a path
    # PATH = "entire_model_weights.pt"
    # # Save
    # torch.save(net.state_dict(), PATH)
    #test

    test_x, test_y=data_process.test_data_peocessing()
    test_x=torch.from_numpy(test_x).type(torch.FloatTensor)


    test_out = net(test_x[:10000].to('cuda'))
    # for value in [0.05,0.1,0.15,0.2,0.25,0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    threshold = torch.tensor([0.5]).cuda()
    pred_y = (test_out > threshold).float() * 1
    count=0
    for i in range(10000):
        label_list.append(pred_y[i].data.cpu().numpy())
        # print("pred=", pred_y[i].data.cpu().numpy())
        # print("label=", test_y[i])
        if pred_y[i].data.cpu().numpy()==test_y[i]:
            count+=1
    test_accuracy=count/10000*100
    print(f"threshold:{0.5}, test accuracy={test_accuracy}%")
    # print("test_pred=", pred_y.data.cpu().numpy())
    # print("test_label=", test_y)
    # print("test_pred=", pred_y[1].data.cpu().numpy())
    # print("test_label=", test_y[1])



    # 寫入kaggle submission
    # with open("kaggle_sb5.csv", 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     for i in label_list:
    #         writer.writerow(str(int(i)))
    # f.close()

    return label_list
if __name__ == '__main__':
    label_list = model()  # 或是任何你想執行的函式