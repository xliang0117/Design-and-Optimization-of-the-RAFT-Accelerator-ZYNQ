# from decimal import DecimalTuple
# import torch
# import torch.nn as nn
# import torch.nn.functional as F



# if __name__ == "__main__":
#     inp = torch.arange(0, 32)
#     inp = inp.reshape((2, 1, 4, 4)).float()
#     print(inp)
#     out_h = 3
#     out_w = 3
#     new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
#     new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
#     print(new_h)
#     print(new_w)
#     grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim=2)
#     grid = grid.unsqueeze(0)
#     temp_grid = torch.zeros((1,3,3,2), dtype=torch.float32)
#     grid = torch.cat((grid, temp_grid), dim=0)
#     outp = F.grid_sample(inp, grid=grid, mode='bilinear', align_corners=True)
#     print(outp) 

# if __name__ == "__main__":
#     pthfile = r'../quantize_raft/checkpoints/5000_raft-chairs.pth'            #.pth文件的路径
#     model = torch.load(pthfile, torch.device('cpu'))    #设置在cpu环境下查询
#     print('type:')
#     print(type(model))  #查看模型字典长度
#     print('length:')
#     print(len(model))
#     print('key:')
#     for k in model.keys():  #查看模型字典里面的key
#         print(k)
#     # print('value:')
#     # for k in model:         #查看模型字典里面的value
#     #     print(k,model[k])
#     print(model["module.fnet.conv1.weight"])

import numpy as np
dataset = ""
for iter in range(18):
    flowPR_1 = np.load("hist/flow_dir_pr{}_{}.npy".format( dataset, iter))
    flowPR_1 = np.sum(flowPR_1, axis=0)
    flowPR_1 = flowPR_1 / np.sum(flowPR_1)
    # print(flowPR_1[2,2])

    flowPR_2 = np.load("hist/flow_dir_pr{}_{}.npy".format(dataset, iter+1))
    flowPR_2 = np.sum(flowPR_2, axis=0)
    flowPR_2 = flowPR_2 / np.sum(flowPR_2)
    # print(flowPR_2[2,2])


    flowPRSquare = np.load("hist/flow_dir_pr_square{}_{}.npy".format(dataset, iter))
    pointNum = np.sum(flowPRSquare)
    flowPRSquare = np.sum(flowPRSquare, axis=0)
    flowPRSquare = flowPRSquare / np.sum(flowPRSquare)
    if flowPR_2[2,2] < flowPRSquare[2,2]:
        print(flowPRSquare)

    centerPRSquare = np.load("hist/center_pr_square{}_{}.npy".format(dataset, iter))
    centerPRSquare = np.average(centerPRSquare, axis=0)  / flowPR_1[2,2]

    comeAndStayPR = np.load("hist/come_and_stay{}_{}.npy".format(dataset, iter))
    comeAndStayPR = np.average(comeAndStayPR, axis=0) / (1 - flowPR_1[2,2])

    print("iter:{:d}".format(iter))
    print("a1:{:.3f}".format(flowPR_1[2,2]))
    print("a2:{:.3f}".format(flowPR_2[2,2]))
    print("2 step keep:{:.3f}".format(flowPRSquare[2,2]))
    print("stay and stay:{:.3f}".format(centerPRSquare))
    print("stay and leave:{:.3f}".format(1-centerPRSquare))
    print("come and stay:{:.3f}".format(comeAndStayPR))
    print("come and back:{:.3f}".format((flowPRSquare[2,2]-centerPRSquare*flowPR_1[2,2]) / (1-flowPR_1[2,2])) )
    print("come and leave:{:.3f}".format(1 - comeAndStayPR - (flowPRSquare[2,2]-centerPRSquare*flowPR_1[2,2]) / (1-flowPR_1[2,2])) )
    print(" ")