import numpy as np
import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':
    # read the triplets from the file
    fp = open('/home/woody/iwso/iwso060h/outputs/cle_result/cleAll', 'r')
    rows = csv.reader(fp)
    # convert to list
    triplets = [list(map(float,l)) for l in rows]
    # separate into different lists
    args=[row[0] for row in triplets]
    top1=[row[1] for row in triplets]
    top5=[row[2] for row in triplets]
    #for row in triplets:
       # args.append(row[0])
        #top1.append(row[1])
        #top5.append(row[2])
    # convert to numpy for sorting
    args_n = np.asarray(args)
    top1_n = np.asarray(top1)
    top5_n = np.asarray(top5)
    # sort the arrays
    indices = args_n.argsort()
    args_s = args_n[indices]
    top1_s = top1_n[indices]
    top5_s = top5_n[indices]
    # plot line graph
    plt.plot(args_s, top1_s, label='top1' ,color= "green", marker='o', markerfacecolor='blue', )
    plt.plot(args_s, top5_s, label='top5', color= "red", marker='o', markerfacecolor='yellow',)
    plt.xlabel('Number of patches activated')
    plt.ylabel('Accuracy of Model')
    plt.title('CLE')
    plt.legend()
    plt.show()
    plt.savefig('cle_paths.png')
    #plot scatter graph
    
