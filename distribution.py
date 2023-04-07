from matplotlib import pyplot as plt
import numpy as np


def import_group(name):
    with open(name, 'r') as txtfile:
        txt_list = list(map(float, txtfile.read().split(", ")[0:-1]))
    return txt_list


group1 = import_group("997_saved_acc.txt")
# group2 = import_group("997_saved_acc_512b.txt")

bins = np.linspace(min([min(group1)]),
                   max([max(group1)]),
                   20)

plt.hist(group1, bins, alpha=0.5, label='997p, 128b, shuf, 0.2lerelu')
# plt.hist(group2, bins, alpha=0.5, label='997p, 128b, shuf, 1e4wdu')
plt.legend(loc='upper right')


print(sum(group1)/len(group1), max(group1), len(group1))

plt.show()