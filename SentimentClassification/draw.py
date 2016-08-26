import matplotlib.pyplot as plt
import numpy as np

N = 3


ind = np.arange(N)  # the x locations for the groups
width = 0.25       # the width of the bars

fig, ax = plt.subplots()

pos_bar = (0.36, 0.32, 0.28)
neg_bar = (0.54, 0.41, 0.34) 
neu_bar = (0.08, 0.26, 0.38)
no_bar = (0.02, 0.01, 0.00)

rects1 = ax.bar(ind+.2, pos_bar, width, color='b' )
rects2 = ax.bar(ind+.2 + width*1, neg_bar, width, color='r')
rects3 = ax.bar(ind+.2 + width*2, neu_bar, width, color='y')
rects4 = ax.bar(ind+.2 + width*3, no_bar, width, color='g')
# add some text for labels, title and axes ticks
ax.set_ylabel('Percentage')
ax.set_title('Quinnipiac Polling Result, Aug 12th, 2015')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Donanld Trump', 'Jeb Bush', 'Ted Cruz'))

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Favorable', 'Unfavorable', 'Never heard', 'No opinion'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                height,
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

axes = plt.gca()

axes.set_ylim([0.0,1.0])

plt.show()