import matplotlib.pyplot as plt
import numpy as np

## OLAF LEFT 2016: "Save room for scrumptious desserts."
plt.rcdefaults()

fig, ax = plt.subplots()

components = {'Right Context':'red', 'Left Context':'blue'}
labels = list(components.keys())
handles = [plt.Rectangle((0,0),1,1, color=components[label]) for label in labels]
plt.legend(handles, labels, loc='lower right')
plt.axvline(x=0, color='black')
words = ('Save', 'room', 'for', 'scrumptious', '.')
y_pos = np.arange(len(words))
influences = [1.24, 0.713, 0.348, 0.335, 0.004]

ax.barh(y_pos, influences, align='center', color=['blue', 'blue', 'blue', 'blue', 'red'])
ax.set_yticks(y_pos)
ax.set_yticklabels(words)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Influence')
ax.set_title('Predicted Label: positive \n True Label: positive')
plt.savefig('Olaf2016left.png', bbox_inches='tight')
plt.show()
'''
## OLAF RIGHT 2016: "Overall I would go back and eat at the restaurant again."
plt.rcdefaults()

fig, ax = plt.subplots()

components = {'Right Context':'red', 'Left Context':'blue'}
labels = list(components.keys())
handles = [plt.Rectangle((0,0),1,1, color=components[label]) for label in labels]
plt.legend(handles, labels, loc='lower right')
plt.axvline(x=0, color='black')
words = ('again', '.', 'I', 'and', 'go', 'back', 'Overall')
y_pos = np.arange(len(words))
influences = [0.757, 0.656, 0.104, 0.076, 0.065, 0.054, -1.162]

ax.barh(y_pos, influences, align='center', color=['red', 'red', 'blue', 'blue', 'blue', 'blue', 'blue'])
ax.set_yticks(y_pos)
ax.set_yticklabels(words)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Influence')
ax.set_title('Predicted Label: Positive \n True Label: Positive')
#
plt.savefig('Olaf2016right.png', bbox_inches='tight')
plt.show()'''