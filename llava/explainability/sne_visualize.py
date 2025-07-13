import sys
import numpy as np
import sklearn
import random
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Random state.
RS = 20220928

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

#读入positive data
max_len = 500
vist_text = np.load(sys.argv[2])
print(vist_text.shape)
vist_text = vist_text.reshape(vist_text.shape[0],-1)
rows_id = random.sample(range(0, vist_text.shape[0]-1), max_len)
vist_text = vist_text[rows_id,:]
vist_tag = [0 for t_data in vist_text]

#读入negative data
style_text = np.load(sys.argv[3])
style_text = style_text.reshape(style_text.shape[0],-1)
rows_id = random.sample(range(0, style_text.shape[0]-1), max_len)
style_text = style_text[rows_id,:]
style_tag = [1 for t_data in style_text]
digits = [vist_text, style_text]


# We first reorder the data points according to the handwritten numbers.
#X = np.vstack([digits.data[digits.target == i]
#               for i in range(10)])
#y = np.hstack([digits.target[digits.target == i]
#               for i in range(10)])
print("load all")

X = np.vstack(digits)
y = np.hstack([vist_tag,style_tag])

print("stack all")
digits_proj = TSNE(random_state=RS).fit_transform(X)

tag_text = {0:sys.argv[2],1:sys.argv[3]}


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 2))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(2):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        t_text =tag_text[i]
        txt = ax.text(xtext, ytext, t_text, fontsize=20)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


scatter(digits_proj, y)
plt.savefig(sys.argv[1], dpi=120)
plt.show()

