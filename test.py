import numpy as np
import matplotlib.pyplot as plt
import brandify

logo_file = 'sample-logo.png'
x = np.linspace(0, 6*np.pi)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(x, y)
ax.set_title('This is my sine wave', loc='left')

# Embed logo (defaults to top right of the image/figure)
fig = brandify.embed_logo(fig, logo_file)

fig.savefig('test-logo.png', bbox_inches='tight', dpi=300)
