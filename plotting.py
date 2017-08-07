import matplotlib.pyplot as plt

# See where_to_move_during_calibration.py to find definition of diffs

plt.close("all")
plt.plot(diffs[:,2])     # Throw up (blue line) plot with C measurements data
plt.title('Expected C length minus measured C length')
plt.plot(diffs[:,2],'*') # Also plot some orange stars there
plt.tick_params(         # Remove the x axis tics and numbering
    axis='x',            # changes apply to the x-axis
    which='both',        # both major and minor ticks are affected
    bottom='off',        # ticks along the bottom edge are off
    top='off',           # ticks along the top edge are off
    labelbottom='off')   # labels along the bottom edge are off
ax = plt.gca()           # Grab an axis object
ax.set_ylabel('C/mm')

plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()

