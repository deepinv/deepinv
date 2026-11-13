# config_matplotlib

### deepinv.utils.plotting.config_matplotlib(fontsize=17)

Config matplotlib for nice plots in the examples.

#### NOTE
This function mutates the global [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams). Prefer
using the plotting helpers ([`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot), [`deepinv.utils.scatter_plot()`](https://deepinv.org/api/stubs/deepinv.utils.scatter_plot.html.md#deepinv.utils.scatter_plot), etc.)
directly, which scope the style change to the function call via
[`matplotlib.pyplot.rc_context()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc_context.html#matplotlib.pyplot.rc_context) and restore the original
rcParams automatically.
