import matplotlib.pyplot as plt

def set_ax_fig_style(title: str, 
                     xaxis_label: str, 
                     yaxis_label: str, 
                     ax: plt.Axes, 
                     fig: plt.Figure,
                     color: str = 'white') -> tuple[plt.Axes, plt.Figure]:
    """
    Set the style of the axes figure
    """
    fig.patch.set_alpha(0) 
    ax.set_facecolor("none")
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    ax.xaxis.set_tick_params(labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)
    ax.title.set_color(color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xaxis_label, fontsize=9)
    ax.set_ylabel(yaxis_label, fontsize=9)


    return ax, fig

def raise_for_invalid_year(year: int | list[int]) -> bool:
    valid_years = [2016, 2017, 2018, 2019]
    if isinstance(year, int) and year not in valid_years:
        print(f"Year {year} not in {valid_years}")
        return False
    if isinstance(year, list):
        if not all(y in valid_years for y in year):
            print(f"All years must be in {valid_years}")
            return False
    return True