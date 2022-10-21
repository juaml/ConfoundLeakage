import numpy as np
from numpy.typing import NDArray
from itertools import product
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from dtreeviz import trees
from sklearn import tree
from baycomp import two_on_single


def save_trees(model, base_save_dir, problem_type, data, X, target):

    if problem_type == 'regression':
        fig = plt.figure()
        tree.plot_tree(model['decisiontreeregressor'])
        fig.savefig(base_save_dir + '_tree.svg', bbox_inches="tight")
        joblib.dump(fig, base_save_dir + '_tree.joblib')
        plt.close('all')
        fig = plt.figure()
        trees.rtreeviz_bivar_heatmap(model['decisiontreeregressor'],
                                     data[X], data[target],
                                     target_name=target,
                                     feature_names=X,
                                     )
        fig.savefig(base_save_dir + '_heatmap.svg', bbox_inches="tight")
        joblib.dump(fig, base_save_dir + '_heatmap.joblib')
        plt.close('all')
    else:
        fig = plt.figure()
        tree.plot_tree(model['decisiontreeclassifier'])
        fig.savefig(base_save_dir + '_tree.svg', bbox_inches="tight")
        joblib.dump(fig, base_save_dir + '_tree.joblib')
        plt.close('all')
        fig = plt.figure()
        trees.ctreeviz_bivar(model['decisiontreeclassifier'],
                             data[X], data[target],
                             target_name=target,
                             feature_names=X,
                             )

        fig.savefig(base_save_dir + '_heatmap.svg', bbox_inches="tight")
        joblib.dump(fig, base_save_dir + '_heatmap.joblib')
        plt.close('all')

# %%


# def corrected_std(vals, n_train, n_test):
#     """This functions is copied from a scikit-learn tutorial.
#     See: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html
#     #sphx-glr-auto-examples-model-selection-plot-grid-search-stats-py
#     Follow their rules for distribution.
#
#
#     Corrects standard deviation using Nadeau and Bengio's approach.
#
#     Parameters
#     ----------
#     differences : ndarray of shape (n_samples,)
#         Vector containing the differences in the score metrics of two models.
#     n_train : int
#         Number of samples in the training set.
#     n_test : int
#         Number of samples in the testing set.
#
#     Returns
#     -------
#     corrected_std : float
#         Variance-corrected standard deviation of the set of differences.
#     """
#     # kr = k times r, r times repeated k-fold crossvalidation,
#     # kr equals the number of times the model was evaluated
#     kr = len(vals)
#
#     corrected_var = np.var(vals, ddof=1) * (1 / kr + n_test / n_train)
#     corrected_std = np.sqrt(corrected_var)
#     return corrected_std


def get_rope(x: NDArray, y: NDArray,
             repeat: int, rope: float = .05) -> str:
    probs = two_on_single(x, y, runs=repeat, rope=rope)
    return [">", "=", "<"][np.argmax(probs)]


def hline(x0_position, x1_position, sign, ax,
          sign_fontsize=12, y_sign=1.03, linewidth=1):

    height = 0.01
    y_min = y_sign - height
    y_text = y_sign + .04

    ax.plot([x0_position, x1_position], [y_sign, y_sign], color='black')
    ax.plot([x0_position, x0_position], [y_min, y_sign], color='black')
    ax.plot([x1_position, x1_position], [y_min, y_sign], color='black')
    tick = x0_position + (x1_position - x0_position)/2
    ax.text(tick, y_text, sign, fontsize=sign_fontsize,
            verticalalignment='center', horizontalalignment='center'
            )


def custom_bar_rope_plot(
        x, y, hue,  cv_repeats, data,
        hue_order, comparisons, comparisons_sing_y=None,  order=None, rope=0.05,
        rope_sign_fontsize=12, show_legend=True,
        ax=None, palette=None,
        rope_line_width=.5, **kwargs):

    df_plot = data.copy()

    if ax is None:
        f = plt.figure()
        ax = plt.gca()
    else:
        f = plt.gcf()

    assert(len(hue_order) == len(df_plot[hue].unique()))
    assert isinstance(comparisons, tuple)
    assert all([(isinstance(comp, tuple) and len(comp) == 2)
                for comp in comparisons])
    # check repeats are always 10
    all_n_repeats = df_plot.groupby([x, hue])[cv_repeats].nunique().values
    n_cv_repeats = all_n_repeats[0]
    assert (all_n_repeats == n_cv_repeats).all()

    x_order = data[x].unique() if order is None else order
    comparisons_sing_y = (np.linspace(1, 1.1, len(comparisons))
                          if comparisons_sing_y is None else
                          comparisons_sing_y
                          )
    assert len(comparisons_sing_y) == len(comparisons)
    comparisons_sing_y = {comparison: y_sign
                          for comparison, y_sign in zip(comparisons, comparisons_sing_y)
                          }

    df_comparisons = []
    for comparison, x_val in product(comparisons, x_order):
        idx_comp_0 = df_plot.query(f"{x}==@x_val & ({hue}==@comparison[0]) ").index
        idx_comp_1 = df_plot.query(f"{x}==@x_val &  ({hue}==@comparison[1])").index
        sign = get_rope(
            x=df_plot.loc[idx_comp_0, y].values,
            y=df_plot.loc[idx_comp_1, y].values,
            repeat=n_cv_repeats, rope=rope
        )
        df = pd.DataFrame({"comparison": [comparison],
                           x: [x_val], "sign": [sign],
                           "hue_0": comparison[0],
                           "hue_1": comparison[1],
                           })

        df_comparisons.append(df)
    df_comparisons = pd.concat(df_comparisons).reset_index(drop=True)

    sns.barplot(x=x, y=y, hue=hue, ci=None, data=df_plot, ax=ax,
                order=x_order, hue_order=hue_order, palette=palette,
                **kwargs)

    bar_positions = ax.patches
    assert len(bar_positions) == (len(x_order) * len(hue_order))

    x_positions = [rect.get_x() + rect.get_width() / 2
                   for rect in bar_positions]

    df_bars = (df_plot.copy()
               .assign(**{
                   x: lambda df: pd.Categorical(df[x], categories=x_order),
                   hue: lambda df: pd.Categorical(df[hue], categories=hue_order)
               })
               .groupby([x, hue, cv_repeats])[y].mean()
               .groupby([x, hue]).agg(["mean", "std"])
               .reset_index()
               .sort_values(by=[hue, x], ignore_index=True)
               .assign(x_positions=x_positions)
               )
    ax.errorbar(x=df_bars.x_positions,
                y=df_bars["mean"],
                yerr=df_bars["std"],
                linestyle="", color="k")
    df_comparisons = (df_comparisons
                      .assign(
                          x0_position=lambda df: df.apply(lambda row:  df_bars.query(
                              f"({x} == @row['{x}']) & ({hue} == @row['hue_0'])"
                          )["x_positions"].values[0], axis=1),
                          x1_position=lambda df: df.apply(lambda row: df_bars.query(
                              f"({x} == @row['{x}']) & ({hue} == @row['hue_1'])"
                          )["x_positions"].values[0], axis=1),
                      )
                      )
    for row in df_comparisons.itertuples():
        hline(row.x0_position, row.x1_position, row.sign, ax=ax,
              y_sign=comparisons_sing_y[row.comparison],
              sign_fontsize=rope_sign_fontsize,
              linewidth=rope_line_width

              )

    if not show_legend:
        ax.get_legend().remove()

    return f, ax


def mm_to_inch(val_in_inch):
    mm = 0.1/2.54
    return val_in_inch * mm
