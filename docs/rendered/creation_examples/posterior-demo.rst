Using a Creator to Calculate True Posteriors for a Galaxy Sample
================================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, others…

last run successfully: March 7, 2024

This notebook demonstrates how to use a RAIL Engine to calculate true
posteriors for galaxy samples drawn from the same Engine. Note that this
notebook assumes you have already read through
``degradation-demo.ipynb``.

Calculating posteriors is more complicated than drawing samples, because
it requires more knowledge of the engine that underlies the Engine. In
this example, we will use the same engine we used in Degradation demo:
``FlowEngine`` which wraps a normalizing flow from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package.

This notebook will cover three scenarios of increasing complexity:

1. Calculating posteriors without errors

2. Calculating posteriors while convolving errors

3. Calculating posteriors with missing bands

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator, FlowPosterior
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.creation.degraders.quantityCut import QuantityCut
    from rail.creation.degraders.spectroscopic_degraders import (
        InvRedshiftIncompleteness,
        LineConfusion,
    )
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.tools.table_tools import ColumnMapper


.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the Rail data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


1. Calculating posteriors without errors
----------------------------------------

For a basic first example, let’s make a Creator with no degradation and
draw a sample.

Note that the ``FlowEngine.sample`` method is handing back a
``DataHandle``. When talking to rail stages we can use this as though it
were the underlying data and pass it as an argument. This allows the
rail stages to keep track of where their inputs are coming from.

.. code:: ipython3

    n_samples = 6
    # create the FlowCreator
    flowCreator = FlowCreator.make_stage(name="truth", model=flow_file, n_samples=n_samples)
    # draw a few samples
    samples_truth = flowCreator.sample(6, seed=0)



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth


Now, let’s calculate true posteriors for this sample. Note the important
fact here: these are literally the true posteriors for the sample
because pzflow gives us direct access to the probability distribution
from which the sample was drawn!

When calculating posteriors, the Engine will always require ``data``,
which is a pandas DataFrame of the galaxies for which we are calculating
posteriors (in out case the ``samples_truth``). Because we are using a
``FlowEngine``, we also must provide ``grid``, because ``FlowEngine``
calculates posteriors over a grid of redshift values.

Let’s calculate posteriors for every galaxy in our sample:

.. code:: ipython3

    flow_post = FlowPosterior.make_stage(
        name="truth_post",
        column="redshift",
        grid=np.linspace(0, 2.5, 100),
        marg_rules=dict(flag=np.nan, u=lambda row: np.linspace(25, 31, 10)),
        flow=flow_file,
    )


.. code:: ipython3

    pdfs = flow_post.get_posterior(samples_truth, column="redshift")



.. parsed-literal::

    Inserting handle into data store.  output_truth_post: inprogress_output_truth_post.hdf5, truth_post


Note that Creator returns the pdfs as a
`qp <https://github.com/LSSTDESC/qp>`__ Ensemble:

.. code:: ipython3

    pdfs.data





.. parsed-literal::

    <qp.ensemble.Ensemble at 0x7f1da5088790>



Let’s plot these pdfs:

.. code:: ipython3

    fig, axes = plt.subplots(2, 3, constrained_layout=True, dpi=120)
    
    for i, ax in enumerate(axes.flatten()):
        # plot the pdf
        pdfs.data[i].plot_native(axes=ax)
    
        # plot the true redshift
        ax.axvline(samples_truth.data["redshift"][i], c="k", ls="--")
    
        # remove x-ticks on top row
        if i < 3:
            ax.set(xticks=[])
        # set x-label on bottom row
        else:
            ax.set(xlabel="redshift")
        # set y-label on far left column
        if i % 3 == 0:
            ax.set(ylabel="p(z)")




.. image:: ../../../docs/rendered/creation_examples/posterior-demo_files/../../../docs/rendered/creation_examples/posterior-demo_14_0.png


The true posteriors are in blue, and the true redshifts are marked by
the vertical black lines.

## 2. Calculating posteriors while convolving errors Now, let’s get a
little more sophisticated.

Let’s recreate the Engine/Degredation we were using at the end of the
Degradation demo.

I will make one change however: the LSST Error Model sometimes results
in non-detections for faint galaxies. These non-detections are flagged
with inf. Calculating posteriors for galaxies with non-detections is
more complicated, so for now, I will add one additional QuantityCut to
remove any galaxies with missing magnitudes. To see how to calculate
posteriors for galaxies with missing magnitudes, see `Section
3 <#MissingBands>`__.

Now let’s draw a degraded sample:

.. code:: ipython3

    # set up the error model
    
    n_samples = 50
    # create the FlowEngine
    flowEngine_degr = FlowCreator.make_stage(
        name="degraded", flow_file=flow_file, n_samples=n_samples
    )
    # draw a few samples
    samples_degr = flowEngine_degr.sample(n_samples, seed=0)
    errorModel = LSSTErrorModel.make_stage(name="lsst_errors", input="xx", sigLim=5)
    quantityCut = QuantityCut.make_stage(
        name="gold_cut", input="xx", cuts={band: np.inf for band in "ugrizy"}
    )
    inv_incomplete = InvRedshiftIncompleteness.make_stage(
        name="incompleteness", pivot_redshift=0.8
    )
    
    OII = 3727
    OIII = 5007
    
    lc_2p_0II_0III = LineConfusion.make_stage(
        name="lc_2p_0II_0III", true_wavelen=OII, wrong_wavelen=OIII, frac_wrong=0.02
    )
    lc_1p_0III_0II = LineConfusion.make_stage(
        name="lc_1p_0III_0II", true_wavelen=OIII, wrong_wavelen=OII, frac_wrong=0.01
    )
    detection = QuantityCut.make_stage(name="detection", cuts={"i": 25.3})
    
    data = samples_degr
    for degr in [
        errorModel,
        quantityCut,
        inv_incomplete,
        lc_2p_0II_0III,
        lc_1p_0III_0II,
        detection,
    ]:
        data = degr(data)



.. parsed-literal::

    Inserting handle into data store.  output_degraded: inprogress_output_degraded.pq, degraded
    Inserting handle into data store.  output_lsst_errors: inprogress_output_lsst_errors.pq, lsst_errors
    Inserting handle into data store.  output_gold_cut: inprogress_output_gold_cut.pq, gold_cut
    Inserting handle into data store.  output_incompleteness: inprogress_output_incompleteness.pq, incompleteness
    Inserting handle into data store.  output_lc_2p_0II_0III: inprogress_output_lc_2p_0II_0III.pq, lc_2p_0II_0III
    Inserting handle into data store.  output_lc_1p_0III_0II: inprogress_output_lc_1p_0III_0II.pq, lc_1p_0III_0II
    Inserting handle into data store.  output_detection: inprogress_output_detection.pq, detection


.. code:: ipython3

    samples_degraded_wo_nondetects = data.data
    samples_degraded_wo_nondetects





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.336240</td>
          <td>26.013103</td>
          <td>0.255567</td>
          <td>25.839737</td>
          <td>0.077815</td>
          <td>25.085784</td>
          <td>0.035065</td>
          <td>24.957165</td>
          <td>0.051105</td>
          <td>24.834599</td>
          <td>0.087734</td>
          <td>24.498555</td>
          <td>0.145807</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2.042985</td>
          <td>25.648312</td>
          <td>0.188751</td>
          <td>25.543347</td>
          <td>0.059882</td>
          <td>25.183189</td>
          <td>0.038220</td>
          <td>25.163569</td>
          <td>0.061381</td>
          <td>24.607464</td>
          <td>0.071798</td>
          <td>24.022070</td>
          <td>0.096361</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.561264</td>
          <td>26.214777</td>
          <td>0.300960</td>
          <td>25.692100</td>
          <td>0.068305</td>
          <td>24.887389</td>
          <td>0.029446</td>
          <td>24.472898</td>
          <td>0.033274</td>
          <td>24.208844</td>
          <td>0.050419</td>
          <td>24.132263</td>
          <td>0.106122</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.576580</td>
          <td>26.280314</td>
          <td>0.317157</td>
          <td>25.992400</td>
          <td>0.089006</td>
          <td>25.062944</td>
          <td>0.034365</td>
          <td>24.517818</td>
          <td>0.034620</td>
          <td>24.367776</td>
          <td>0.058059</td>
          <td>24.126923</td>
          <td>0.105628</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.544048</td>
          <td>27.586809</td>
          <td>0.820192</td>
          <td>26.483476</td>
          <td>0.136533</td>
          <td>25.411218</td>
          <td>0.046785</td>
          <td>24.954369</td>
          <td>0.050978</td>
          <td>24.809610</td>
          <td>0.085825</td>
          <td>24.756805</td>
          <td>0.181759</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.264244</td>
          <td>24.142741</td>
          <td>0.051047</td>
          <td>24.051334</td>
          <td>0.016425</td>
          <td>23.927372</td>
          <td>0.013176</td>
          <td>24.094684</td>
          <td>0.023905</td>
          <td>23.845760</td>
          <td>0.036543</td>
          <td>24.129781</td>
          <td>0.105892</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.302007</td>
          <td>26.617960</td>
          <td>0.412923</td>
          <td>26.150604</td>
          <td>0.102245</td>
          <td>25.513147</td>
          <td>0.051217</td>
          <td>25.160401</td>
          <td>0.061209</td>
          <td>25.148884</td>
          <td>0.115530</td>
          <td>25.212315</td>
          <td>0.265538</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.260134</td>
          <td>24.526436</td>
          <td>0.071544</td>
          <td>24.028151</td>
          <td>0.016118</td>
          <td>23.504458</td>
          <td>0.009718</td>
          <td>23.371913</td>
          <td>0.013153</td>
          <td>23.133055</td>
          <td>0.019667</td>
          <td>23.184434</td>
          <td>0.045927</td>
        </tr>
        <tr>
          <th>13</th>
          <td>0.112887</td>
          <td>25.800273</td>
          <td>0.214372</td>
          <td>24.932612</td>
          <td>0.034893</td>
          <td>24.574676</td>
          <td>0.022439</td>
          <td>24.408248</td>
          <td>0.031432</td>
          <td>24.414219</td>
          <td>0.060501</td>
          <td>24.352076</td>
          <td>0.128492</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.236191</td>
          <td>26.007388</td>
          <td>0.254374</td>
          <td>24.236774</td>
          <td>0.019141</td>
          <td>23.177545</td>
          <td>0.007983</td>
          <td>22.748764</td>
          <td>0.008563</td>
          <td>22.512078</td>
          <td>0.011958</td>
          <td>22.305319</td>
          <td>0.021259</td>
        </tr>
        <tr>
          <th>16</th>
          <td>0.829715</td>
          <td>25.443429</td>
          <td>0.158664</td>
          <td>25.213030</td>
          <td>0.044695</td>
          <td>24.646369</td>
          <td>0.023869</td>
          <td>23.842940</td>
          <td>0.019277</td>
          <td>23.619985</td>
          <td>0.029950</td>
          <td>23.485320</td>
          <td>0.059987</td>
        </tr>
        <tr>
          <th>17</th>
          <td>0.639660</td>
          <td>23.453601</td>
          <td>0.027984</td>
          <td>23.346267</td>
          <td>0.009732</td>
          <td>22.726911</td>
          <td>0.006515</td>
          <td>22.130597</td>
          <td>0.006416</td>
          <td>21.906731</td>
          <td>0.008058</td>
          <td>21.706502</td>
          <td>0.013032</td>
        </tr>
        <tr>
          <th>18</th>
          <td>1.241073</td>
          <td>25.302192</td>
          <td>0.140599</td>
          <td>25.150865</td>
          <td>0.042303</td>
          <td>25.013402</td>
          <td>0.032896</td>
          <td>24.535551</td>
          <td>0.035166</td>
          <td>24.045582</td>
          <td>0.043618</td>
          <td>23.485191</td>
          <td>0.059980</td>
        </tr>
        <tr>
          <th>20</th>
          <td>1.099543</td>
          <td>25.926323</td>
          <td>0.237981</td>
          <td>25.435832</td>
          <td>0.054443</td>
          <td>24.897869</td>
          <td>0.029718</td>
          <td>24.415269</td>
          <td>0.031627</td>
          <td>23.792580</td>
          <td>0.034865</td>
          <td>23.560429</td>
          <td>0.064118</td>
        </tr>
        <tr>
          <th>21</th>
          <td>0.829761</td>
          <td>25.623656</td>
          <td>0.184870</td>
          <td>25.965930</td>
          <td>0.086959</td>
          <td>25.563862</td>
          <td>0.053576</td>
          <td>24.750236</td>
          <td>0.042529</td>
          <td>24.383654</td>
          <td>0.058882</td>
          <td>24.326717</td>
          <td>0.125699</td>
        </tr>
        <tr>
          <th>23</th>
          <td>0.454979</td>
          <td>30.568626</td>
          <td>3.172564</td>
          <td>26.173784</td>
          <td>0.104338</td>
          <td>25.387700</td>
          <td>0.045818</td>
          <td>25.172704</td>
          <td>0.061880</td>
          <td>24.849918</td>
          <td>0.088924</td>
          <td>24.737538</td>
          <td>0.178816</td>
        </tr>
        <tr>
          <th>27</th>
          <td>0.935433</td>
          <td>26.159404</td>
          <td>0.287841</td>
          <td>25.927198</td>
          <td>0.084047</td>
          <td>25.030035</td>
          <td>0.033382</td>
          <td>24.387334</td>
          <td>0.030860</td>
          <td>23.863827</td>
          <td>0.037131</td>
          <td>23.659572</td>
          <td>0.070003</td>
        </tr>
        <tr>
          <th>28</th>
          <td>0.431346</td>
          <td>27.736257</td>
          <td>0.901934</td>
          <td>25.901932</td>
          <td>0.082198</td>
          <td>24.490223</td>
          <td>0.020875</td>
          <td>24.218550</td>
          <td>0.026620</td>
          <td>24.047961</td>
          <td>0.043710</td>
          <td>23.721986</td>
          <td>0.073976</td>
        </tr>
        <tr>
          <th>29</th>
          <td>1.742907</td>
          <td>25.646833</td>
          <td>0.188517</td>
          <td>25.653896</td>
          <td>0.066036</td>
          <td>25.334431</td>
          <td>0.043703</td>
          <td>24.912766</td>
          <td>0.049129</td>
          <td>24.522562</td>
          <td>0.066599</td>
          <td>24.190177</td>
          <td>0.111626</td>
        </tr>
        <tr>
          <th>30</th>
          <td>0.214016</td>
          <td>24.355966</td>
          <td>0.061582</td>
          <td>23.542877</td>
          <td>0.011125</td>
          <td>23.079748</td>
          <td>0.007586</td>
          <td>22.866631</td>
          <td>0.009201</td>
          <td>22.845544</td>
          <td>0.015500</td>
          <td>22.789098</td>
          <td>0.032371</td>
        </tr>
        <tr>
          <th>31</th>
          <td>0.357573</td>
          <td>24.788775</td>
          <td>0.090053</td>
          <td>23.743596</td>
          <td>0.012891</td>
          <td>22.760936</td>
          <td>0.006597</td>
          <td>22.519061</td>
          <td>0.007556</td>
          <td>22.213933</td>
          <td>0.009700</td>
          <td>22.215507</td>
          <td>0.019697</td>
        </tr>
        <tr>
          <th>32</th>
          <td>0.419062</td>
          <td>26.218772</td>
          <td>0.301927</td>
          <td>26.073485</td>
          <td>0.095570</td>
          <td>25.366380</td>
          <td>0.044959</td>
          <td>25.096615</td>
          <td>0.057841</td>
          <td>24.962612</td>
          <td>0.098176</td>
          <td>25.529599</td>
          <td>0.342612</td>
        </tr>
      </tbody>
    </table>
    </div>



This sample has photometric errors that we would like to convolve in the
redshift posteriors, so that the posteriors are fully consistent with
the errors. We can perform this convolution by sampling from the error
distributions, calculating posteriors, and averaging.

``FlowEngine`` has this functionality already built in - we just have to
provide ``err_samples`` to the ``get_posterior`` method.

Let’s calculate posteriors with a variable number of error samples.

.. code:: ipython3

    grid = np.linspace(0, 2.5, 100)
    
    
    def get_degr_post(key, data, **kwargs):
        flow_degr_post = FlowPosterior.make_stage(name=f"degr_post_{key}", **kwargs)
        return flow_degr_post.get_posterior(data, column="redshift")


.. code:: ipython3

    degr_kwargs = dict(
        column="redshift",
        flow_file=flow_file,
        marg_rules=dict(flag=np.nan, u=lambda row: np.linspace(25, 31, 10)),
        grid=grid,
        seed=0,
        batch_size=2,
    )
    pdfs_errs_convolved = {
        err_samples: get_degr_post(
            f"{str(err_samples)}", data, err_samples=err_samples, **degr_kwargs
        )
        for err_samples in [1, 10, 100, 1000]
    }



.. parsed-literal::

    Inserting handle into data store.  output_degr_post_1: inprogress_output_degr_post_1.hdf5, degr_post_1


.. parsed-literal::

    Inserting handle into data store.  output_degr_post_10: inprogress_output_degr_post_10.hdf5, degr_post_10


.. parsed-literal::

    Inserting handle into data store.  output_degr_post_100: inprogress_output_degr_post_100.hdf5, degr_post_100


.. parsed-literal::

    Inserting handle into data store.  output_degr_post_1000: inprogress_output_degr_post_1000.hdf5, degr_post_1000


.. code:: ipython3

    fig, axes = plt.subplots(2, 3, dpi=120)
    
    for i, ax in enumerate(axes.flatten()):
        # set dummy values for xlim
        xlim = [np.inf, -np.inf]
    
        for pdfs_ in pdfs_errs_convolved.values():
            # plot the pdf
            pdfs_.data[i].plot_native(axes=ax)
    
            # get the x value where the pdf first rises above 2
            xmin = grid[np.argmax(pdfs_.data[i].pdf(grid)[0] > 2)]
            if xmin < xlim[0]:
                xlim[0] = xmin
    
            # get the x value where the pdf finally falls below 2
            xmax = grid[-np.argmax(pdfs_.data[i].pdf(grid)[0, ::-1] > 2)]
            if xmax > xlim[1]:
                xlim[1] = xmax
    
        # plot the true redshift
        z_true = samples_degraded_wo_nondetects["redshift"].iloc[i]
        ax.axvline(z_true, c="k", ls="--")
    
        # set x-label on bottom row
        if i >= 3:
            ax.set(xlabel="redshift")
        # set y-label on far left column
        if i % 3 == 0:
            ax.set(ylabel="p(z)")
    
        # set the x-limits so we can see more detail
        xlim[0] -= 0.2
        xlim[1] += 0.2
        ax.set(xlim=xlim, yticks=[])
    
    # create the legend
    axes[0, 1].plot([], [], c="C0", label=f"1 sample")
    for i, n in enumerate([10, 100, 1000]):
        axes[0, 1].plot([], [], c=f"C{i+1}", label=f"{n} samples")
    axes[0, 1].legend(
        bbox_to_anchor=(0.5, 1.3),
        loc="upper center",
        ncol=4,
    )
    
    plt.show()




.. image:: ../../../docs/rendered/creation_examples/posterior-demo_files/../../../docs/rendered/creation_examples/posterior-demo_23_0.png


You can see the effect of convolving the errors. In particular, notice
that without error convolution (1 sample), the redshift posterior is
often totally inconsistent with the true redshift (marked by the
vertical black line). As you convolve more samples, the posterior
generally broadens and becomes consistent with the true redshift.

Also notice how the posterior continues to change as you convolve more
and more samples. This suggests that you need to do a little testing to
ensure that you have convolved enough samples.

3. Calculating posteriors with missing bands
--------------------------------------------

Now let’s finally tackle posterior calculation with missing bands.

First, lets make a sample that has missing bands. Let’s use the same
degrader as we used above, except without the final QuantityCut that
removed non-detections:

.. code:: ipython3

    samples_degraded = DS["output_lc_1p_0III_0II"]


You can see that galaxy 3 has a non-detection in the u band.
``FlowEngine`` can handle missing values by marginalizing over that
value. By default, ``FlowEngine`` will marginalize over NaNs in the u
band, using the grid ``u = np.linspace(25, 31, 10)``. This default grid
should work in most cases, but you may want to change the flag for
non-detections, use a different grid for the u band, or marginalize over
non-detections in other bands. In order to do these things, you must
supply ``FlowEngine`` with marginalization rules in the form of the
``marg_rules`` dictionary.

Let’s imagine we want to use a different grid for u band
marginalization. In order to determine what grid to use, we will create
a histogram of non-detections in u band vs true u band magnitude
(assuming year 10 LSST errors). This will tell me what are reasonable
values of u to marginalize over.

.. code:: ipython3

    # get true u band magnitudes
    true_u = DS["output_degraded"].data["u"].to_numpy()
    # get the observed u band magnitudes
    obs_u = DS["output_lsst_errors"].data["u"].to_numpy()
    
    # create the figure
    fig, ax = plt.subplots(constrained_layout=True, dpi=100)
    # plot the u band detections
    ax.hist(true_u[np.isfinite(obs_u)], bins=10, range=(23, 31), label="detected")
    # plot the u band non-detections
    ax.hist(true_u[~np.isfinite(obs_u)], bins=10, range=(23, 31), label="non-detected")
    
    ax.legend()
    ax.set(xlabel="true u magnitude")
    
    plt.show()




.. image:: ../../../docs/rendered/creation_examples/posterior-demo_files/../../../docs/rendered/creation_examples/posterior-demo_28_0.png


Based on this histogram, I will marginalize over u band values from 25
to 31. Like how I tested different numbers of error samples above, here
I will test different resolutions for the u band grid.

I will provide our new u band grid in the ``marg_rules`` dictionary,
which will also include ``"flag"`` which tells ``FlowEngine`` what my
flag for non-detections is. In this simple example, we are using a fixed
grid for the u band, but notice that the u band rule takes the form of a
function - this is because the grid over which to marginalize can be a
function of any of the other variables in the row. If I wanted to
marginalize over any other bands, I would need to include corresponding
rules in ``marg_rules`` too.

For this example, I will only calculate pdfs for galaxy 3, which is the
galaxy with a non-detection in the u band. Also, similarly to how I
tested the error convolution with a variable number of samples, I will
test the marginalization with varying resolutions for the marginalized
grid.

.. code:: ipython3

    from rail.tools.table_tools import RowSelector
    
    # dict to save the marginalized posteriors
    pdfs_u_marginalized = {}
    
    row3_selector = RowSelector.make_stage(name="select_row3", start=3, stop=4)
    row3_degraded = row3_selector(samples_degraded)
    
    degr_post_kwargs = dict(
        grid=grid, err_samples=10000, seed=0, flow_file=flow_file, column="redshift"
    )
    
    # iterate over variable grid resolution
    for nbins in [10, 20, 50, 100]:
        # set up the marginalization rules for this grid resolution
        marg_rules = {
            "flag": errorModel.config["ndFlag"],
            "u": lambda row: np.linspace(25, 31, nbins),
        }
    
        # calculate the posterior by marginalizing over u and sampling
        # from the error distributions of the other galaxies
        pdfs_u_marginalized[nbins] = get_degr_post(
            f"degr_post_nbins_{nbins}",
            row3_degraded,
            marg_rules=marg_rules,
            **degr_post_kwargs,
        )



.. parsed-literal::

    Inserting handle into data store.  output_select_row3: inprogress_output_select_row3.pq, select_row3


.. parsed-literal::

    Inserting handle into data store.  output_degr_post_degr_post_nbins_10: inprogress_output_degr_post_degr_post_nbins_10.hdf5, degr_post_degr_post_nbins_10


.. parsed-literal::

    Inserting handle into data store.  output_degr_post_degr_post_nbins_20: inprogress_output_degr_post_degr_post_nbins_20.hdf5, degr_post_degr_post_nbins_20


.. parsed-literal::

    Inserting handle into data store.  output_degr_post_degr_post_nbins_50: inprogress_output_degr_post_degr_post_nbins_50.hdf5, degr_post_degr_post_nbins_50


.. parsed-literal::

    Inserting handle into data store.  output_degr_post_degr_post_nbins_100: inprogress_output_degr_post_degr_post_nbins_100.hdf5, degr_post_degr_post_nbins_100


.. code:: ipython3

    fig, ax = plt.subplots(dpi=100)
    for i in [10, 20, 50, 100]:
        pdfs_u_marginalized[i]()[0].plot_native(axes=ax, label=f"{i} bins")
    ax.axvline(samples_degraded().iloc[3]["redshift"], label="True redshift", c="k")
    ax.legend()
    ax.set(xlabel="Redshift")
    plt.show()




.. image:: ../../../docs/rendered/creation_examples/posterior-demo_files/../../../docs/rendered/creation_examples/posterior-demo_31_0.png


Notice that the resolution with only 10 bins is sufficient for this
marginalization.

In this example, only one of the bands featured a non-detection, but you
can easily marginalize over more bands by including corresponding rules
in the ``marg_rules`` dict. Note that marginalizing over multiple bands
quickly gets expensive.
