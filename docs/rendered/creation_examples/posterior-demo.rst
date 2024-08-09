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

    <qp.ensemble.Ensemble at 0x7fb024a2b0d0>



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
          <td>26.231991</td>
          <td>0.305143</td>
          <td>25.995328</td>
          <td>0.089235</td>
          <td>25.050767</td>
          <td>0.033998</td>
          <td>25.065510</td>
          <td>0.056266</td>
          <td>24.707030</td>
          <td>0.078402</td>
          <td>24.809389</td>
          <td>0.190018</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2.042985</td>
          <td>25.862227</td>
          <td>0.225698</td>
          <td>25.572243</td>
          <td>0.061433</td>
          <td>25.224269</td>
          <td>0.039636</td>
          <td>25.118810</td>
          <td>0.058991</td>
          <td>24.467562</td>
          <td>0.063431</td>
          <td>23.996784</td>
          <td>0.094246</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.008050</td>
          <td>24.025391</td>
          <td>0.046043</td>
          <td>23.899159</td>
          <td>0.014539</td>
          <td>23.619498</td>
          <td>0.010506</td>
          <td>23.111180</td>
          <td>0.010842</td>
          <td>22.604295</td>
          <td>0.012818</td>
          <td>22.395031</td>
          <td>0.022958</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.561264</td>
          <td>26.169416</td>
          <td>0.290175</td>
          <td>25.680621</td>
          <td>0.067615</td>
          <td>24.823218</td>
          <td>0.027837</td>
          <td>24.389587</td>
          <td>0.030921</td>
          <td>24.232611</td>
          <td>0.051495</td>
          <td>24.212800</td>
          <td>0.113849</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.576580</td>
          <td>26.294730</td>
          <td>0.320820</td>
          <td>25.953042</td>
          <td>0.085980</td>
          <td>25.018890</td>
          <td>0.033055</td>
          <td>24.529033</td>
          <td>0.034964</td>
          <td>24.370537</td>
          <td>0.058201</td>
          <td>24.195666</td>
          <td>0.112161</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.544048</td>
          <td>27.341285</td>
          <td>0.697026</td>
          <td>26.373415</td>
          <td>0.124138</td>
          <td>25.447158</td>
          <td>0.048302</td>
          <td>25.050798</td>
          <td>0.055535</td>
          <td>25.053965</td>
          <td>0.106349</td>
          <td>24.602660</td>
          <td>0.159419</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.264244</td>
          <td>24.236799</td>
          <td>0.055451</td>
          <td>24.025217</td>
          <td>0.016080</td>
          <td>23.953817</td>
          <td>0.013448</td>
          <td>24.108960</td>
          <td>0.024202</td>
          <td>23.878234</td>
          <td>0.037607</td>
          <td>24.418298</td>
          <td>0.136065</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.302007</td>
          <td>26.612784</td>
          <td>0.411291</td>
          <td>26.130092</td>
          <td>0.100427</td>
          <td>25.540339</td>
          <td>0.052468</td>
          <td>25.230074</td>
          <td>0.065109</td>
          <td>25.045538</td>
          <td>0.105568</td>
          <td>25.005124</td>
          <td>0.223867</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.260134</td>
          <td>24.526254</td>
          <td>0.071532</td>
          <td>24.017114</td>
          <td>0.015975</td>
          <td>23.516336</td>
          <td>0.009795</td>
          <td>23.368257</td>
          <td>0.013116</td>
          <td>23.131393</td>
          <td>0.019639</td>
          <td>23.197485</td>
          <td>0.046462</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.112887</td>
          <td>25.501857</td>
          <td>0.166755</td>
          <td>24.915667</td>
          <td>0.034377</td>
          <td>24.550955</td>
          <td>0.021987</td>
          <td>24.441011</td>
          <td>0.032352</td>
          <td>24.482942</td>
          <td>0.064302</td>
          <td>24.395446</td>
          <td>0.133405</td>
        </tr>
        <tr>
          <th>15</th>
          <td>0.236191</td>
          <td>25.924955</td>
          <td>0.237713</td>
          <td>24.274623</td>
          <td>0.019758</td>
          <td>23.160856</td>
          <td>0.007911</td>
          <td>22.757491</td>
          <td>0.008607</td>
          <td>22.512001</td>
          <td>0.011958</td>
          <td>22.306227</td>
          <td>0.021275</td>
        </tr>
        <tr>
          <th>17</th>
          <td>0.829715</td>
          <td>25.425247</td>
          <td>0.156222</td>
          <td>25.202597</td>
          <td>0.044284</td>
          <td>24.611374</td>
          <td>0.023159</td>
          <td>23.851221</td>
          <td>0.019413</td>
          <td>23.623434</td>
          <td>0.030041</td>
          <td>23.478314</td>
          <td>0.059615</td>
        </tr>
        <tr>
          <th>18</th>
          <td>0.639660</td>
          <td>23.468101</td>
          <td>0.028335</td>
          <td>23.346064</td>
          <td>0.009731</td>
          <td>22.733876</td>
          <td>0.006532</td>
          <td>22.127551</td>
          <td>0.006409</td>
          <td>21.904284</td>
          <td>0.008047</td>
          <td>21.708161</td>
          <td>0.013049</td>
        </tr>
        <tr>
          <th>20</th>
          <td>1.241073</td>
          <td>25.472270</td>
          <td>0.162611</td>
          <td>25.163575</td>
          <td>0.042781</td>
          <td>25.044950</td>
          <td>0.033824</td>
          <td>24.534962</td>
          <td>0.035148</td>
          <td>24.024651</td>
          <td>0.042816</td>
          <td>23.571393</td>
          <td>0.064744</td>
        </tr>
        <tr>
          <th>21</th>
          <td>1.099543</td>
          <td>25.610485</td>
          <td>0.182827</td>
          <td>25.580969</td>
          <td>0.061910</td>
          <td>24.930851</td>
          <td>0.030591</td>
          <td>24.347719</td>
          <td>0.029804</td>
          <td>23.763693</td>
          <td>0.033988</td>
          <td>23.654608</td>
          <td>0.069696</td>
        </tr>
        <tr>
          <th>22</th>
          <td>0.829761</td>
          <td>26.168569</td>
          <td>0.289977</td>
          <td>26.082850</td>
          <td>0.096358</td>
          <td>25.529912</td>
          <td>0.051985</td>
          <td>24.775384</td>
          <td>0.043489</td>
          <td>24.311721</td>
          <td>0.055241</td>
          <td>24.583246</td>
          <td>0.156794</td>
        </tr>
        <tr>
          <th>25</th>
          <td>0.454979</td>
          <td>27.310747</td>
          <td>0.682677</td>
          <td>26.032834</td>
          <td>0.092223</td>
          <td>25.409096</td>
          <td>0.046697</td>
          <td>25.088735</td>
          <td>0.057438</td>
          <td>24.869434</td>
          <td>0.090464</td>
          <td>24.535757</td>
          <td>0.150541</td>
        </tr>
        <tr>
          <th>26</th>
          <td>1.156458</td>
          <td>26.676563</td>
          <td>0.431779</td>
          <td>26.102265</td>
          <td>0.098011</td>
          <td>25.537929</td>
          <td>0.052356</td>
          <td>24.957862</td>
          <td>0.051137</td>
          <td>24.315290</td>
          <td>0.055416</td>
          <td>24.039666</td>
          <td>0.097859</td>
        </tr>
        <tr>
          <th>29</th>
          <td>0.935433</td>
          <td>28.072870</td>
          <td>1.104390</td>
          <td>25.822087</td>
          <td>0.076613</td>
          <td>25.032521</td>
          <td>0.033455</td>
          <td>24.307510</td>
          <td>0.028772</td>
          <td>23.866298</td>
          <td>0.037212</td>
          <td>23.599146</td>
          <td>0.066356</td>
        </tr>
        <tr>
          <th>30</th>
          <td>1.742907</td>
          <td>26.264383</td>
          <td>0.313152</td>
          <td>25.694111</td>
          <td>0.068426</td>
          <td>25.305664</td>
          <td>0.042602</td>
          <td>24.827904</td>
          <td>0.045564</td>
          <td>24.487511</td>
          <td>0.064563</td>
          <td>24.402736</td>
          <td>0.134249</td>
        </tr>
        <tr>
          <th>32</th>
          <td>0.214016</td>
          <td>24.168792</td>
          <td>0.052230</td>
          <td>23.550005</td>
          <td>0.011181</td>
          <td>23.086917</td>
          <td>0.007613</td>
          <td>22.863412</td>
          <td>0.009183</td>
          <td>22.846497</td>
          <td>0.015512</td>
          <td>22.758702</td>
          <td>0.031517</td>
        </tr>
        <tr>
          <th>33</th>
          <td>0.357573</td>
          <td>24.649042</td>
          <td>0.079677</td>
          <td>23.738797</td>
          <td>0.012844</td>
          <td>22.749577</td>
          <td>0.006570</td>
          <td>22.514450</td>
          <td>0.007539</td>
          <td>22.212471</td>
          <td>0.009691</td>
          <td>22.170814</td>
          <td>0.018969</td>
        </tr>
        <tr>
          <th>34</th>
          <td>0.419062</td>
          <td>26.476248</td>
          <td>0.370125</td>
          <td>26.467422</td>
          <td>0.134654</td>
          <td>25.477777</td>
          <td>0.049633</td>
          <td>25.287660</td>
          <td>0.068517</td>
          <td>25.001880</td>
          <td>0.101612</td>
          <td>24.583851</td>
          <td>0.156875</td>
        </tr>
        <tr>
          <th>37</th>
          <td>0.561019</td>
          <td>25.870618</td>
          <td>0.227273</td>
          <td>25.741562</td>
          <td>0.071356</td>
          <td>25.377930</td>
          <td>0.045423</td>
          <td>25.253656</td>
          <td>0.066484</td>
          <td>25.388629</td>
          <td>0.142191</td>
          <td>24.819377</td>
          <td>0.191625</td>
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
