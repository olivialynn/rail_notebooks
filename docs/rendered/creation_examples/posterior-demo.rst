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

    <qp.ensemble.Ensemble at 0x7f0efc152740>



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
          <td>26.470851</td>
          <td>0.368571</td>
          <td>25.957324</td>
          <td>0.086304</td>
          <td>25.118097</td>
          <td>0.036081</td>
          <td>25.016649</td>
          <td>0.053877</td>
          <td>24.725753</td>
          <td>0.079709</td>
          <td>24.812755</td>
          <td>0.190558</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.008050</td>
          <td>24.000810</td>
          <td>0.045060</td>
          <td>23.880502</td>
          <td>0.014327</td>
          <td>23.625955</td>
          <td>0.010553</td>
          <td>23.121388</td>
          <td>0.010920</td>
          <td>22.606426</td>
          <td>0.012839</td>
          <td>22.334444</td>
          <td>0.021795</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.561264</td>
          <td>28.611496</td>
          <td>1.477344</td>
          <td>25.731191</td>
          <td>0.070706</td>
          <td>24.819637</td>
          <td>0.027750</td>
          <td>24.325834</td>
          <td>0.029237</td>
          <td>24.280072</td>
          <td>0.053711</td>
          <td>24.157877</td>
          <td>0.108524</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.576580</td>
          <td>30.111069</td>
          <td>2.745232</td>
          <td>26.104048</td>
          <td>0.098164</td>
          <td>25.075374</td>
          <td>0.034744</td>
          <td>24.506416</td>
          <td>0.034273</td>
          <td>24.294857</td>
          <td>0.054420</td>
          <td>24.133850</td>
          <td>0.106269</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.544048</td>
          <td>28.528381</td>
          <td>1.416175</td>
          <td>26.183496</td>
          <td>0.105227</td>
          <td>25.526876</td>
          <td>0.051845</td>
          <td>24.993726</td>
          <td>0.052791</td>
          <td>24.925076</td>
          <td>0.094996</td>
          <td>24.573929</td>
          <td>0.155548</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.264244</td>
          <td>24.191945</td>
          <td>0.053305</td>
          <td>24.049801</td>
          <td>0.016404</td>
          <td>23.946484</td>
          <td>0.013372</td>
          <td>24.078758</td>
          <td>0.023579</td>
          <td>23.860681</td>
          <td>0.037028</td>
          <td>24.191618</td>
          <td>0.111766</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.260134</td>
          <td>24.641377</td>
          <td>0.079143</td>
          <td>23.986967</td>
          <td>0.015592</td>
          <td>23.496976</td>
          <td>0.009671</td>
          <td>23.378015</td>
          <td>0.013216</td>
          <td>23.130330</td>
          <td>0.019622</td>
          <td>23.198500</td>
          <td>0.046504</td>
        </tr>
        <tr>
          <th>16</th>
          <td>0.112887</td>
          <td>26.213712</td>
          <td>0.300703</td>
          <td>24.871574</td>
          <td>0.033070</td>
          <td>24.554247</td>
          <td>0.022049</td>
          <td>24.357965</td>
          <td>0.030073</td>
          <td>24.421979</td>
          <td>0.060919</td>
          <td>24.431477</td>
          <td>0.137622</td>
        </tr>
        <tr>
          <th>17</th>
          <td>0.236191</td>
          <td>26.015455</td>
          <td>0.256059</td>
          <td>24.253001</td>
          <td>0.019403</td>
          <td>23.164806</td>
          <td>0.007928</td>
          <td>22.752810</td>
          <td>0.008583</td>
          <td>22.507467</td>
          <td>0.011917</td>
          <td>22.344322</td>
          <td>0.021980</td>
        </tr>
        <tr>
          <th>18</th>
          <td>0.639660</td>
          <td>23.492305</td>
          <td>0.028931</td>
          <td>23.352373</td>
          <td>0.009771</td>
          <td>22.728697</td>
          <td>0.006520</td>
          <td>22.125136</td>
          <td>0.006404</td>
          <td>21.904017</td>
          <td>0.008046</td>
          <td>21.671924</td>
          <td>0.012693</td>
        </tr>
        <tr>
          <th>21</th>
          <td>1.241073</td>
          <td>25.174546</td>
          <td>0.125959</td>
          <td>25.131797</td>
          <td>0.041596</td>
          <td>24.979436</td>
          <td>0.031926</td>
          <td>24.515122</td>
          <td>0.034537</td>
          <td>24.042693</td>
          <td>0.043507</td>
          <td>23.491625</td>
          <td>0.060323</td>
        </tr>
        <tr>
          <th>22</th>
          <td>1.099543</td>
          <td>25.729982</td>
          <td>0.202150</td>
          <td>25.486583</td>
          <td>0.056946</td>
          <td>24.926510</td>
          <td>0.030474</td>
          <td>24.444395</td>
          <td>0.032449</td>
          <td>23.771542</td>
          <td>0.034224</td>
          <td>23.444329</td>
          <td>0.057844</td>
        </tr>
        <tr>
          <th>23</th>
          <td>0.829761</td>
          <td>26.610826</td>
          <td>0.410675</td>
          <td>25.998079</td>
          <td>0.089451</td>
          <td>25.505043</td>
          <td>0.050849</td>
          <td>24.795296</td>
          <td>0.044264</td>
          <td>24.389541</td>
          <td>0.059191</td>
          <td>24.514250</td>
          <td>0.147787</td>
        </tr>
        <tr>
          <th>24</th>
          <td>0.454979</td>
          <td>26.707497</td>
          <td>0.442012</td>
          <td>26.057459</td>
          <td>0.094237</td>
          <td>25.456967</td>
          <td>0.048724</td>
          <td>25.130775</td>
          <td>0.059621</td>
          <td>24.990348</td>
          <td>0.100591</td>
          <td>24.845423</td>
          <td>0.195875</td>
        </tr>
        <tr>
          <th>25</th>
          <td>1.156458</td>
          <td>26.963779</td>
          <td>0.534496</td>
          <td>26.121026</td>
          <td>0.099634</td>
          <td>25.522105</td>
          <td>0.051626</td>
          <td>25.023142</td>
          <td>0.054188</td>
          <td>24.285643</td>
          <td>0.053977</td>
          <td>23.841758</td>
          <td>0.082228</td>
        </tr>
        <tr>
          <th>29</th>
          <td>0.935433</td>
          <td>26.109721</td>
          <td>0.276499</td>
          <td>26.141319</td>
          <td>0.101419</td>
          <td>25.051710</td>
          <td>0.034026</td>
          <td>24.382360</td>
          <td>0.030725</td>
          <td>23.813228</td>
          <td>0.035507</td>
          <td>23.653952</td>
          <td>0.069655</td>
        </tr>
        <tr>
          <th>30</th>
          <td>0.431346</td>
          <td>27.693303</td>
          <td>0.877918</td>
          <td>25.799329</td>
          <td>0.075090</td>
          <td>24.544604</td>
          <td>0.021868</td>
          <td>24.253805</td>
          <td>0.027451</td>
          <td>24.020097</td>
          <td>0.042643</td>
          <td>23.772722</td>
          <td>0.077368</td>
        </tr>
        <tr>
          <th>32</th>
          <td>0.214016</td>
          <td>24.204269</td>
          <td>0.053886</td>
          <td>23.543257</td>
          <td>0.011128</td>
          <td>23.083297</td>
          <td>0.007599</td>
          <td>22.864500</td>
          <td>0.009189</td>
          <td>22.856350</td>
          <td>0.015636</td>
          <td>22.787302</td>
          <td>0.032320</td>
        </tr>
        <tr>
          <th>33</th>
          <td>0.357573</td>
          <td>24.624694</td>
          <td>0.077993</td>
          <td>23.741173</td>
          <td>0.012867</td>
          <td>22.756085</td>
          <td>0.006586</td>
          <td>22.511452</td>
          <td>0.007527</td>
          <td>22.212438</td>
          <td>0.009690</td>
          <td>22.185889</td>
          <td>0.019211</td>
        </tr>
        <tr>
          <th>34</th>
          <td>0.419062</td>
          <td>28.390104</td>
          <td>1.317219</td>
          <td>26.240165</td>
          <td>0.110560</td>
          <td>25.364330</td>
          <td>0.044878</td>
          <td>25.195996</td>
          <td>0.063172</td>
          <td>25.011273</td>
          <td>0.102451</td>
          <td>24.999994</td>
          <td>0.222915</td>
        </tr>
        <tr>
          <th>35</th>
          <td>0.561019</td>
          <td>25.720694</td>
          <td>0.200583</td>
          <td>25.861768</td>
          <td>0.079341</td>
          <td>25.485612</td>
          <td>0.049980</td>
          <td>25.289643</td>
          <td>0.068638</td>
          <td>25.355560</td>
          <td>0.138196</td>
          <td>25.830056</td>
          <td>0.432404</td>
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
