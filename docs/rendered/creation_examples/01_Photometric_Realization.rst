Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f8e681aaa10>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.009473  0.009299  
    1      25.391064  0.137136  0.128132  
    2      24.304707  0.175831  0.130941  
    3      25.291103  0.033338  0.031577  
    4      25.096743  0.016748  0.011759  
    ...          ...       ...       ...  
    99995  24.737946  0.045570  0.041107  
    99996  24.224169  0.124734  0.064832  
    99997  25.613836  0.133768  0.112796  
    99998  25.274899  0.247178  0.162377  
    99999  25.699642  0.012925  0.007844  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




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
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.469750</td>
          <td>0.368255</td>
          <td>26.635353</td>
          <td>0.155562</td>
          <td>26.036950</td>
          <td>0.081460</td>
          <td>25.319265</td>
          <td>0.070462</td>
          <td>24.546720</td>
          <td>0.068040</td>
          <td>24.067564</td>
          <td>0.100281</td>
          <td>0.009473</td>
          <td>0.009299</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.459099</td>
          <td>1.366146</td>
          <td>28.094538</td>
          <td>0.503181</td>
          <td>26.727513</td>
          <td>0.148742</td>
          <td>26.229081</td>
          <td>0.155944</td>
          <td>25.767412</td>
          <td>0.196326</td>
          <td>24.913534</td>
          <td>0.207399</td>
          <td>0.137136</td>
          <td>0.128132</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.160539</td>
          <td>0.528116</td>
          <td>27.729526</td>
          <td>0.341048</td>
          <td>26.310799</td>
          <td>0.167217</td>
          <td>25.098309</td>
          <td>0.110547</td>
          <td>24.149230</td>
          <td>0.107707</td>
          <td>0.175831</td>
          <td>0.130941</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.975179</td>
          <td>0.460453</td>
          <td>27.582648</td>
          <td>0.303396</td>
          <td>26.231495</td>
          <td>0.156267</td>
          <td>25.569343</td>
          <td>0.166005</td>
          <td>25.742211</td>
          <td>0.404337</td>
          <td>0.033338</td>
          <td>0.031577</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.201640</td>
          <td>0.633151</td>
          <td>26.202498</td>
          <td>0.106987</td>
          <td>25.956508</td>
          <td>0.075875</td>
          <td>25.563333</td>
          <td>0.087405</td>
          <td>25.353921</td>
          <td>0.138000</td>
          <td>25.226988</td>
          <td>0.268735</td>
          <td>0.016748</td>
          <td>0.011759</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.154908</td>
          <td>0.612767</td>
          <td>26.467437</td>
          <td>0.134656</td>
          <td>25.584954</td>
          <td>0.054588</td>
          <td>25.129929</td>
          <td>0.059576</td>
          <td>24.705341</td>
          <td>0.078285</td>
          <td>24.801513</td>
          <td>0.188759</td>
          <td>0.045570</td>
          <td>0.041107</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.420495</td>
          <td>3.032948</td>
          <td>27.442447</td>
          <td>0.304323</td>
          <td>26.222988</td>
          <td>0.095948</td>
          <td>25.074586</td>
          <td>0.056721</td>
          <td>24.842387</td>
          <td>0.088337</td>
          <td>24.220128</td>
          <td>0.114578</td>
          <td>0.124734</td>
          <td>0.064832</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.369934</td>
          <td>0.710683</td>
          <td>26.533709</td>
          <td>0.142570</td>
          <td>26.311134</td>
          <td>0.103653</td>
          <td>26.505515</td>
          <td>0.197186</td>
          <td>26.295498</td>
          <td>0.303231</td>
          <td>24.969348</td>
          <td>0.217299</td>
          <td>0.133768</td>
          <td>0.112796</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.250378</td>
          <td>0.309667</td>
          <td>26.505903</td>
          <td>0.139198</td>
          <td>26.082671</td>
          <td>0.084810</td>
          <td>25.959055</td>
          <td>0.123551</td>
          <td>25.652229</td>
          <td>0.178126</td>
          <td>25.573513</td>
          <td>0.354664</td>
          <td>0.247178</td>
          <td>0.162377</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.593721</td>
          <td>0.405325</td>
          <td>27.230741</td>
          <td>0.256316</td>
          <td>26.508863</td>
          <td>0.123148</td>
          <td>26.209033</td>
          <td>0.153289</td>
          <td>25.748545</td>
          <td>0.193233</td>
          <td>25.469615</td>
          <td>0.326714</td>
          <td>0.012925</td>
          <td>0.007844</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




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
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.632408</td>
          <td>0.461960</td>
          <td>26.920342</td>
          <td>0.226974</td>
          <td>26.010219</td>
          <td>0.093585</td>
          <td>25.234344</td>
          <td>0.077479</td>
          <td>24.860664</td>
          <td>0.105502</td>
          <td>24.008981</td>
          <td>0.112532</td>
          <td>0.009473</td>
          <td>0.009299</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.822418</td>
          <td>0.484437</td>
          <td>26.633069</td>
          <td>0.169639</td>
          <td>26.300113</td>
          <td>0.206103</td>
          <td>25.528029</td>
          <td>0.197743</td>
          <td>25.318321</td>
          <td>0.354850</td>
          <td>0.137136</td>
          <td>0.128132</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.659969</td>
          <td>0.495246</td>
          <td>29.499047</td>
          <td>1.414666</td>
          <td>32.201742</td>
          <td>3.695114</td>
          <td>25.831750</td>
          <td>0.141006</td>
          <td>24.838321</td>
          <td>0.111458</td>
          <td>24.326543</td>
          <td>0.159649</td>
          <td>0.175831</td>
          <td>0.130941</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.201099</td>
          <td>0.612414</td>
          <td>28.475622</td>
          <td>0.680696</td>
          <td>26.259644</td>
          <td>0.189047</td>
          <td>26.115253</td>
          <td>0.305031</td>
          <td>25.026290</td>
          <td>0.267289</td>
          <td>0.033338</td>
          <td>0.031577</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.708564</td>
          <td>0.222110</td>
          <td>26.272073</td>
          <td>0.131008</td>
          <td>25.916792</td>
          <td>0.086242</td>
          <td>25.689943</td>
          <td>0.115600</td>
          <td>25.490969</td>
          <td>0.181661</td>
          <td>25.090221</td>
          <td>0.280784</td>
          <td>0.016748</td>
          <td>0.011759</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.267008</td>
          <td>0.728116</td>
          <td>26.177420</td>
          <td>0.121310</td>
          <td>25.537442</td>
          <td>0.062030</td>
          <td>25.067814</td>
          <td>0.067297</td>
          <td>24.770519</td>
          <td>0.098091</td>
          <td>25.150574</td>
          <td>0.296398</td>
          <td>0.045570</td>
          <td>0.041107</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.384945</td>
          <td>2.238106</td>
          <td>26.974539</td>
          <td>0.243910</td>
          <td>26.180380</td>
          <td>0.112119</td>
          <td>25.260598</td>
          <td>0.081970</td>
          <td>24.628654</td>
          <td>0.088871</td>
          <td>24.175912</td>
          <td>0.134352</td>
          <td>0.124734</td>
          <td>0.064832</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.483390</td>
          <td>0.426438</td>
          <td>26.485057</td>
          <td>0.164229</td>
          <td>26.348378</td>
          <td>0.131959</td>
          <td>26.143512</td>
          <td>0.179366</td>
          <td>25.731134</td>
          <td>0.232691</td>
          <td>25.298827</td>
          <td>0.347190</td>
          <td>0.133768</td>
          <td>0.112796</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.440894</td>
          <td>0.435683</td>
          <td>26.073367</td>
          <td>0.123679</td>
          <td>26.245321</td>
          <td>0.130365</td>
          <td>26.192394</td>
          <td>0.201905</td>
          <td>25.536741</td>
          <td>0.213147</td>
          <td>24.821682</td>
          <td>0.254470</td>
          <td>0.247178</td>
          <td>0.162377</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.666641</td>
          <td>0.473956</td>
          <td>26.782750</td>
          <td>0.202381</td>
          <td>26.568934</td>
          <td>0.152062</td>
          <td>26.263063</td>
          <td>0.188992</td>
          <td>25.947989</td>
          <td>0.265633</td>
          <td>26.493004</td>
          <td>0.791596</td>
          <td>0.012925</td>
          <td>0.007844</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




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
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.460869</td>
          <td>0.134031</td>
          <td>26.110965</td>
          <td>0.087055</td>
          <td>25.231813</td>
          <td>0.065293</td>
          <td>24.741561</td>
          <td>0.080925</td>
          <td>24.093734</td>
          <td>0.102733</td>
          <td>0.009473</td>
          <td>0.009299</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.023592</td>
          <td>0.625964</td>
          <td>27.136836</td>
          <td>0.278561</td>
          <td>26.443008</td>
          <td>0.140811</td>
          <td>26.271545</td>
          <td>0.196493</td>
          <td>25.666629</td>
          <td>0.217030</td>
          <td>25.913223</td>
          <td>0.544584</td>
          <td>0.137136</td>
          <td>0.128132</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.470630</td>
          <td>0.778787</td>
          <td>28.173483</td>
          <td>0.587718</td>
          <td>25.994587</td>
          <td>0.163583</td>
          <td>24.929242</td>
          <td>0.121708</td>
          <td>24.308502</td>
          <td>0.158606</td>
          <td>0.175831</td>
          <td>0.130941</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.555054</td>
          <td>1.443863</td>
          <td>29.899744</td>
          <td>1.544487</td>
          <td>27.907127</td>
          <td>0.396709</td>
          <td>25.911122</td>
          <td>0.120263</td>
          <td>25.468061</td>
          <td>0.154345</td>
          <td>25.038473</td>
          <td>0.233356</td>
          <td>0.033338</td>
          <td>0.031577</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.500343</td>
          <td>0.377784</td>
          <td>26.124909</td>
          <td>0.100219</td>
          <td>25.869373</td>
          <td>0.070449</td>
          <td>25.646424</td>
          <td>0.094311</td>
          <td>25.368788</td>
          <td>0.140170</td>
          <td>24.802925</td>
          <td>0.189520</td>
          <td>0.016748</td>
          <td>0.011759</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.162610</td>
          <td>0.624695</td>
          <td>26.301498</td>
          <td>0.119156</td>
          <td>25.457625</td>
          <td>0.050001</td>
          <td>25.087230</td>
          <td>0.058900</td>
          <td>24.768375</td>
          <td>0.084858</td>
          <td>24.966013</td>
          <td>0.222078</td>
          <td>0.045570</td>
          <td>0.041107</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.522177</td>
          <td>0.835952</td>
          <td>26.745716</td>
          <td>0.188424</td>
          <td>26.156066</td>
          <td>0.101445</td>
          <td>25.190186</td>
          <td>0.070910</td>
          <td>24.744459</td>
          <td>0.090896</td>
          <td>24.309403</td>
          <td>0.139146</td>
          <td>0.124734</td>
          <td>0.064832</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.325158</td>
          <td>0.758799</td>
          <td>27.094751</td>
          <td>0.264312</td>
          <td>26.219145</td>
          <td>0.113510</td>
          <td>26.561483</td>
          <td>0.244844</td>
          <td>25.633590</td>
          <td>0.206788</td>
          <td>26.290117</td>
          <td>0.696959</td>
          <td>0.133768</td>
          <td>0.112796</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.209697</td>
          <td>0.388352</td>
          <td>26.220450</td>
          <td>0.151818</td>
          <td>26.003229</td>
          <td>0.114950</td>
          <td>25.808619</td>
          <td>0.158646</td>
          <td>25.524439</td>
          <td>0.228695</td>
          <td>25.861397</td>
          <td>0.609427</td>
          <td>0.247178</td>
          <td>0.162377</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.454183</td>
          <td>0.364155</td>
          <td>26.702593</td>
          <td>0.164971</td>
          <td>26.701934</td>
          <td>0.145729</td>
          <td>25.966052</td>
          <td>0.124504</td>
          <td>26.197884</td>
          <td>0.280661</td>
          <td>25.203918</td>
          <td>0.264119</td>
          <td>0.012925</td>
          <td>0.007844</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
