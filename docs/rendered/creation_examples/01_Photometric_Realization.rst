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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f9522d2dfc0>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>26.875716</td>
          <td>0.501139</td>
          <td>26.707900</td>
          <td>0.165503</td>
          <td>26.097566</td>
          <td>0.085931</td>
          <td>25.159892</td>
          <td>0.061181</td>
          <td>24.780336</td>
          <td>0.083640</td>
          <td>23.921390</td>
          <td>0.088203</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.534512</td>
          <td>0.327532</td>
          <td>26.711217</td>
          <td>0.146674</td>
          <td>26.115941</td>
          <td>0.141504</td>
          <td>25.905693</td>
          <td>0.220415</td>
          <td>25.571455</td>
          <td>0.354091</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.700891</td>
          <td>0.333408</td>
          <td>25.940164</td>
          <td>0.121541</td>
          <td>24.917268</td>
          <td>0.094347</td>
          <td>24.324700</td>
          <td>0.125480</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.872323</td>
          <td>0.980756</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.629422</td>
          <td>0.314977</td>
          <td>26.177338</td>
          <td>0.149177</td>
          <td>25.405643</td>
          <td>0.144289</td>
          <td>25.144059</td>
          <td>0.251104</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.220071</td>
          <td>0.302241</td>
          <td>26.256136</td>
          <td>0.112110</td>
          <td>26.072479</td>
          <td>0.084052</td>
          <td>25.711019</td>
          <td>0.099512</td>
          <td>25.774063</td>
          <td>0.197428</td>
          <td>24.953145</td>
          <td>0.214382</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.591650</td>
          <td>0.149846</td>
          <td>25.373928</td>
          <td>0.045262</td>
          <td>25.047911</td>
          <td>0.055393</td>
          <td>24.956760</td>
          <td>0.097673</td>
          <td>24.734216</td>
          <td>0.178313</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.838821</td>
          <td>0.184949</td>
          <td>25.990743</td>
          <td>0.078205</td>
          <td>25.239085</td>
          <td>0.065631</td>
          <td>24.787630</td>
          <td>0.084179</td>
          <td>24.130031</td>
          <td>0.105915</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.444013</td>
          <td>0.746871</td>
          <td>26.973051</td>
          <td>0.207051</td>
          <td>26.329989</td>
          <td>0.105376</td>
          <td>26.197927</td>
          <td>0.151836</td>
          <td>25.768728</td>
          <td>0.196544</td>
          <td>25.444923</td>
          <td>0.320356</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.386404</td>
          <td>0.344970</td>
          <td>26.088345</td>
          <td>0.096823</td>
          <td>26.174133</td>
          <td>0.091919</td>
          <td>25.762438</td>
          <td>0.104095</td>
          <td>25.645309</td>
          <td>0.177083</td>
          <td>25.354006</td>
          <td>0.297858</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.895471</td>
          <td>0.994576</td>
          <td>26.716304</td>
          <td>0.166692</td>
          <td>26.432327</td>
          <td>0.115220</td>
          <td>26.494312</td>
          <td>0.195336</td>
          <td>26.225770</td>
          <td>0.286662</td>
          <td>28.868644</td>
          <td>2.389326</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.748544</td>
          <td>0.196592</td>
          <td>26.008575</td>
          <td>0.093425</td>
          <td>25.217845</td>
          <td>0.076337</td>
          <td>24.839157</td>
          <td>0.103509</td>
          <td>24.139593</td>
          <td>0.126026</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.482720</td>
          <td>0.835934</td>
          <td>27.144359</td>
          <td>0.272824</td>
          <td>26.905605</td>
          <td>0.202315</td>
          <td>26.104873</td>
          <td>0.165234</td>
          <td>25.967925</td>
          <td>0.269948</td>
          <td>26.294775</td>
          <td>0.693445</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.337716</td>
          <td>0.627962</td>
          <td>25.841465</td>
          <td>0.134776</td>
          <td>25.055177</td>
          <td>0.127716</td>
          <td>23.960237</td>
          <td>0.110302</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.601625</td>
          <td>1.615773</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.369983</td>
          <td>0.665304</td>
          <td>26.209766</td>
          <td>0.193054</td>
          <td>25.563577</td>
          <td>0.205807</td>
          <td>25.293579</td>
          <td>0.351421</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.467089</td>
          <td>0.407570</td>
          <td>26.009152</td>
          <td>0.104218</td>
          <td>26.133826</td>
          <td>0.104297</td>
          <td>25.795891</td>
          <td>0.126696</td>
          <td>25.339897</td>
          <td>0.159700</td>
          <td>25.413918</td>
          <td>0.363292</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.461998</td>
          <td>0.157038</td>
          <td>25.596310</td>
          <td>0.066326</td>
          <td>25.138828</td>
          <td>0.072765</td>
          <td>24.817696</td>
          <td>0.103731</td>
          <td>24.653373</td>
          <td>0.199613</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.620500</td>
          <td>1.587696</td>
          <td>26.938803</td>
          <td>0.231240</td>
          <td>26.137052</td>
          <td>0.104993</td>
          <td>25.425305</td>
          <td>0.092043</td>
          <td>24.884565</td>
          <td>0.108147</td>
          <td>24.035852</td>
          <td>0.115656</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.230558</td>
          <td>0.713161</td>
          <td>26.534786</td>
          <td>0.165898</td>
          <td>26.588651</td>
          <td>0.156537</td>
          <td>26.170222</td>
          <td>0.176892</td>
          <td>25.685061</td>
          <td>0.216330</td>
          <td>26.096953</td>
          <td>0.610866</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.890847</td>
          <td>0.569482</td>
          <td>26.123988</td>
          <td>0.118441</td>
          <td>26.010815</td>
          <td>0.096596</td>
          <td>25.818361</td>
          <td>0.133343</td>
          <td>26.142114</td>
          <td>0.319650</td>
          <td>25.561643</td>
          <td>0.418840</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.957144</td>
          <td>1.121287</td>
          <td>26.513801</td>
          <td>0.162565</td>
          <td>26.732714</td>
          <td>0.176512</td>
          <td>26.247603</td>
          <td>0.188346</td>
          <td>25.955985</td>
          <td>0.269780</td>
          <td>25.048751</td>
          <td>0.273909</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>26.632444</td>
          <td>0.417552</td>
          <td>26.561649</td>
          <td>0.146052</td>
          <td>26.000582</td>
          <td>0.078897</td>
          <td>25.192522</td>
          <td>0.062986</td>
          <td>24.888290</td>
          <td>0.091988</td>
          <td>24.006290</td>
          <td>0.095048</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.135619</td>
          <td>0.237206</td>
          <td>26.423374</td>
          <td>0.114432</td>
          <td>26.312478</td>
          <td>0.167618</td>
          <td>25.463044</td>
          <td>0.151722</td>
          <td>25.265497</td>
          <td>0.277540</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.614380</td>
          <td>0.871083</td>
          <td>36.805836</td>
          <td>8.219313</td>
          <td>27.788714</td>
          <td>0.384214</td>
          <td>25.922342</td>
          <td>0.130221</td>
          <td>25.106835</td>
          <td>0.120776</td>
          <td>24.248830</td>
          <td>0.127742</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.796852</td>
          <td>1.767304</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.232298</td>
          <td>0.281178</td>
          <td>26.445215</td>
          <td>0.234176</td>
          <td>25.464467</td>
          <td>0.188707</td>
          <td>27.327867</td>
          <td>1.353248</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.580397</td>
          <td>0.401542</td>
          <td>26.333473</td>
          <td>0.120056</td>
          <td>25.898690</td>
          <td>0.072197</td>
          <td>25.696730</td>
          <td>0.098420</td>
          <td>25.733093</td>
          <td>0.190992</td>
          <td>25.375021</td>
          <td>0.303341</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>32.217004</td>
          <td>4.837008</td>
          <td>26.001252</td>
          <td>0.096109</td>
          <td>25.456305</td>
          <td>0.052739</td>
          <td>25.020979</td>
          <td>0.058788</td>
          <td>24.839344</td>
          <td>0.095318</td>
          <td>24.698104</td>
          <td>0.187081</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.722668</td>
          <td>0.169939</td>
          <td>25.954317</td>
          <td>0.076995</td>
          <td>25.185755</td>
          <td>0.063704</td>
          <td>24.817732</td>
          <td>0.087879</td>
          <td>24.023472</td>
          <td>0.098143</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.168198</td>
          <td>0.635688</td>
          <td>26.675566</td>
          <td>0.167817</td>
          <td>26.359289</td>
          <td>0.113493</td>
          <td>26.008028</td>
          <td>0.135580</td>
          <td>25.964583</td>
          <td>0.242346</td>
          <td>25.730270</td>
          <td>0.418766</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.715175</td>
          <td>0.215910</td>
          <td>26.203899</td>
          <td>0.118399</td>
          <td>26.188763</td>
          <td>0.104401</td>
          <td>25.699960</td>
          <td>0.111040</td>
          <td>26.060860</td>
          <td>0.278735</td>
          <td>25.252132</td>
          <td>0.305890</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.213160</td>
          <td>0.308144</td>
          <td>26.991755</td>
          <td>0.217199</td>
          <td>26.640794</td>
          <td>0.143414</td>
          <td>26.314270</td>
          <td>0.174467</td>
          <td>25.751061</td>
          <td>0.200980</td>
          <td>25.464360</td>
          <td>0.337425</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
