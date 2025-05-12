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

    <pzflow.flow.Flow at 0x7f9ca149e980>



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
          <td>27.699896</td>
          <td>0.881577</td>
          <td>26.709280</td>
          <td>0.165697</td>
          <td>26.013031</td>
          <td>0.079759</td>
          <td>25.198645</td>
          <td>0.063320</td>
          <td>24.767119</td>
          <td>0.082671</td>
          <td>24.118725</td>
          <td>0.104874</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.771883</td>
          <td>1.598773</td>
          <td>27.800848</td>
          <td>0.403348</td>
          <td>26.835043</td>
          <td>0.163086</td>
          <td>26.488024</td>
          <td>0.194305</td>
          <td>26.062338</td>
          <td>0.250904</td>
          <td>24.893786</td>
          <td>0.203996</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.365379</td>
          <td>0.611607</td>
          <td>28.089311</td>
          <td>0.450278</td>
          <td>26.172366</td>
          <td>0.148541</td>
          <td>24.918765</td>
          <td>0.094471</td>
          <td>24.422726</td>
          <td>0.136586</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.055525</td>
          <td>0.571053</td>
          <td>28.995644</td>
          <td>0.928368</td>
          <td>27.340375</td>
          <td>0.249169</td>
          <td>26.441177</td>
          <td>0.186777</td>
          <td>25.419309</td>
          <td>0.145995</td>
          <td>25.341829</td>
          <td>0.294951</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.812291</td>
          <td>0.216528</td>
          <td>26.160716</td>
          <td>0.103153</td>
          <td>25.928794</td>
          <td>0.074039</td>
          <td>25.675258</td>
          <td>0.096440</td>
          <td>25.413636</td>
          <td>0.145284</td>
          <td>25.015949</td>
          <td>0.225890</td>
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
          <td>27.707539</td>
          <td>0.885831</td>
          <td>26.485036</td>
          <td>0.136716</td>
          <td>25.396129</td>
          <td>0.046162</td>
          <td>25.026328</td>
          <td>0.054342</td>
          <td>24.737634</td>
          <td>0.080549</td>
          <td>24.646147</td>
          <td>0.165448</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.652775</td>
          <td>0.424042</td>
          <td>26.884834</td>
          <td>0.192269</td>
          <td>26.041596</td>
          <td>0.081794</td>
          <td>25.200172</td>
          <td>0.063406</td>
          <td>24.625802</td>
          <td>0.072972</td>
          <td>24.102299</td>
          <td>0.103378</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.838433</td>
          <td>0.960736</td>
          <td>26.680002</td>
          <td>0.161612</td>
          <td>26.523376</td>
          <td>0.124709</td>
          <td>26.156635</td>
          <td>0.146547</td>
          <td>25.913308</td>
          <td>0.221817</td>
          <td>25.258615</td>
          <td>0.275743</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.066672</td>
          <td>0.266992</td>
          <td>26.173244</td>
          <td>0.104289</td>
          <td>26.285881</td>
          <td>0.101387</td>
          <td>25.815571</td>
          <td>0.109043</td>
          <td>25.465895</td>
          <td>0.151953</td>
          <td>25.601873</td>
          <td>0.362636</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.070735</td>
          <td>0.577293</td>
          <td>26.749087</td>
          <td>0.171406</td>
          <td>26.588169</td>
          <td>0.131909</td>
          <td>26.271039</td>
          <td>0.161641</td>
          <td>25.689009</td>
          <td>0.183761</td>
          <td>25.626681</td>
          <td>0.369734</td>
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
          <td>28.372795</td>
          <td>1.400050</td>
          <td>26.761439</td>
          <td>0.198734</td>
          <td>26.270195</td>
          <td>0.117432</td>
          <td>25.138903</td>
          <td>0.071193</td>
          <td>24.638275</td>
          <td>0.086782</td>
          <td>24.096634</td>
          <td>0.121416</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.608621</td>
          <td>0.453769</td>
          <td>26.835285</td>
          <td>0.211449</td>
          <td>26.492666</td>
          <td>0.142396</td>
          <td>26.425493</td>
          <td>0.216550</td>
          <td>25.770342</td>
          <td>0.229481</td>
          <td>25.554259</td>
          <td>0.405008</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.105234</td>
          <td>2.881115</td>
          <td>29.158824</td>
          <td>1.142064</td>
          <td>28.049344</td>
          <td>0.510658</td>
          <td>26.193401</td>
          <td>0.182120</td>
          <td>25.158409</td>
          <td>0.139630</td>
          <td>24.291250</td>
          <td>0.146919</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.094268</td>
          <td>0.594853</td>
          <td>27.270558</td>
          <td>0.290970</td>
          <td>26.372440</td>
          <td>0.221222</td>
          <td>26.121333</td>
          <td>0.324737</td>
          <td>25.503315</td>
          <td>0.413534</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.791973</td>
          <td>0.519887</td>
          <td>26.191677</td>
          <td>0.122161</td>
          <td>25.768068</td>
          <td>0.075614</td>
          <td>25.510125</td>
          <td>0.098760</td>
          <td>25.635049</td>
          <td>0.205030</td>
          <td>25.199896</td>
          <td>0.306640</td>
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
          <td>28.158273</td>
          <td>1.261486</td>
          <td>26.273238</td>
          <td>0.133530</td>
          <td>25.320477</td>
          <td>0.051937</td>
          <td>25.107642</td>
          <td>0.070786</td>
          <td>24.802906</td>
          <td>0.102397</td>
          <td>24.586818</td>
          <td>0.188736</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.142979</td>
          <td>1.241033</td>
          <td>26.752670</td>
          <td>0.197986</td>
          <td>26.073063</td>
          <td>0.099275</td>
          <td>25.166545</td>
          <td>0.073272</td>
          <td>24.848412</td>
          <td>0.104784</td>
          <td>24.190088</td>
          <td>0.132214</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.593042</td>
          <td>0.452179</td>
          <td>26.937894</td>
          <td>0.232766</td>
          <td>26.420267</td>
          <td>0.135442</td>
          <td>26.248041</td>
          <td>0.188931</td>
          <td>25.741818</td>
          <td>0.226790</td>
          <td>27.197448</td>
          <td>1.221626</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.961933</td>
          <td>0.279487</td>
          <td>26.462771</td>
          <td>0.158592</td>
          <td>26.100612</td>
          <td>0.104498</td>
          <td>25.812037</td>
          <td>0.132617</td>
          <td>25.627390</td>
          <td>0.209839</td>
          <td>25.635424</td>
          <td>0.442990</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.037587</td>
          <td>0.292647</td>
          <td>27.085595</td>
          <td>0.262213</td>
          <td>26.535583</td>
          <td>0.149176</td>
          <td>26.337507</td>
          <td>0.203146</td>
          <td>27.070473</td>
          <td>0.630197</td>
          <td>26.057771</td>
          <td>0.592834</td>
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
          <td>26.856179</td>
          <td>0.494003</td>
          <td>26.497299</td>
          <td>0.138185</td>
          <td>26.033734</td>
          <td>0.081240</td>
          <td>25.132401</td>
          <td>0.059715</td>
          <td>24.720850</td>
          <td>0.079375</td>
          <td>23.934009</td>
          <td>0.089199</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.094132</td>
          <td>0.503376</td>
          <td>26.896170</td>
          <td>0.171961</td>
          <td>26.306848</td>
          <td>0.166816</td>
          <td>26.382892</td>
          <td>0.325445</td>
          <td>25.733962</td>
          <td>0.402126</td>
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
          <td>28.036562</td>
          <td>0.464130</td>
          <td>25.975617</td>
          <td>0.136357</td>
          <td>24.926784</td>
          <td>0.103229</td>
          <td>24.308778</td>
          <td>0.134541</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.331337</td>
          <td>2.963741</td>
          <td>27.372886</td>
          <td>0.314860</td>
          <td>26.255110</td>
          <td>0.199852</td>
          <td>25.049138</td>
          <td>0.132319</td>
          <td>25.703677</td>
          <td>0.479556</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.617290</td>
          <td>0.413065</td>
          <td>26.254213</td>
          <td>0.112059</td>
          <td>25.994607</td>
          <td>0.078584</td>
          <td>25.673287</td>
          <td>0.096417</td>
          <td>25.295653</td>
          <td>0.131409</td>
          <td>25.105805</td>
          <td>0.243661</td>
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
          <td>26.465344</td>
          <td>0.385578</td>
          <td>26.191291</td>
          <td>0.113461</td>
          <td>25.471969</td>
          <td>0.053478</td>
          <td>25.168672</td>
          <td>0.067012</td>
          <td>24.899403</td>
          <td>0.100471</td>
          <td>24.687068</td>
          <td>0.185345</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.762860</td>
          <td>0.465347</td>
          <td>26.655757</td>
          <td>0.160521</td>
          <td>26.117753</td>
          <td>0.088928</td>
          <td>25.214498</td>
          <td>0.065348</td>
          <td>24.726553</td>
          <td>0.081096</td>
          <td>24.058070</td>
          <td>0.101163</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.402397</td>
          <td>0.360249</td>
          <td>26.523305</td>
          <td>0.147335</td>
          <td>26.460881</td>
          <td>0.123975</td>
          <td>25.989680</td>
          <td>0.133448</td>
          <td>25.998891</td>
          <td>0.249289</td>
          <td>25.151021</td>
          <td>0.264765</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.408365</td>
          <td>0.377589</td>
          <td>26.207416</td>
          <td>0.118761</td>
          <td>26.147838</td>
          <td>0.100728</td>
          <td>26.078284</td>
          <td>0.154024</td>
          <td>26.226028</td>
          <td>0.318355</td>
          <td>25.136155</td>
          <td>0.278566</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.524914</td>
          <td>0.393715</td>
          <td>27.094439</td>
          <td>0.236517</td>
          <td>26.548395</td>
          <td>0.132426</td>
          <td>26.550127</td>
          <td>0.212819</td>
          <td>25.689963</td>
          <td>0.190910</td>
          <td>27.794095</td>
          <td>1.535677</td>
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
