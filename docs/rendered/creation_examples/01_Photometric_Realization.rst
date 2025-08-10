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

    <pzflow.flow.Flow at 0x7fb5a1461120>



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
    0      23.994413  0.031975  0.027863  
    1      25.391064  0.050658  0.042538  
    2      24.304707  0.017589  0.009817  
    3      25.291103  0.136450  0.123133  
    4      25.096743  0.014963  0.013584  
    ...          ...       ...       ...  
    99995  24.737946  0.131469  0.098780  
    99996  24.224169  0.084246  0.067505  
    99997  25.613836  0.118881  0.100811  
    99998  25.274899  0.081592  0.079046  
    99999  25.699642  0.030898  0.024828  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>28.526974</td>
          <td>1.415150</td>
          <td>26.743703</td>
          <td>0.170623</td>
          <td>25.945445</td>
          <td>0.075137</td>
          <td>25.249107</td>
          <td>0.066217</td>
          <td>24.743334</td>
          <td>0.080955</td>
          <td>23.939799</td>
          <td>0.089643</td>
          <td>0.031975</td>
          <td>0.027863</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.625361</td>
          <td>0.351907</td>
          <td>26.510056</td>
          <td>0.123276</td>
          <td>26.258728</td>
          <td>0.159950</td>
          <td>25.591573</td>
          <td>0.169179</td>
          <td>25.446673</td>
          <td>0.320803</td>
          <td>0.050658</td>
          <td>0.042538</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.288904</td>
          <td>0.672544</td>
          <td>28.547930</td>
          <td>0.693981</td>
          <td>28.080966</td>
          <td>0.447454</td>
          <td>26.003250</td>
          <td>0.128378</td>
          <td>25.027444</td>
          <td>0.103911</td>
          <td>24.187670</td>
          <td>0.111382</td>
          <td>0.017589</td>
          <td>0.009817</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.988568</td>
          <td>1.769415</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.499523</td>
          <td>0.283725</td>
          <td>26.261318</td>
          <td>0.160304</td>
          <td>25.346626</td>
          <td>0.137135</td>
          <td>25.830326</td>
          <td>0.432493</td>
          <td>0.136450</td>
          <td>0.123133</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.116702</td>
          <td>0.278069</td>
          <td>26.139796</td>
          <td>0.101284</td>
          <td>25.857916</td>
          <td>0.069539</td>
          <td>25.611211</td>
          <td>0.091165</td>
          <td>25.590418</td>
          <td>0.169012</td>
          <td>24.804412</td>
          <td>0.189221</td>
          <td>0.014963</td>
          <td>0.013584</td>
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
          <td>26.381406</td>
          <td>0.125001</td>
          <td>25.515923</td>
          <td>0.051343</td>
          <td>25.080221</td>
          <td>0.057005</td>
          <td>24.896127</td>
          <td>0.092611</td>
          <td>24.644118</td>
          <td>0.165162</td>
          <td>0.131469</td>
          <td>0.098780</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.221249</td>
          <td>0.641851</td>
          <td>26.456476</td>
          <td>0.133388</td>
          <td>25.995488</td>
          <td>0.078533</td>
          <td>25.158333</td>
          <td>0.061097</td>
          <td>24.825593</td>
          <td>0.087041</td>
          <td>24.129668</td>
          <td>0.105882</td>
          <td>0.084246</td>
          <td>0.067505</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.713860</td>
          <td>0.444141</td>
          <td>26.559307</td>
          <td>0.145742</td>
          <td>26.360832</td>
          <td>0.108255</td>
          <td>26.284417</td>
          <td>0.163497</td>
          <td>25.661256</td>
          <td>0.179494</td>
          <td>25.855650</td>
          <td>0.440875</td>
          <td>0.118881</td>
          <td>0.100811</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.410325</td>
          <td>0.730258</td>
          <td>26.314299</td>
          <td>0.117929</td>
          <td>25.997481</td>
          <td>0.078671</td>
          <td>25.887353</td>
          <td>0.116085</td>
          <td>25.662787</td>
          <td>0.179727</td>
          <td>25.520305</td>
          <td>0.340106</td>
          <td>0.081592</td>
          <td>0.079046</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.519584</td>
          <td>1.409774</td>
          <td>26.844777</td>
          <td>0.185882</td>
          <td>26.504591</td>
          <td>0.122692</td>
          <td>26.530808</td>
          <td>0.201422</td>
          <td>25.775744</td>
          <td>0.197707</td>
          <td>24.820313</td>
          <td>0.191776</td>
          <td>0.030898</td>
          <td>0.024828</td>
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
          <td>26.302877</td>
          <td>0.359536</td>
          <td>26.556746</td>
          <td>0.167590</td>
          <td>26.012216</td>
          <td>0.094008</td>
          <td>25.235542</td>
          <td>0.077784</td>
          <td>24.665398</td>
          <td>0.089147</td>
          <td>23.855270</td>
          <td>0.098665</td>
          <td>0.031975</td>
          <td>0.027863</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.031535</td>
          <td>1.918632</td>
          <td>26.785736</td>
          <td>0.204116</td>
          <td>26.624906</td>
          <td>0.160616</td>
          <td>26.373789</td>
          <td>0.208864</td>
          <td>26.302548</td>
          <td>0.355145</td>
          <td>25.872084</td>
          <td>0.517383</td>
          <td>0.050658</td>
          <td>0.042538</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.173133</td>
          <td>1.259571</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.757415</td>
          <td>0.819297</td>
          <td>25.977636</td>
          <td>0.148256</td>
          <td>25.125733</td>
          <td>0.132893</td>
          <td>24.152084</td>
          <td>0.127482</td>
          <td>0.017589</td>
          <td>0.009817</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.931403</td>
          <td>1.129722</td>
          <td>30.228024</td>
          <td>1.970609</td>
          <td>27.380745</td>
          <td>0.314187</td>
          <td>26.203655</td>
          <td>0.189638</td>
          <td>25.797580</td>
          <td>0.246934</td>
          <td>25.315786</td>
          <td>0.353429</td>
          <td>0.136450</td>
          <td>0.123133</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.269845</td>
          <td>0.349783</td>
          <td>25.991382</td>
          <td>0.102644</td>
          <td>25.939359</td>
          <td>0.087970</td>
          <td>25.560021</td>
          <td>0.103207</td>
          <td>25.094898</td>
          <td>0.129396</td>
          <td>24.833044</td>
          <td>0.227363</td>
          <td>0.014963</td>
          <td>0.013584</td>
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
          <td>26.309291</td>
          <td>0.371552</td>
          <td>26.738335</td>
          <td>0.202450</td>
          <td>25.416091</td>
          <td>0.057857</td>
          <td>25.200871</td>
          <td>0.078711</td>
          <td>24.769729</td>
          <td>0.101756</td>
          <td>24.653269</td>
          <td>0.204110</td>
          <td>0.131469</td>
          <td>0.098780</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.871603</td>
          <td>0.557456</td>
          <td>26.930563</td>
          <td>0.232668</td>
          <td>25.995752</td>
          <td>0.094191</td>
          <td>25.234943</td>
          <td>0.079081</td>
          <td>25.033372</td>
          <td>0.124969</td>
          <td>24.225285</td>
          <td>0.138408</td>
          <td>0.084246</td>
          <td>0.067505</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.342249</td>
          <td>0.380092</td>
          <td>27.111444</td>
          <td>0.274537</td>
          <td>26.656624</td>
          <td>0.170268</td>
          <td>26.407744</td>
          <td>0.221780</td>
          <td>26.420098</td>
          <td>0.400390</td>
          <td>26.868494</td>
          <td>1.029854</td>
          <td>0.118881</td>
          <td>0.100811</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.016135</td>
          <td>0.289951</td>
          <td>26.191060</td>
          <td>0.124423</td>
          <td>26.010434</td>
          <td>0.095610</td>
          <td>25.871510</td>
          <td>0.138201</td>
          <td>25.494425</td>
          <td>0.185873</td>
          <td>25.364698</td>
          <td>0.356384</td>
          <td>0.081592</td>
          <td>0.079046</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.455941</td>
          <td>0.404721</td>
          <td>26.836742</td>
          <td>0.212148</td>
          <td>26.664629</td>
          <td>0.165392</td>
          <td>26.514461</td>
          <td>0.233717</td>
          <td>26.037711</td>
          <td>0.286322</td>
          <td>29.725280</td>
          <td>3.361636</td>
          <td>0.030898</td>
          <td>0.024828</td>
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
          <td>27.041359</td>
          <td>0.221386</td>
          <td>25.987942</td>
          <td>0.078965</td>
          <td>25.110497</td>
          <td>0.059314</td>
          <td>24.674895</td>
          <td>0.077139</td>
          <td>23.955990</td>
          <td>0.092077</td>
          <td>0.031975</td>
          <td>0.027863</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.642475</td>
          <td>0.364727</td>
          <td>26.578007</td>
          <td>0.134518</td>
          <td>26.320098</td>
          <td>0.173553</td>
          <td>25.843636</td>
          <td>0.215100</td>
          <td>25.473845</td>
          <td>0.336785</td>
          <td>0.050658</td>
          <td>0.042538</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.439345</td>
          <td>3.052685</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.557164</td>
          <td>0.633920</td>
          <td>25.799465</td>
          <td>0.107829</td>
          <td>24.854427</td>
          <td>0.089524</td>
          <td>24.320149</td>
          <td>0.125339</td>
          <td>0.017589</td>
          <td>0.009817</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.926398</td>
          <td>0.210928</td>
          <td>25.914522</td>
          <td>0.144029</td>
          <td>25.158601</td>
          <td>0.140117</td>
          <td>24.872042</td>
          <td>0.240568</td>
          <td>0.136450</td>
          <td>0.123133</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.389288</td>
          <td>0.346349</td>
          <td>26.359846</td>
          <td>0.122977</td>
          <td>25.969020</td>
          <td>0.076933</td>
          <td>25.572086</td>
          <td>0.088338</td>
          <td>25.891011</td>
          <td>0.218308</td>
          <td>25.083462</td>
          <td>0.239526</td>
          <td>0.014963</td>
          <td>0.013584</td>
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
          <td>28.597591</td>
          <td>1.562707</td>
          <td>26.469321</td>
          <td>0.154137</td>
          <td>25.392690</td>
          <td>0.053799</td>
          <td>25.128839</td>
          <td>0.070009</td>
          <td>25.118524</td>
          <td>0.131015</td>
          <td>24.639381</td>
          <td>0.191820</td>
          <td>0.131469</td>
          <td>0.098780</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.040327</td>
          <td>0.589119</td>
          <td>26.568251</td>
          <td>0.156331</td>
          <td>26.121042</td>
          <td>0.094386</td>
          <td>25.122292</td>
          <td>0.063932</td>
          <td>24.683319</td>
          <td>0.082626</td>
          <td>24.257464</td>
          <td>0.127564</td>
          <td>0.084246</td>
          <td>0.067505</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.258115</td>
          <td>0.341748</td>
          <td>26.681062</td>
          <td>0.182527</td>
          <td>26.307934</td>
          <td>0.118954</td>
          <td>26.389235</td>
          <td>0.205915</td>
          <td>26.168405</td>
          <td>0.311511</td>
          <td>25.647761</td>
          <td>0.427179</td>
          <td>0.118881</td>
          <td>0.100811</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.439478</td>
          <td>0.378353</td>
          <td>26.097409</td>
          <td>0.104709</td>
          <td>26.175224</td>
          <td>0.099720</td>
          <td>25.923325</td>
          <td>0.130169</td>
          <td>25.759403</td>
          <td>0.210573</td>
          <td>26.037484</td>
          <td>0.541233</td>
          <td>0.081592</td>
          <td>0.079046</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.886855</td>
          <td>1.694992</td>
          <td>26.761679</td>
          <td>0.174791</td>
          <td>26.418295</td>
          <td>0.115018</td>
          <td>26.171910</td>
          <td>0.150100</td>
          <td>25.886028</td>
          <td>0.219012</td>
          <td>26.102869</td>
          <td>0.534577</td>
          <td>0.030898</td>
          <td>0.024828</td>
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
