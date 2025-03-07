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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fb1e7908250>



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
          <td>27.766742</td>
          <td>0.919232</td>
          <td>26.289384</td>
          <td>0.115402</td>
          <td>26.231387</td>
          <td>0.096658</td>
          <td>25.180637</td>
          <td>0.062317</td>
          <td>24.848264</td>
          <td>0.088795</td>
          <td>24.307744</td>
          <td>0.123648</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.591017</td>
          <td>1.462158</td>
          <td>27.149809</td>
          <td>0.239816</td>
          <td>26.694235</td>
          <td>0.144547</td>
          <td>26.073357</td>
          <td>0.136402</td>
          <td>25.621809</td>
          <td>0.173585</td>
          <td>25.023590</td>
          <td>0.227327</td>
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
          <td>27.554640</td>
          <td>0.296639</td>
          <td>26.043130</td>
          <td>0.132886</td>
          <td>25.086517</td>
          <td>0.109416</td>
          <td>24.121110</td>
          <td>0.105093</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.717122</td>
          <td>0.378072</td>
          <td>27.369201</td>
          <td>0.255136</td>
          <td>26.230954</td>
          <td>0.156194</td>
          <td>25.559742</td>
          <td>0.164652</td>
          <td>25.367482</td>
          <td>0.301103</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.731688</td>
          <td>0.450151</td>
          <td>26.118164</td>
          <td>0.099385</td>
          <td>26.041357</td>
          <td>0.081777</td>
          <td>25.608744</td>
          <td>0.090968</td>
          <td>25.678329</td>
          <td>0.182108</td>
          <td>24.613365</td>
          <td>0.160884</td>
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
          <td>26.902230</td>
          <td>0.511005</td>
          <td>26.311795</td>
          <td>0.117673</td>
          <td>25.427783</td>
          <td>0.047478</td>
          <td>25.136987</td>
          <td>0.059950</td>
          <td>24.664667</td>
          <td>0.075523</td>
          <td>24.712278</td>
          <td>0.175025</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.620902</td>
          <td>0.838390</td>
          <td>26.369505</td>
          <td>0.123718</td>
          <td>26.032924</td>
          <td>0.081171</td>
          <td>25.157717</td>
          <td>0.061063</td>
          <td>24.862674</td>
          <td>0.089928</td>
          <td>24.308276</td>
          <td>0.123705</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.042391</td>
          <td>0.565705</td>
          <td>26.825731</td>
          <td>0.182914</td>
          <td>26.350252</td>
          <td>0.107259</td>
          <td>26.168908</td>
          <td>0.148101</td>
          <td>26.013571</td>
          <td>0.241030</td>
          <td>25.448848</td>
          <td>0.321360</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.143054</td>
          <td>0.284064</td>
          <td>26.164585</td>
          <td>0.103503</td>
          <td>25.930108</td>
          <td>0.074125</td>
          <td>25.741631</td>
          <td>0.102217</td>
          <td>25.771689</td>
          <td>0.197034</td>
          <td>24.996635</td>
          <td>0.222293</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.196456</td>
          <td>0.630865</td>
          <td>26.494645</td>
          <td>0.137854</td>
          <td>26.774414</td>
          <td>0.154847</td>
          <td>26.320718</td>
          <td>0.168636</td>
          <td>25.583075</td>
          <td>0.167959</td>
          <td>25.848375</td>
          <td>0.438454</td>
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
          <td>26.480870</td>
          <td>0.156672</td>
          <td>26.091525</td>
          <td>0.100474</td>
          <td>25.168800</td>
          <td>0.073100</td>
          <td>24.718875</td>
          <td>0.093154</td>
          <td>24.052496</td>
          <td>0.116846</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.491363</td>
          <td>0.415180</td>
          <td>27.403400</td>
          <td>0.335866</td>
          <td>26.462340</td>
          <td>0.138724</td>
          <td>26.225225</td>
          <td>0.183018</td>
          <td>25.510323</td>
          <td>0.184575</td>
          <td>25.236633</td>
          <td>0.315752</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.263081</td>
          <td>0.193156</td>
          <td>25.256866</td>
          <td>0.151962</td>
          <td>24.292063</td>
          <td>0.147021</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.445856</td>
          <td>0.419414</td>
          <td>27.752117</td>
          <td>0.463489</td>
          <td>27.442397</td>
          <td>0.333839</td>
          <td>26.323764</td>
          <td>0.212425</td>
          <td>25.311472</td>
          <td>0.166318</td>
          <td>26.215624</td>
          <td>0.692945</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.615589</td>
          <td>0.456186</td>
          <td>26.124343</td>
          <td>0.115223</td>
          <td>26.055110</td>
          <td>0.097350</td>
          <td>25.714419</td>
          <td>0.118045</td>
          <td>25.366648</td>
          <td>0.163390</td>
          <td>24.826517</td>
          <td>0.226060</td>
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
          <td>26.525633</td>
          <td>0.432057</td>
          <td>26.356395</td>
          <td>0.143446</td>
          <td>25.339271</td>
          <td>0.052810</td>
          <td>25.108541</td>
          <td>0.070842</td>
          <td>24.965795</td>
          <td>0.118031</td>
          <td>25.046831</td>
          <td>0.276343</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.314280</td>
          <td>0.363038</td>
          <td>26.887860</td>
          <td>0.221669</td>
          <td>26.018412</td>
          <td>0.094630</td>
          <td>25.228453</td>
          <td>0.077391</td>
          <td>24.805716</td>
          <td>0.100943</td>
          <td>24.081168</td>
          <td>0.120305</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.213197</td>
          <td>0.337298</td>
          <td>27.072221</td>
          <td>0.259963</td>
          <td>26.211734</td>
          <td>0.113028</td>
          <td>26.321091</td>
          <td>0.200912</td>
          <td>25.523759</td>
          <td>0.188951</td>
          <td>25.175819</td>
          <td>0.304310</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.685031</td>
          <td>0.490222</td>
          <td>26.496964</td>
          <td>0.163288</td>
          <td>26.221055</td>
          <td>0.116075</td>
          <td>26.083446</td>
          <td>0.167405</td>
          <td>25.386918</td>
          <td>0.171318</td>
          <td>25.031578</td>
          <td>0.275651</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.411229</td>
          <td>0.392968</td>
          <td>26.411400</td>
          <td>0.148933</td>
          <td>26.725995</td>
          <td>0.175508</td>
          <td>26.192954</td>
          <td>0.179840</td>
          <td>25.881379</td>
          <td>0.253818</td>
          <td>25.340538</td>
          <td>0.346023</td>
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
          <td>27.383463</td>
          <td>0.717248</td>
          <td>26.943849</td>
          <td>0.202068</td>
          <td>25.997511</td>
          <td>0.078684</td>
          <td>25.105094</td>
          <td>0.058286</td>
          <td>24.591338</td>
          <td>0.070790</td>
          <td>24.140813</td>
          <td>0.106932</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.438656</td>
          <td>0.303626</td>
          <td>26.759562</td>
          <td>0.153030</td>
          <td>26.145364</td>
          <td>0.145275</td>
          <td>26.134997</td>
          <td>0.266517</td>
          <td>25.879599</td>
          <td>0.449301</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.097255</td>
          <td>0.535284</td>
          <td>30.236645</td>
          <td>1.757146</td>
          <td>26.009057</td>
          <td>0.140348</td>
          <td>24.889757</td>
          <td>0.099937</td>
          <td>24.229899</td>
          <td>0.125664</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.493590</td>
          <td>1.530706</td>
          <td>30.409060</td>
          <td>2.132383</td>
          <td>27.177186</td>
          <td>0.268861</td>
          <td>25.902553</td>
          <td>0.148118</td>
          <td>25.429224</td>
          <td>0.183171</td>
          <td>25.665956</td>
          <td>0.466245</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.784582</td>
          <td>0.468761</td>
          <td>26.064544</td>
          <td>0.094941</td>
          <td>25.992113</td>
          <td>0.078411</td>
          <td>25.820951</td>
          <td>0.109719</td>
          <td>25.360101</td>
          <td>0.138931</td>
          <td>24.981132</td>
          <td>0.219750</td>
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
          <td>27.233183</td>
          <td>0.676153</td>
          <td>26.320467</td>
          <td>0.126926</td>
          <td>25.413952</td>
          <td>0.050793</td>
          <td>25.074854</td>
          <td>0.061665</td>
          <td>24.725693</td>
          <td>0.086256</td>
          <td>24.633376</td>
          <td>0.177107</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.274025</td>
          <td>0.318876</td>
          <td>26.582392</td>
          <td>0.150755</td>
          <td>26.074630</td>
          <td>0.085616</td>
          <td>25.228100</td>
          <td>0.066141</td>
          <td>24.736718</td>
          <td>0.081826</td>
          <td>24.290753</td>
          <td>0.123915</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.499720</td>
          <td>0.388574</td>
          <td>27.051088</td>
          <td>0.230084</td>
          <td>26.487244</td>
          <td>0.126842</td>
          <td>26.194610</td>
          <td>0.159158</td>
          <td>25.906580</td>
          <td>0.231001</td>
          <td>25.534569</td>
          <td>0.359922</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.339641</td>
          <td>0.357885</td>
          <td>26.250118</td>
          <td>0.123244</td>
          <td>26.008962</td>
          <td>0.089170</td>
          <td>25.969343</td>
          <td>0.140260</td>
          <td>25.566027</td>
          <td>0.184910</td>
          <td>25.295685</td>
          <td>0.316737</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.995618</td>
          <td>1.075104</td>
          <td>26.585203</td>
          <td>0.154034</td>
          <td>26.488103</td>
          <td>0.125690</td>
          <td>26.389479</td>
          <td>0.185946</td>
          <td>25.825532</td>
          <td>0.213910</td>
          <td>26.492856</td>
          <td>0.719232</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
