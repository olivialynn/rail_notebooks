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

    <pzflow.flow.Flow at 0x7f97ad8bfeb0>



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
    0      1.398945  27.667538  26.723339  26.032640  25.178589  24.695959   
    1      2.285624  28.786999  27.476589  26.640173  26.259747  25.865671   
    2      1.495130  30.011343  29.789326  28.200378  26.014816  25.030161   
    3      0.842595  29.306242  28.721798  27.353014  26.256908  25.529823   
    4      1.588960  26.273870  26.115385  25.950439  25.687403  25.466604   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270809  26.371513  25.436861  25.077417  24.852785   
    99996  1.481047  27.478111  26.735254  26.042774  25.204937  24.825092   
    99997  2.023549  26.990149  26.714739  26.377953  26.250345  25.917372   
    99998  1.548203  26.367432  26.206882  26.087980  25.876928  25.715893   
    99999  1.739491  26.881981  26.773064  26.553120  26.319618  25.955980   
    
                   y     major     minor  
    0      23.994417  0.003319  0.002869  
    1      25.391062  0.008733  0.007945  
    2      24.304695  0.103938  0.052162  
    3      25.291105  0.147522  0.143359  
    4      25.096741  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737953  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613838  0.073146  0.047825  
    99998  25.274897  0.100551  0.094662  
    99999  25.699638  0.059611  0.049181  
    
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
          <td>1.398945</td>
          <td>26.702144</td>
          <td>0.440227</td>
          <td>26.451153</td>
          <td>0.132776</td>
          <td>25.899422</td>
          <td>0.072140</td>
          <td>25.224203</td>
          <td>0.064771</td>
          <td>24.826408</td>
          <td>0.087103</td>
          <td>24.151274</td>
          <td>0.107900</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.169527</td>
          <td>0.619090</td>
          <td>28.915063</td>
          <td>0.882799</td>
          <td>26.576108</td>
          <td>0.130539</td>
          <td>26.168370</td>
          <td>0.148032</td>
          <td>25.709356</td>
          <td>0.186949</td>
          <td>25.131439</td>
          <td>0.248513</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.369460</td>
          <td>0.710456</td>
          <td>29.085632</td>
          <td>0.980982</td>
          <td>28.269120</td>
          <td>0.514687</td>
          <td>26.246486</td>
          <td>0.158284</td>
          <td>24.858313</td>
          <td>0.089583</td>
          <td>24.248432</td>
          <td>0.117436</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.863215</td>
          <td>0.423071</td>
          <td>27.621960</td>
          <td>0.313104</td>
          <td>25.955480</td>
          <td>0.123168</td>
          <td>25.523669</td>
          <td>0.159658</td>
          <td>25.468197</td>
          <td>0.326346</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.512808</td>
          <td>0.380794</td>
          <td>26.141839</td>
          <td>0.101465</td>
          <td>26.090729</td>
          <td>0.085415</td>
          <td>25.829123</td>
          <td>0.110340</td>
          <td>25.539433</td>
          <td>0.161823</td>
          <td>24.949625</td>
          <td>0.213753</td>
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
          <td>26.599589</td>
          <td>0.407154</td>
          <td>26.689465</td>
          <td>0.162922</td>
          <td>25.377187</td>
          <td>0.045393</td>
          <td>24.975205</td>
          <td>0.051930</td>
          <td>24.819586</td>
          <td>0.086582</td>
          <td>24.614697</td>
          <td>0.161067</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.647832</td>
          <td>0.852953</td>
          <td>26.790003</td>
          <td>0.177463</td>
          <td>25.960694</td>
          <td>0.076156</td>
          <td>25.279251</td>
          <td>0.068009</td>
          <td>24.812108</td>
          <td>0.086014</td>
          <td>24.230805</td>
          <td>0.115648</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.630667</td>
          <td>0.843651</td>
          <td>26.567470</td>
          <td>0.146768</td>
          <td>26.408405</td>
          <td>0.112843</td>
          <td>26.272857</td>
          <td>0.161892</td>
          <td>25.727507</td>
          <td>0.189836</td>
          <td>25.691277</td>
          <td>0.388763</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.805296</td>
          <td>0.215271</td>
          <td>26.324550</td>
          <td>0.118984</td>
          <td>25.969659</td>
          <td>0.076762</td>
          <td>25.800348</td>
          <td>0.107603</td>
          <td>26.246964</td>
          <td>0.291613</td>
          <td>25.403904</td>
          <td>0.310030</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.400448</td>
          <td>0.348804</td>
          <td>26.391468</td>
          <td>0.126095</td>
          <td>26.314929</td>
          <td>0.103997</td>
          <td>26.343801</td>
          <td>0.171981</td>
          <td>25.937324</td>
          <td>0.226289</td>
          <td>25.425252</td>
          <td>0.315368</td>
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
          <td>1.398945</td>
          <td>27.124375</td>
          <td>0.658321</td>
          <td>26.764493</td>
          <td>0.199244</td>
          <td>26.201307</td>
          <td>0.110592</td>
          <td>25.110845</td>
          <td>0.069447</td>
          <td>24.627075</td>
          <td>0.085931</td>
          <td>23.970142</td>
          <td>0.108755</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.067437</td>
          <td>0.632905</td>
          <td>27.821287</td>
          <td>0.463523</td>
          <td>26.445722</td>
          <td>0.136749</td>
          <td>26.833336</td>
          <td>0.302445</td>
          <td>25.916661</td>
          <td>0.258881</td>
          <td>25.684308</td>
          <td>0.447168</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>29.235123</td>
          <td>2.101138</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.205936</td>
          <td>0.572064</td>
          <td>26.143052</td>
          <td>0.174510</td>
          <td>24.874225</td>
          <td>0.109123</td>
          <td>24.918321</td>
          <td>0.249103</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.339011</td>
          <td>2.075597</td>
          <td>27.943186</td>
          <td>0.490297</td>
          <td>26.647266</td>
          <td>0.277316</td>
          <td>25.239194</td>
          <td>0.156364</td>
          <td>25.432533</td>
          <td>0.391624</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.046700</td>
          <td>0.292810</td>
          <td>26.201923</td>
          <td>0.123251</td>
          <td>25.945149</td>
          <td>0.088389</td>
          <td>25.669601</td>
          <td>0.113529</td>
          <td>25.344125</td>
          <td>0.160278</td>
          <td>25.190068</td>
          <td>0.304232</td>
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
          <td>28.205870</td>
          <td>1.294402</td>
          <td>26.537433</td>
          <td>0.167474</td>
          <td>25.371252</td>
          <td>0.054330</td>
          <td>25.121024</td>
          <td>0.071628</td>
          <td>25.004308</td>
          <td>0.122048</td>
          <td>24.773287</td>
          <td>0.220664</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.399466</td>
          <td>0.146655</td>
          <td>26.025301</td>
          <td>0.095203</td>
          <td>25.398454</td>
          <td>0.089897</td>
          <td>24.941497</td>
          <td>0.113651</td>
          <td>24.187052</td>
          <td>0.131867</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>29.191589</td>
          <td>2.056937</td>
          <td>26.631095</td>
          <td>0.180034</td>
          <td>26.364444</td>
          <td>0.129061</td>
          <td>26.494353</td>
          <td>0.232143</td>
          <td>26.042010</td>
          <td>0.290010</td>
          <td>26.095662</td>
          <td>0.610311</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.129901</td>
          <td>0.319842</td>
          <td>26.149260</td>
          <td>0.121068</td>
          <td>26.148943</td>
          <td>0.109004</td>
          <td>25.920323</td>
          <td>0.145593</td>
          <td>26.121152</td>
          <td>0.314347</td>
          <td>25.143072</td>
          <td>0.301637</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.411395</td>
          <td>0.148932</td>
          <td>26.748286</td>
          <td>0.178858</td>
          <td>26.453035</td>
          <td>0.223719</td>
          <td>26.137931</td>
          <td>0.312463</td>
          <td>25.353233</td>
          <td>0.349500</td>
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
          <td>1.398945</td>
          <td>28.789925</td>
          <td>1.612780</td>
          <td>26.837397</td>
          <td>0.184746</td>
          <td>26.085458</td>
          <td>0.085030</td>
          <td>25.252934</td>
          <td>0.066451</td>
          <td>24.829876</td>
          <td>0.087381</td>
          <td>23.968290</td>
          <td>0.091929</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.176406</td>
          <td>0.245319</td>
          <td>26.615770</td>
          <td>0.135219</td>
          <td>26.355874</td>
          <td>0.173922</td>
          <td>26.064725</td>
          <td>0.251619</td>
          <td>25.132487</td>
          <td>0.248956</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.045025</td>
          <td>0.515264</td>
          <td>27.772543</td>
          <td>0.379423</td>
          <td>26.319823</td>
          <td>0.183021</td>
          <td>25.151805</td>
          <td>0.125582</td>
          <td>24.221959</td>
          <td>0.124801</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.783660</td>
          <td>0.194017</td>
          <td>26.192248</td>
          <td>0.189553</td>
          <td>25.466055</td>
          <td>0.188960</td>
          <td>26.459179</td>
          <td>0.812441</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.461524</td>
          <td>0.366219</td>
          <td>26.115476</td>
          <td>0.099273</td>
          <td>25.909356</td>
          <td>0.072881</td>
          <td>25.914933</td>
          <td>0.119081</td>
          <td>25.371785</td>
          <td>0.140338</td>
          <td>24.762857</td>
          <td>0.182952</td>
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
          <td>27.933811</td>
          <td>1.056785</td>
          <td>26.432601</td>
          <td>0.139830</td>
          <td>25.495071</td>
          <td>0.054586</td>
          <td>25.043398</td>
          <td>0.059969</td>
          <td>24.885902</td>
          <td>0.099289</td>
          <td>24.610855</td>
          <td>0.173754</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.721693</td>
          <td>0.451192</td>
          <td>26.628010</td>
          <td>0.156760</td>
          <td>26.045124</td>
          <td>0.083419</td>
          <td>25.357294</td>
          <td>0.074154</td>
          <td>24.860012</td>
          <td>0.091209</td>
          <td>24.138073</td>
          <td>0.108494</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.889204</td>
          <td>0.520891</td>
          <td>26.548724</td>
          <td>0.150583</td>
          <td>26.397087</td>
          <td>0.117290</td>
          <td>26.333333</td>
          <td>0.179109</td>
          <td>25.876547</td>
          <td>0.225317</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.484875</td>
          <td>0.400586</td>
          <td>26.058146</td>
          <td>0.104282</td>
          <td>26.055219</td>
          <td>0.092870</td>
          <td>25.990763</td>
          <td>0.142871</td>
          <td>25.853200</td>
          <td>0.235120</td>
          <td>25.078996</td>
          <td>0.265905</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.635922</td>
          <td>0.428639</td>
          <td>26.624844</td>
          <td>0.159346</td>
          <td>26.547454</td>
          <td>0.132318</td>
          <td>25.881988</td>
          <td>0.120309</td>
          <td>26.466313</td>
          <td>0.359639</td>
          <td>25.646468</td>
          <td>0.389101</td>
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
