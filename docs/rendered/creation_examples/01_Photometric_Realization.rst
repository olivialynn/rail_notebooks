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

    <pzflow.flow.Flow at 0x7f1d33f6fee0>



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
          <td>26.999036</td>
          <td>0.548324</td>
          <td>26.913611</td>
          <td>0.196981</td>
          <td>26.066002</td>
          <td>0.083574</td>
          <td>25.085616</td>
          <td>0.057279</td>
          <td>24.732217</td>
          <td>0.080165</td>
          <td>23.939969</td>
          <td>0.089656</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.027010</td>
          <td>0.478637</td>
          <td>26.549763</td>
          <td>0.127595</td>
          <td>26.354839</td>
          <td>0.173602</td>
          <td>25.936082</td>
          <td>0.226055</td>
          <td>25.407146</td>
          <td>0.310836</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.209394</td>
          <td>0.636581</td>
          <td>28.147385</td>
          <td>0.523071</td>
          <td>28.911102</td>
          <td>0.803123</td>
          <td>26.063676</td>
          <td>0.135266</td>
          <td>24.967216</td>
          <td>0.098573</td>
          <td>24.549324</td>
          <td>0.152303</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.117082</td>
          <td>1.873895</td>
          <td>28.764004</td>
          <td>0.801359</td>
          <td>27.086985</td>
          <td>0.201860</td>
          <td>26.376928</td>
          <td>0.176888</td>
          <td>25.843125</td>
          <td>0.209201</td>
          <td>25.063385</td>
          <td>0.234948</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.728650</td>
          <td>0.449122</td>
          <td>26.051212</td>
          <td>0.093722</td>
          <td>25.990866</td>
          <td>0.078213</td>
          <td>25.539321</td>
          <td>0.085576</td>
          <td>25.658034</td>
          <td>0.179004</td>
          <td>24.920318</td>
          <td>0.208580</td>
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
          <td>26.527635</td>
          <td>0.385195</td>
          <td>26.348260</td>
          <td>0.121459</td>
          <td>25.402103</td>
          <td>0.046408</td>
          <td>25.042073</td>
          <td>0.055107</td>
          <td>24.790498</td>
          <td>0.084392</td>
          <td>24.618797</td>
          <td>0.161632</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.653739</td>
          <td>0.158027</td>
          <td>25.999307</td>
          <td>0.078798</td>
          <td>25.143645</td>
          <td>0.060306</td>
          <td>24.897044</td>
          <td>0.092686</td>
          <td>24.089809</td>
          <td>0.102254</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.213674</td>
          <td>0.300694</td>
          <td>26.751006</td>
          <td>0.171686</td>
          <td>26.199393</td>
          <td>0.093981</td>
          <td>26.176940</td>
          <td>0.149126</td>
          <td>25.705748</td>
          <td>0.186380</td>
          <td>25.127784</td>
          <td>0.247767</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.199412</td>
          <td>0.297269</td>
          <td>26.321233</td>
          <td>0.118642</td>
          <td>26.007534</td>
          <td>0.079373</td>
          <td>25.909688</td>
          <td>0.118364</td>
          <td>26.087925</td>
          <td>0.256227</td>
          <td>24.648405</td>
          <td>0.165767</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.424737</td>
          <td>0.355520</td>
          <td>27.260977</td>
          <td>0.262735</td>
          <td>26.540297</td>
          <td>0.126552</td>
          <td>26.792540</td>
          <td>0.250350</td>
          <td>26.104755</td>
          <td>0.259783</td>
          <td>25.710354</td>
          <td>0.394537</td>
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
          <td>28.528703</td>
          <td>1.515174</td>
          <td>26.752149</td>
          <td>0.197189</td>
          <td>26.138431</td>
          <td>0.104684</td>
          <td>25.142232</td>
          <td>0.071403</td>
          <td>24.750903</td>
          <td>0.095809</td>
          <td>24.055294</td>
          <td>0.117131</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.643683</td>
          <td>0.405090</td>
          <td>26.520543</td>
          <td>0.145852</td>
          <td>26.409571</td>
          <td>0.213692</td>
          <td>26.151882</td>
          <td>0.313157</td>
          <td>24.804902</td>
          <td>0.222010</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.687295</td>
          <td>1.513510</td>
          <td>28.230776</td>
          <td>0.582302</td>
          <td>26.129082</td>
          <td>0.172451</td>
          <td>24.950606</td>
          <td>0.116633</td>
          <td>24.259930</td>
          <td>0.143015</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.181006</td>
          <td>1.305438</td>
          <td>29.497814</td>
          <td>1.407684</td>
          <td>27.288975</td>
          <td>0.295324</td>
          <td>26.248232</td>
          <td>0.199402</td>
          <td>26.231835</td>
          <td>0.354372</td>
          <td>25.068920</td>
          <td>0.293853</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.942034</td>
          <td>0.579382</td>
          <td>26.197412</td>
          <td>0.122770</td>
          <td>25.965253</td>
          <td>0.089966</td>
          <td>25.711219</td>
          <td>0.117717</td>
          <td>25.443681</td>
          <td>0.174462</td>
          <td>24.986374</td>
          <td>0.257915</td>
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
          <td>26.888120</td>
          <td>0.564670</td>
          <td>26.424468</td>
          <td>0.152075</td>
          <td>25.387821</td>
          <td>0.055134</td>
          <td>24.926700</td>
          <td>0.060304</td>
          <td>24.855372</td>
          <td>0.107203</td>
          <td>24.219766</td>
          <td>0.137971</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.554595</td>
          <td>0.436776</td>
          <td>26.944994</td>
          <td>0.232428</td>
          <td>26.091266</td>
          <td>0.100870</td>
          <td>25.214734</td>
          <td>0.076459</td>
          <td>24.877145</td>
          <td>0.107448</td>
          <td>24.229454</td>
          <td>0.136786</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.196008</td>
          <td>0.332745</td>
          <td>26.390900</td>
          <td>0.146692</td>
          <td>26.292394</td>
          <td>0.121244</td>
          <td>26.719446</td>
          <td>0.279196</td>
          <td>26.176021</td>
          <td>0.322908</td>
          <td>24.999157</td>
          <td>0.263749</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.246682</td>
          <td>0.350782</td>
          <td>26.303614</td>
          <td>0.138351</td>
          <td>26.023937</td>
          <td>0.097714</td>
          <td>26.053253</td>
          <td>0.163151</td>
          <td>26.640840</td>
          <td>0.470037</td>
          <td>25.226690</td>
          <td>0.322499</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.626291</td>
          <td>0.920178</td>
          <td>26.606927</td>
          <td>0.175963</td>
          <td>26.333053</td>
          <td>0.125258</td>
          <td>26.354389</td>
          <td>0.206041</td>
          <td>25.966204</td>
          <td>0.272034</td>
          <td>25.536685</td>
          <td>0.403119</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.985782</td>
          <td>0.209290</td>
          <td>26.014535</td>
          <td>0.079875</td>
          <td>25.137944</td>
          <td>0.060010</td>
          <td>24.696009</td>
          <td>0.077653</td>
          <td>24.180003</td>
          <td>0.110655</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.739589</td>
          <td>0.904235</td>
          <td>27.616816</td>
          <td>0.349808</td>
          <td>26.656939</td>
          <td>0.140109</td>
          <td>26.195984</td>
          <td>0.151730</td>
          <td>25.902696</td>
          <td>0.220063</td>
          <td>25.790744</td>
          <td>0.420011</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.690668</td>
          <td>0.913809</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.955031</td>
          <td>0.133955</td>
          <td>25.084004</td>
          <td>0.118403</td>
          <td>24.180192</td>
          <td>0.120358</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.522262</td>
          <td>0.887999</td>
          <td>27.475688</td>
          <td>0.374190</td>
          <td>27.469839</td>
          <td>0.340072</td>
          <td>26.390833</td>
          <td>0.223851</td>
          <td>25.470008</td>
          <td>0.189591</td>
          <td>25.488943</td>
          <td>0.407702</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.395078</td>
          <td>0.347639</td>
          <td>26.049080</td>
          <td>0.093662</td>
          <td>25.954118</td>
          <td>0.075823</td>
          <td>25.784226</td>
          <td>0.106255</td>
          <td>25.593367</td>
          <td>0.169670</td>
          <td>25.002696</td>
          <td>0.223728</td>
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
          <td>26.353703</td>
          <td>0.130628</td>
          <td>25.436043</td>
          <td>0.051799</td>
          <td>25.119466</td>
          <td>0.064153</td>
          <td>24.826931</td>
          <td>0.094285</td>
          <td>24.524421</td>
          <td>0.161422</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.215581</td>
          <td>0.645153</td>
          <td>26.944224</td>
          <td>0.204891</td>
          <td>25.987085</td>
          <td>0.079255</td>
          <td>25.171575</td>
          <td>0.062909</td>
          <td>24.852242</td>
          <td>0.090588</td>
          <td>24.292737</td>
          <td>0.124129</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.461980</td>
          <td>2.198742</td>
          <td>26.729227</td>
          <td>0.175646</td>
          <td>26.356637</td>
          <td>0.113231</td>
          <td>26.021075</td>
          <td>0.137115</td>
          <td>25.859796</td>
          <td>0.222201</td>
          <td>25.393715</td>
          <td>0.322021</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.682053</td>
          <td>0.465224</td>
          <td>26.257229</td>
          <td>0.124006</td>
          <td>26.135519</td>
          <td>0.099647</td>
          <td>26.252112</td>
          <td>0.178620</td>
          <td>25.711552</td>
          <td>0.208985</td>
          <td>25.074061</td>
          <td>0.264836</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.208780</td>
          <td>0.650161</td>
          <td>27.042750</td>
          <td>0.226607</td>
          <td>26.268674</td>
          <td>0.103821</td>
          <td>26.523928</td>
          <td>0.208208</td>
          <td>25.711668</td>
          <td>0.194433</td>
          <td>25.310587</td>
          <td>0.298462</td>
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
