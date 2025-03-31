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

    <pzflow.flow.Flow at 0x7f9d01d095d0>



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
          <td>27.949267</td>
          <td>1.027154</td>
          <td>26.641778</td>
          <td>0.156419</td>
          <td>26.054963</td>
          <td>0.082764</td>
          <td>25.144965</td>
          <td>0.060376</td>
          <td>24.651169</td>
          <td>0.074627</td>
          <td>23.920371</td>
          <td>0.088124</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.644061</td>
          <td>0.357112</td>
          <td>26.704219</td>
          <td>0.145794</td>
          <td>26.131535</td>
          <td>0.143417</td>
          <td>25.872881</td>
          <td>0.214468</td>
          <td>25.053104</td>
          <td>0.232958</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.817624</td>
          <td>0.408577</td>
          <td>27.683928</td>
          <td>0.328952</td>
          <td>25.904382</td>
          <td>0.117819</td>
          <td>24.952386</td>
          <td>0.097299</td>
          <td>24.390546</td>
          <td>0.132842</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.201761</td>
          <td>0.544171</td>
          <td>28.512040</td>
          <td>0.612851</td>
          <td>26.420513</td>
          <td>0.183543</td>
          <td>25.215762</td>
          <td>0.122447</td>
          <td>25.402359</td>
          <td>0.309647</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.733833</td>
          <td>0.202803</td>
          <td>26.322962</td>
          <td>0.118820</td>
          <td>25.858057</td>
          <td>0.069547</td>
          <td>25.773585</td>
          <td>0.105115</td>
          <td>25.334884</td>
          <td>0.135752</td>
          <td>24.929380</td>
          <td>0.210167</td>
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
          <td>26.276554</td>
          <td>0.114121</td>
          <td>25.491512</td>
          <td>0.050242</td>
          <td>25.083720</td>
          <td>0.057182</td>
          <td>24.868002</td>
          <td>0.090350</td>
          <td>24.886350</td>
          <td>0.202727</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.797815</td>
          <td>0.937079</td>
          <td>26.844084</td>
          <td>0.185773</td>
          <td>25.910798</td>
          <td>0.072870</td>
          <td>25.223031</td>
          <td>0.064704</td>
          <td>24.936173</td>
          <td>0.095925</td>
          <td>24.105878</td>
          <td>0.103702</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.665406</td>
          <td>0.428136</td>
          <td>26.408395</td>
          <td>0.127957</td>
          <td>26.449839</td>
          <td>0.116990</td>
          <td>26.263940</td>
          <td>0.160664</td>
          <td>25.872900</td>
          <td>0.214472</td>
          <td>25.776602</td>
          <td>0.415141</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.836205</td>
          <td>0.486715</td>
          <td>26.064808</td>
          <td>0.094846</td>
          <td>26.066436</td>
          <td>0.083606</td>
          <td>25.586353</td>
          <td>0.089194</td>
          <td>25.634665</td>
          <td>0.175491</td>
          <td>24.979553</td>
          <td>0.219154</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.101894</td>
          <td>0.590239</td>
          <td>26.565647</td>
          <td>0.146538</td>
          <td>26.388307</td>
          <td>0.110882</td>
          <td>26.404921</td>
          <td>0.181136</td>
          <td>26.328963</td>
          <td>0.311475</td>
          <td>25.446772</td>
          <td>0.320829</td>
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
          <td>29.146453</td>
          <td>2.009377</td>
          <td>26.708530</td>
          <td>0.190082</td>
          <td>25.803633</td>
          <td>0.078001</td>
          <td>25.245326</td>
          <td>0.078212</td>
          <td>24.711962</td>
          <td>0.092590</td>
          <td>24.191089</td>
          <td>0.131771</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.693894</td>
          <td>1.641836</td>
          <td>27.562505</td>
          <td>0.380477</td>
          <td>26.739699</td>
          <td>0.175883</td>
          <td>26.029327</td>
          <td>0.154905</td>
          <td>25.910776</td>
          <td>0.257637</td>
          <td>24.495608</td>
          <td>0.171141</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.122376</td>
          <td>0.587202</td>
          <td>28.001927</td>
          <td>0.493117</td>
          <td>26.071773</td>
          <td>0.164239</td>
          <td>25.104797</td>
          <td>0.133317</td>
          <td>24.276332</td>
          <td>0.145047</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.551893</td>
          <td>0.454472</td>
          <td>29.745290</td>
          <td>1.592862</td>
          <td>27.024451</td>
          <td>0.237984</td>
          <td>26.559332</td>
          <td>0.258128</td>
          <td>25.513719</td>
          <td>0.197374</td>
          <td>25.323920</td>
          <td>0.359889</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.840148</td>
          <td>0.538447</td>
          <td>26.109925</td>
          <td>0.113787</td>
          <td>25.911942</td>
          <td>0.085843</td>
          <td>25.710194</td>
          <td>0.117612</td>
          <td>25.467178</td>
          <td>0.177975</td>
          <td>25.841297</td>
          <td>0.502763</td>
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
          <td>27.970664</td>
          <td>1.136183</td>
          <td>26.037794</td>
          <td>0.108859</td>
          <td>25.446292</td>
          <td>0.058069</td>
          <td>25.196024</td>
          <td>0.076536</td>
          <td>24.725847</td>
          <td>0.095713</td>
          <td>24.546288</td>
          <td>0.182381</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.555667</td>
          <td>0.437131</td>
          <td>26.441508</td>
          <td>0.152038</td>
          <td>26.198504</td>
          <td>0.110780</td>
          <td>25.316102</td>
          <td>0.083611</td>
          <td>24.757256</td>
          <td>0.096747</td>
          <td>24.126063</td>
          <td>0.125085</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.947122</td>
          <td>0.585974</td>
          <td>26.803078</td>
          <td>0.208069</td>
          <td>26.396705</td>
          <td>0.132713</td>
          <td>26.138049</td>
          <td>0.172125</td>
          <td>25.837969</td>
          <td>0.245552</td>
          <td>25.627220</td>
          <td>0.433051</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.695665</td>
          <td>0.224688</td>
          <td>25.985030</td>
          <td>0.104943</td>
          <td>26.035078</td>
          <td>0.098673</td>
          <td>25.957537</td>
          <td>0.150321</td>
          <td>25.724727</td>
          <td>0.227561</td>
          <td>25.186067</td>
          <td>0.312213</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.666356</td>
          <td>0.476824</td>
          <td>26.740295</td>
          <td>0.196933</td>
          <td>26.523530</td>
          <td>0.147640</td>
          <td>26.757667</td>
          <td>0.287221</td>
          <td>26.363093</td>
          <td>0.373247</td>
          <td>25.560287</td>
          <td>0.410490</td>
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
          <td>27.182739</td>
          <td>0.246438</td>
          <td>25.938734</td>
          <td>0.074702</td>
          <td>25.049593</td>
          <td>0.055484</td>
          <td>24.734329</td>
          <td>0.080325</td>
          <td>24.000852</td>
          <td>0.094596</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.301903</td>
          <td>0.678907</td>
          <td>27.993524</td>
          <td>0.467151</td>
          <td>26.728003</td>
          <td>0.148942</td>
          <td>26.470058</td>
          <td>0.191569</td>
          <td>25.665982</td>
          <td>0.180378</td>
          <td>25.215833</td>
          <td>0.266545</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.630737</td>
          <td>2.370915</td>
          <td>28.424172</td>
          <td>0.674373</td>
          <td>28.400391</td>
          <td>0.604875</td>
          <td>26.453154</td>
          <td>0.204771</td>
          <td>25.111776</td>
          <td>0.121295</td>
          <td>24.385223</td>
          <td>0.143707</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.347624</td>
          <td>2.229679</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.986534</td>
          <td>0.229859</td>
          <td>26.218098</td>
          <td>0.193728</td>
          <td>25.742932</td>
          <td>0.238116</td>
          <td>25.599701</td>
          <td>0.443583</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.898525</td>
          <td>0.510036</td>
          <td>26.138754</td>
          <td>0.101316</td>
          <td>25.965894</td>
          <td>0.076616</td>
          <td>25.662861</td>
          <td>0.095539</td>
          <td>25.337784</td>
          <td>0.136282</td>
          <td>24.887869</td>
          <td>0.203272</td>
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
          <td>26.361624</td>
          <td>0.355652</td>
          <td>26.407545</td>
          <td>0.136843</td>
          <td>25.485200</td>
          <td>0.054109</td>
          <td>25.022846</td>
          <td>0.058886</td>
          <td>24.802803</td>
          <td>0.092308</td>
          <td>24.405155</td>
          <td>0.145740</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.566812</td>
          <td>0.816518</td>
          <td>26.622405</td>
          <td>0.156010</td>
          <td>26.103002</td>
          <td>0.087782</td>
          <td>25.058066</td>
          <td>0.056882</td>
          <td>24.929851</td>
          <td>0.096977</td>
          <td>24.180121</td>
          <td>0.112547</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.746459</td>
          <td>0.468745</td>
          <td>26.722535</td>
          <td>0.174652</td>
          <td>26.413350</td>
          <td>0.118961</td>
          <td>26.521249</td>
          <td>0.209818</td>
          <td>25.548594</td>
          <td>0.171007</td>
          <td>26.238553</td>
          <td>0.608513</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.078406</td>
          <td>0.290783</td>
          <td>26.168766</td>
          <td>0.114838</td>
          <td>26.142288</td>
          <td>0.100240</td>
          <td>25.819961</td>
          <td>0.123260</td>
          <td>26.121172</td>
          <td>0.292674</td>
          <td>26.705631</td>
          <td>0.875856</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.311417</td>
          <td>0.697569</td>
          <td>26.854597</td>
          <td>0.193628</td>
          <td>26.600720</td>
          <td>0.138547</td>
          <td>26.123759</td>
          <td>0.148264</td>
          <td>26.461358</td>
          <td>0.358245</td>
          <td>25.594330</td>
          <td>0.373669</td>
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
