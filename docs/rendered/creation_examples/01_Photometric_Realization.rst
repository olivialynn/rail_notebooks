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

    <pzflow.flow.Flow at 0x7f762c588c10>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.784693</td>
          <td>0.176666</td>
          <td>26.000610</td>
          <td>0.078889</td>
          <td>25.230612</td>
          <td>0.065140</td>
          <td>24.755354</td>
          <td>0.081818</td>
          <td>23.939665</td>
          <td>0.089632</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.058989</td>
          <td>0.572469</td>
          <td>27.375368</td>
          <td>0.288324</td>
          <td>26.436665</td>
          <td>0.115656</td>
          <td>25.947476</td>
          <td>0.122315</td>
          <td>25.514945</td>
          <td>0.158472</td>
          <td>25.274261</td>
          <td>0.279268</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.670466</td>
          <td>2.347169</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.929895</td>
          <td>0.398766</td>
          <td>26.001223</td>
          <td>0.128153</td>
          <td>24.931187</td>
          <td>0.095507</td>
          <td>24.330972</td>
          <td>0.126164</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.478106</td>
          <td>0.763950</td>
          <td>27.462256</td>
          <td>0.309193</td>
          <td>27.098981</td>
          <td>0.203902</td>
          <td>25.963436</td>
          <td>0.124022</td>
          <td>25.453038</td>
          <td>0.150286</td>
          <td>24.893734</td>
          <td>0.203987</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.476456</td>
          <td>0.370185</td>
          <td>26.173992</td>
          <td>0.104357</td>
          <td>26.133007</td>
          <td>0.088654</td>
          <td>25.686835</td>
          <td>0.097425</td>
          <td>25.586771</td>
          <td>0.168488</td>
          <td>24.993376</td>
          <td>0.221691</td>
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
          <td>26.505326</td>
          <td>0.378590</td>
          <td>26.514336</td>
          <td>0.140212</td>
          <td>25.374688</td>
          <td>0.045292</td>
          <td>25.066651</td>
          <td>0.056323</td>
          <td>24.907549</td>
          <td>0.093545</td>
          <td>24.996129</td>
          <td>0.222199</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.148760</td>
          <td>1.153427</td>
          <td>27.143916</td>
          <td>0.238652</td>
          <td>26.087644</td>
          <td>0.085183</td>
          <td>25.181126</td>
          <td>0.062344</td>
          <td>24.841737</td>
          <td>0.088287</td>
          <td>24.175753</td>
          <td>0.110230</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.433457</td>
          <td>0.741637</td>
          <td>27.150580</td>
          <td>0.239969</td>
          <td>26.270850</td>
          <td>0.100060</td>
          <td>26.398604</td>
          <td>0.180169</td>
          <td>25.749748</td>
          <td>0.193429</td>
          <td>25.339020</td>
          <td>0.294284</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.266175</td>
          <td>0.662116</td>
          <td>26.129911</td>
          <td>0.100411</td>
          <td>26.146526</td>
          <td>0.089714</td>
          <td>25.874584</td>
          <td>0.114802</td>
          <td>25.656386</td>
          <td>0.178754</td>
          <td>25.898659</td>
          <td>0.455411</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.403703</td>
          <td>0.349697</td>
          <td>26.827759</td>
          <td>0.183227</td>
          <td>26.745509</td>
          <td>0.151057</td>
          <td>26.388607</td>
          <td>0.178649</td>
          <td>26.020721</td>
          <td>0.242456</td>
          <td>25.348437</td>
          <td>0.296525</td>
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
          <td>26.468479</td>
          <td>0.407917</td>
          <td>26.796390</td>
          <td>0.204645</td>
          <td>26.122696</td>
          <td>0.103253</td>
          <td>25.188063</td>
          <td>0.074355</td>
          <td>24.806209</td>
          <td>0.100567</td>
          <td>23.805102</td>
          <td>0.094126</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.068858</td>
          <td>0.633532</td>
          <td>27.516702</td>
          <td>0.367150</td>
          <td>26.588847</td>
          <td>0.154656</td>
          <td>26.754791</td>
          <td>0.283882</td>
          <td>25.956014</td>
          <td>0.267340</td>
          <td>25.175156</td>
          <td>0.300577</td>
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
          <td>27.970643</td>
          <td>0.481807</td>
          <td>25.721639</td>
          <td>0.121491</td>
          <td>24.950969</td>
          <td>0.116670</td>
          <td>24.369593</td>
          <td>0.157125</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.654363</td>
          <td>0.866784</td>
          <td>27.143112</td>
          <td>0.262356</td>
          <td>26.188000</td>
          <td>0.189544</td>
          <td>25.002070</td>
          <td>0.127487</td>
          <td>25.227618</td>
          <td>0.333594</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.027699</td>
          <td>0.288360</td>
          <td>26.187723</td>
          <td>0.121743</td>
          <td>25.919138</td>
          <td>0.086389</td>
          <td>25.919718</td>
          <td>0.141004</td>
          <td>25.574855</td>
          <td>0.194923</td>
          <td>25.451353</td>
          <td>0.374061</td>
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
          <td>26.887427</td>
          <td>0.564390</td>
          <td>26.085049</td>
          <td>0.113433</td>
          <td>25.481388</td>
          <td>0.059905</td>
          <td>25.142638</td>
          <td>0.073010</td>
          <td>24.896021</td>
          <td>0.111074</td>
          <td>24.872829</td>
          <td>0.239642</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.897573</td>
          <td>0.562579</td>
          <td>26.618114</td>
          <td>0.176734</td>
          <td>26.038558</td>
          <td>0.096317</td>
          <td>25.225136</td>
          <td>0.077165</td>
          <td>24.853520</td>
          <td>0.105253</td>
          <td>24.330933</td>
          <td>0.149269</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.043043</td>
          <td>0.626972</td>
          <td>26.842957</td>
          <td>0.215115</td>
          <td>26.255738</td>
          <td>0.117442</td>
          <td>26.444374</td>
          <td>0.222712</td>
          <td>25.965438</td>
          <td>0.272557</td>
          <td>27.494590</td>
          <td>1.430906</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.788270</td>
          <td>0.528825</td>
          <td>26.202671</td>
          <td>0.126802</td>
          <td>26.028397</td>
          <td>0.098097</td>
          <td>25.874863</td>
          <td>0.140006</td>
          <td>25.616130</td>
          <td>0.207871</td>
          <td>25.405678</td>
          <td>0.371345</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.519268</td>
          <td>0.860344</td>
          <td>26.779985</td>
          <td>0.203603</td>
          <td>26.625601</td>
          <td>0.161129</td>
          <td>26.384313</td>
          <td>0.211265</td>
          <td>26.119601</td>
          <td>0.307912</td>
          <td>25.348947</td>
          <td>0.348323</td>
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
          <td>27.691523</td>
          <td>0.876989</td>
          <td>27.089772</td>
          <td>0.228222</td>
          <td>26.209349</td>
          <td>0.094819</td>
          <td>25.151327</td>
          <td>0.060726</td>
          <td>24.876194</td>
          <td>0.091015</td>
          <td>24.075906</td>
          <td>0.101030</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.197445</td>
          <td>0.542839</td>
          <td>26.594240</td>
          <td>0.132727</td>
          <td>26.173215</td>
          <td>0.148794</td>
          <td>26.534169</td>
          <td>0.366663</td>
          <td>25.547769</td>
          <td>0.347862</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.947668</td>
          <td>1.790068</td>
          <td>28.628095</td>
          <td>0.773457</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.284391</td>
          <td>0.177609</td>
          <td>25.015484</td>
          <td>0.111544</td>
          <td>24.513538</td>
          <td>0.160421</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.514857</td>
          <td>0.440905</td>
          <td>28.502001</td>
          <td>0.783774</td>
          <td>27.515141</td>
          <td>0.352431</td>
          <td>26.246064</td>
          <td>0.198339</td>
          <td>25.299149</td>
          <td>0.164011</td>
          <td>25.625250</td>
          <td>0.452214</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.515093</td>
          <td>0.381801</td>
          <td>25.986486</td>
          <td>0.088654</td>
          <td>25.919962</td>
          <td>0.073568</td>
          <td>25.559142</td>
          <td>0.087214</td>
          <td>25.612837</td>
          <td>0.172503</td>
          <td>25.197456</td>
          <td>0.262695</td>
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
          <td>26.375696</td>
          <td>0.359593</td>
          <td>26.217843</td>
          <td>0.116113</td>
          <td>25.338972</td>
          <td>0.047523</td>
          <td>24.985005</td>
          <td>0.056941</td>
          <td>24.869634</td>
          <td>0.097884</td>
          <td>24.805412</td>
          <td>0.204758</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.843544</td>
          <td>0.494121</td>
          <td>26.647307</td>
          <td>0.159367</td>
          <td>26.088774</td>
          <td>0.086689</td>
          <td>25.149331</td>
          <td>0.061680</td>
          <td>24.960039</td>
          <td>0.099577</td>
          <td>24.277793</td>
          <td>0.122529</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.406810</td>
          <td>0.747852</td>
          <td>26.723310</td>
          <td>0.174766</td>
          <td>26.344335</td>
          <td>0.112023</td>
          <td>26.123129</td>
          <td>0.149704</td>
          <td>25.806638</td>
          <td>0.212570</td>
          <td>25.100078</td>
          <td>0.253953</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.834680</td>
          <td>0.238376</td>
          <td>26.199360</td>
          <td>0.117933</td>
          <td>26.190738</td>
          <td>0.104582</td>
          <td>25.901950</td>
          <td>0.132333</td>
          <td>26.103323</td>
          <td>0.288487</td>
          <td>25.050602</td>
          <td>0.259807</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.502198</td>
          <td>0.386868</td>
          <td>27.209740</td>
          <td>0.260033</td>
          <td>26.418947</td>
          <td>0.118363</td>
          <td>26.510717</td>
          <td>0.205917</td>
          <td>25.954207</td>
          <td>0.238036</td>
          <td>25.134334</td>
          <td>0.258675</td>
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
