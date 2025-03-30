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

    <pzflow.flow.Flow at 0x7f0930356f80>



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
          <td>26.699125</td>
          <td>0.164270</td>
          <td>25.979758</td>
          <td>0.077450</td>
          <td>25.227836</td>
          <td>0.064980</td>
          <td>24.696576</td>
          <td>0.077682</td>
          <td>24.067121</td>
          <td>0.100242</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.353719</td>
          <td>1.291784</td>
          <td>27.837743</td>
          <td>0.414922</td>
          <td>26.553204</td>
          <td>0.127976</td>
          <td>26.465573</td>
          <td>0.190663</td>
          <td>25.974947</td>
          <td>0.233458</td>
          <td>25.632733</td>
          <td>0.371483</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.879465</td>
          <td>1.518621</td>
          <td>27.835722</td>
          <td>0.370694</td>
          <td>26.027544</td>
          <td>0.131106</td>
          <td>24.907200</td>
          <td>0.093516</td>
          <td>24.435870</td>
          <td>0.138144</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.358275</td>
          <td>2.075863</td>
          <td>28.047888</td>
          <td>0.486122</td>
          <td>27.079457</td>
          <td>0.200589</td>
          <td>26.192972</td>
          <td>0.151192</td>
          <td>25.485984</td>
          <td>0.154592</td>
          <td>25.118095</td>
          <td>0.245800</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.162006</td>
          <td>0.288446</td>
          <td>25.937789</td>
          <td>0.084834</td>
          <td>25.952204</td>
          <td>0.075587</td>
          <td>25.602689</td>
          <td>0.090485</td>
          <td>25.294694</td>
          <td>0.131117</td>
          <td>24.742773</td>
          <td>0.179611</td>
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
          <td>27.163453</td>
          <td>0.616457</td>
          <td>26.188698</td>
          <td>0.105706</td>
          <td>25.404893</td>
          <td>0.046523</td>
          <td>25.048352</td>
          <td>0.055415</td>
          <td>24.915685</td>
          <td>0.094216</td>
          <td>24.819312</td>
          <td>0.191614</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.735101</td>
          <td>1.570543</td>
          <td>26.816901</td>
          <td>0.181552</td>
          <td>26.076205</td>
          <td>0.084329</td>
          <td>25.227717</td>
          <td>0.064974</td>
          <td>24.780679</td>
          <td>0.083665</td>
          <td>24.095031</td>
          <td>0.102722</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.388872</td>
          <td>0.345641</td>
          <td>26.710843</td>
          <td>0.165918</td>
          <td>26.484574</td>
          <td>0.120578</td>
          <td>26.323514</td>
          <td>0.169038</td>
          <td>25.974490</td>
          <td>0.233370</td>
          <td>25.138280</td>
          <td>0.249915</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.865844</td>
          <td>0.976908</td>
          <td>26.454731</td>
          <td>0.133187</td>
          <td>26.124260</td>
          <td>0.087974</td>
          <td>26.051597</td>
          <td>0.133862</td>
          <td>25.666390</td>
          <td>0.180276</td>
          <td>26.326877</td>
          <td>0.621756</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.879579</td>
          <td>0.985075</td>
          <td>26.805644</td>
          <td>0.179830</td>
          <td>26.895417</td>
          <td>0.171694</td>
          <td>26.229203</td>
          <td>0.155960</td>
          <td>25.630504</td>
          <td>0.174872</td>
          <td>26.830959</td>
          <td>0.870551</td>
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
          <td>30.592721</td>
          <td>3.322094</td>
          <td>26.714819</td>
          <td>0.191092</td>
          <td>25.964145</td>
          <td>0.089849</td>
          <td>25.152965</td>
          <td>0.072083</td>
          <td>24.652967</td>
          <td>0.087911</td>
          <td>23.847371</td>
          <td>0.097681</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.669054</td>
          <td>0.474762</td>
          <td>27.590497</td>
          <td>0.388820</td>
          <td>26.643576</td>
          <td>0.162065</td>
          <td>26.042144</td>
          <td>0.156614</td>
          <td>26.044823</td>
          <td>0.287330</td>
          <td>25.305025</td>
          <td>0.333405</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.863909</td>
          <td>0.486659</td>
          <td>28.093897</td>
          <td>0.527582</td>
          <td>26.058017</td>
          <td>0.162322</td>
          <td>25.040652</td>
          <td>0.126119</td>
          <td>24.340998</td>
          <td>0.153325</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.010329</td>
          <td>1.076701</td>
          <td>27.099012</td>
          <td>0.253050</td>
          <td>26.532149</td>
          <td>0.252440</td>
          <td>25.924089</td>
          <td>0.277124</td>
          <td>24.908398</td>
          <td>0.257920</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.846666</td>
          <td>0.248860</td>
          <td>26.057829</td>
          <td>0.108739</td>
          <td>25.912537</td>
          <td>0.085888</td>
          <td>25.699830</td>
          <td>0.116556</td>
          <td>25.797651</td>
          <td>0.234756</td>
          <td>25.243691</td>
          <td>0.317571</td>
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
          <td>27.161925</td>
          <td>0.684013</td>
          <td>26.643178</td>
          <td>0.183189</td>
          <td>25.405386</td>
          <td>0.056000</td>
          <td>25.009827</td>
          <td>0.064915</td>
          <td>24.900333</td>
          <td>0.111492</td>
          <td>24.706732</td>
          <td>0.208744</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.874510</td>
          <td>0.219221</td>
          <td>25.852677</td>
          <td>0.081792</td>
          <td>25.499906</td>
          <td>0.098270</td>
          <td>24.880402</td>
          <td>0.107754</td>
          <td>24.207458</td>
          <td>0.134213</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.432539</td>
          <td>0.400208</td>
          <td>26.809260</td>
          <td>0.209148</td>
          <td>26.353260</td>
          <td>0.127817</td>
          <td>26.103501</td>
          <td>0.167138</td>
          <td>26.318380</td>
          <td>0.361320</td>
          <td>25.544745</td>
          <td>0.406625</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.219382</td>
          <td>0.715608</td>
          <td>26.055144</td>
          <td>0.111558</td>
          <td>25.994541</td>
          <td>0.095227</td>
          <td>25.771702</td>
          <td>0.128068</td>
          <td>25.787924</td>
          <td>0.239779</td>
          <td>24.715533</td>
          <td>0.212435</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.594420</td>
          <td>0.902088</td>
          <td>26.717950</td>
          <td>0.193267</td>
          <td>26.537963</td>
          <td>0.149481</td>
          <td>26.542762</td>
          <td>0.240975</td>
          <td>25.868263</td>
          <td>0.251100</td>
          <td>31.356370</td>
          <td>4.962415</td>
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
          <td>27.548712</td>
          <td>0.800225</td>
          <td>26.564410</td>
          <td>0.146399</td>
          <td>26.022482</td>
          <td>0.080437</td>
          <td>25.235440</td>
          <td>0.065429</td>
          <td>24.725068</td>
          <td>0.079671</td>
          <td>23.934766</td>
          <td>0.089259</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.812550</td>
          <td>0.946051</td>
          <td>27.048197</td>
          <td>0.220622</td>
          <td>26.777418</td>
          <td>0.155389</td>
          <td>26.388029</td>
          <td>0.178733</td>
          <td>25.716980</td>
          <td>0.188328</td>
          <td>24.972236</td>
          <td>0.218026</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.458741</td>
          <td>1.413917</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.896970</td>
          <td>0.417603</td>
          <td>26.090820</td>
          <td>0.150570</td>
          <td>24.963697</td>
          <td>0.106615</td>
          <td>24.513522</td>
          <td>0.160418</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.167141</td>
          <td>1.293549</td>
          <td>28.295259</td>
          <td>0.682434</td>
          <td>28.327267</td>
          <td>0.644129</td>
          <td>26.079683</td>
          <td>0.172320</td>
          <td>25.418832</td>
          <td>0.181567</td>
          <td>25.100362</td>
          <td>0.300377</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.241994</td>
          <td>0.307873</td>
          <td>26.199964</td>
          <td>0.106882</td>
          <td>25.844788</td>
          <td>0.068834</td>
          <td>25.797606</td>
          <td>0.107505</td>
          <td>25.588330</td>
          <td>0.168944</td>
          <td>24.549268</td>
          <td>0.152515</td>
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
          <td>28.841072</td>
          <td>1.702403</td>
          <td>26.396053</td>
          <td>0.135494</td>
          <td>25.488710</td>
          <td>0.054278</td>
          <td>25.067583</td>
          <td>0.061269</td>
          <td>24.746973</td>
          <td>0.087887</td>
          <td>24.755025</td>
          <td>0.196276</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.757922</td>
          <td>0.463630</td>
          <td>26.661626</td>
          <td>0.161327</td>
          <td>26.007830</td>
          <td>0.080720</td>
          <td>25.252988</td>
          <td>0.067615</td>
          <td>24.940333</td>
          <td>0.097872</td>
          <td>24.303085</td>
          <td>0.125248</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.345261</td>
          <td>0.717669</td>
          <td>26.789746</td>
          <td>0.184880</td>
          <td>26.297045</td>
          <td>0.107494</td>
          <td>26.455333</td>
          <td>0.198537</td>
          <td>25.827277</td>
          <td>0.216264</td>
          <td>25.830739</td>
          <td>0.451923</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.333099</td>
          <td>0.356055</td>
          <td>26.313925</td>
          <td>0.130244</td>
          <td>26.137023</td>
          <td>0.099779</td>
          <td>25.921824</td>
          <td>0.134626</td>
          <td>25.657555</td>
          <td>0.199736</td>
          <td>25.568175</td>
          <td>0.392348</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.140899</td>
          <td>0.290770</td>
          <td>27.276790</td>
          <td>0.274643</td>
          <td>26.558388</td>
          <td>0.133575</td>
          <td>26.352343</td>
          <td>0.180194</td>
          <td>25.445046</td>
          <td>0.155030</td>
          <td>25.757686</td>
          <td>0.423794</td>
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
