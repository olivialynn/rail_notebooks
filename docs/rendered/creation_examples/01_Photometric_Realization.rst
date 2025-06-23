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

    <pzflow.flow.Flow at 0x7fb5038d0ee0>



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
    0      23.994413  0.058650  0.037731  
    1      25.391064  0.120471  0.115757  
    2      24.304707  0.021114  0.020311  
    3      25.291103  0.139794  0.136781  
    4      25.096743  0.061381  0.034190  
    ...          ...       ...       ...  
    99995  24.737946  0.198761  0.145643  
    99996  24.224169  0.018943  0.014378  
    99997  25.613836  0.008151  0.008125  
    99998  25.274899  0.152035  0.131041  
    99999  25.699642  0.097289  0.050924  
    
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
          <td>27.471556</td>
          <td>0.760648</td>
          <td>26.959581</td>
          <td>0.204729</td>
          <td>26.083595</td>
          <td>0.084879</td>
          <td>25.205862</td>
          <td>0.063727</td>
          <td>24.774268</td>
          <td>0.083193</td>
          <td>23.849221</td>
          <td>0.082771</td>
          <td>0.058650</td>
          <td>0.037731</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.982343</td>
          <td>0.462933</td>
          <td>26.408772</td>
          <td>0.112879</td>
          <td>26.253654</td>
          <td>0.159257</td>
          <td>25.901128</td>
          <td>0.219579</td>
          <td>25.335799</td>
          <td>0.293521</td>
          <td>0.120471</td>
          <td>0.115757</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.846246</td>
          <td>0.965328</td>
          <td>29.926089</td>
          <td>1.553913</td>
          <td>28.880724</td>
          <td>0.787351</td>
          <td>26.052957</td>
          <td>0.134019</td>
          <td>25.103771</td>
          <td>0.111076</td>
          <td>24.280209</td>
          <td>0.120726</td>
          <td>0.021114</td>
          <td>0.020311</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.902019</td>
          <td>1.700392</td>
          <td>27.767071</td>
          <td>0.392987</td>
          <td>27.334147</td>
          <td>0.247896</td>
          <td>26.459404</td>
          <td>0.189674</td>
          <td>25.378825</td>
          <td>0.140995</td>
          <td>25.609192</td>
          <td>0.364718</td>
          <td>0.139794</td>
          <td>0.136781</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.126738</td>
          <td>0.280339</td>
          <td>26.513797</td>
          <td>0.140147</td>
          <td>26.144990</td>
          <td>0.089593</td>
          <td>25.811636</td>
          <td>0.108669</td>
          <td>25.526462</td>
          <td>0.160039</td>
          <td>25.105229</td>
          <td>0.243208</td>
          <td>0.061381</td>
          <td>0.034190</td>
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
          <td>26.372951</td>
          <td>0.341332</td>
          <td>26.257535</td>
          <td>0.112246</td>
          <td>25.421280</td>
          <td>0.047205</td>
          <td>25.128286</td>
          <td>0.059489</td>
          <td>24.842033</td>
          <td>0.088310</td>
          <td>25.019230</td>
          <td>0.226506</td>
          <td>0.198761</td>
          <td>0.145643</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.753720</td>
          <td>0.457669</td>
          <td>26.644675</td>
          <td>0.156807</td>
          <td>25.993163</td>
          <td>0.078372</td>
          <td>25.158252</td>
          <td>0.061092</td>
          <td>24.816700</td>
          <td>0.086362</td>
          <td>24.240275</td>
          <td>0.116606</td>
          <td>0.018943</td>
          <td>0.014378</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.760227</td>
          <td>0.915518</td>
          <td>26.544731</td>
          <td>0.143928</td>
          <td>26.319181</td>
          <td>0.104385</td>
          <td>26.205172</td>
          <td>0.152782</td>
          <td>25.510151</td>
          <td>0.157823</td>
          <td>25.760285</td>
          <td>0.409986</td>
          <td>0.008151</td>
          <td>0.008125</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.337199</td>
          <td>0.331821</td>
          <td>26.048236</td>
          <td>0.093478</td>
          <td>26.083118</td>
          <td>0.084844</td>
          <td>25.903594</td>
          <td>0.117738</td>
          <td>25.867547</td>
          <td>0.213515</td>
          <td>25.028338</td>
          <td>0.228225</td>
          <td>0.152035</td>
          <td>0.131041</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.770912</td>
          <td>0.463605</td>
          <td>26.886977</td>
          <td>0.192616</td>
          <td>26.411563</td>
          <td>0.113154</td>
          <td>26.170605</td>
          <td>0.148317</td>
          <td>25.842312</td>
          <td>0.209059</td>
          <td>26.600806</td>
          <td>0.749715</td>
          <td>0.097289</td>
          <td>0.050924</td>
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
          <td>27.415055</td>
          <td>0.803749</td>
          <td>26.825398</td>
          <td>0.211149</td>
          <td>26.012785</td>
          <td>0.094541</td>
          <td>25.108124</td>
          <td>0.069874</td>
          <td>24.454610</td>
          <td>0.074415</td>
          <td>23.857670</td>
          <td>0.099397</td>
          <td>0.058650</td>
          <td>0.037731</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.375203</td>
          <td>0.799908</td>
          <td>27.835890</td>
          <td>0.485116</td>
          <td>26.771668</td>
          <td>0.188733</td>
          <td>26.441792</td>
          <td>0.229408</td>
          <td>25.596826</td>
          <td>0.207265</td>
          <td>25.845212</td>
          <td>0.524020</td>
          <td>0.120471</td>
          <td>0.115757</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.233729</td>
          <td>0.573727</td>
          <td>26.008117</td>
          <td>0.152303</td>
          <td>25.000290</td>
          <td>0.119292</td>
          <td>24.105861</td>
          <td>0.122570</td>
          <td>0.021114</td>
          <td>0.020311</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.878733</td>
          <td>0.575249</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.809622</td>
          <td>0.197831</td>
          <td>26.533435</td>
          <td>0.251237</td>
          <td>25.358439</td>
          <td>0.172088</td>
          <td>25.254506</td>
          <td>0.338872</td>
          <td>0.139794</td>
          <td>0.136781</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.536720</td>
          <td>0.432067</td>
          <td>26.191130</td>
          <td>0.122974</td>
          <td>25.973933</td>
          <td>0.091381</td>
          <td>25.751369</td>
          <td>0.122894</td>
          <td>25.371942</td>
          <td>0.165402</td>
          <td>25.391658</td>
          <td>0.359627</td>
          <td>0.061381</td>
          <td>0.034190</td>
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
          <td>27.820585</td>
          <td>1.081113</td>
          <td>26.305231</td>
          <td>0.146388</td>
          <td>25.356114</td>
          <td>0.057658</td>
          <td>25.139599</td>
          <td>0.078465</td>
          <td>24.738374</td>
          <td>0.103969</td>
          <td>25.207332</td>
          <td>0.336182</td>
          <td>0.198761</td>
          <td>0.145643</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.290232</td>
          <td>0.737208</td>
          <td>26.637619</td>
          <td>0.179173</td>
          <td>26.032014</td>
          <td>0.095456</td>
          <td>25.135189</td>
          <td>0.071028</td>
          <td>24.859507</td>
          <td>0.105465</td>
          <td>24.276389</td>
          <td>0.141968</td>
          <td>0.018943</td>
          <td>0.014378</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.140919</td>
          <td>0.315744</td>
          <td>26.495041</td>
          <td>0.158607</td>
          <td>26.414118</td>
          <td>0.133066</td>
          <td>26.107908</td>
          <td>0.165660</td>
          <td>25.475518</td>
          <td>0.179215</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008151</td>
          <td>0.008125</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.315537</td>
          <td>0.378598</td>
          <td>26.356621</td>
          <td>0.149074</td>
          <td>25.962094</td>
          <td>0.095625</td>
          <td>25.601826</td>
          <td>0.114244</td>
          <td>25.714848</td>
          <td>0.232755</td>
          <td>24.822182</td>
          <td>0.239560</td>
          <td>0.152035</td>
          <td>0.131041</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.601972</td>
          <td>0.176752</td>
          <td>26.710798</td>
          <td>0.174928</td>
          <td>26.333202</td>
          <td>0.204406</td>
          <td>26.150550</td>
          <td>0.318483</td>
          <td>26.148288</td>
          <td>0.636953</td>
          <td>0.097289</td>
          <td>0.050924</td>
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
          <td>26.908470</td>
          <td>0.201395</td>
          <td>25.983860</td>
          <td>0.080257</td>
          <td>25.168549</td>
          <td>0.063766</td>
          <td>24.673676</td>
          <td>0.078599</td>
          <td>23.885945</td>
          <td>0.088371</td>
          <td>0.058650</td>
          <td>0.037731</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.305929</td>
          <td>0.656338</td>
          <td>26.482064</td>
          <td>0.140823</td>
          <td>26.554439</td>
          <td>0.240523</td>
          <td>26.258413</td>
          <td>0.340207</td>
          <td>25.146581</td>
          <td>0.293184</td>
          <td>0.120471</td>
          <td>0.115757</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.296887</td>
          <td>0.678355</td>
          <td>29.999861</td>
          <td>1.614921</td>
          <td>31.184863</td>
          <td>2.494081</td>
          <td>26.036731</td>
          <td>0.132947</td>
          <td>25.129681</td>
          <td>0.114268</td>
          <td>24.552496</td>
          <td>0.153619</td>
          <td>0.021114</td>
          <td>0.020311</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.411291</td>
          <td>1.330036</td>
          <td>27.252843</td>
          <td>0.281311</td>
          <td>26.322438</td>
          <td>0.207838</td>
          <td>25.250586</td>
          <td>0.154682</td>
          <td>25.808801</td>
          <td>0.510538</td>
          <td>0.139794</td>
          <td>0.136781</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.306210</td>
          <td>0.330421</td>
          <td>26.108855</td>
          <td>0.101372</td>
          <td>26.063966</td>
          <td>0.086160</td>
          <td>25.926267</td>
          <td>0.124151</td>
          <td>25.580069</td>
          <td>0.172821</td>
          <td>24.699458</td>
          <td>0.178799</td>
          <td>0.061381</td>
          <td>0.034190</td>
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
          <td>29.424736</td>
          <td>2.351808</td>
          <td>26.513468</td>
          <td>0.180949</td>
          <td>25.443221</td>
          <td>0.064741</td>
          <td>25.010815</td>
          <td>0.072851</td>
          <td>24.996133</td>
          <td>0.135089</td>
          <td>24.911172</td>
          <td>0.274686</td>
          <td>0.198761</td>
          <td>0.145643</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.609785</td>
          <td>0.411297</td>
          <td>26.590274</td>
          <td>0.150155</td>
          <td>26.070192</td>
          <td>0.084206</td>
          <td>25.203584</td>
          <td>0.063857</td>
          <td>24.729249</td>
          <td>0.080263</td>
          <td>24.193506</td>
          <td>0.112394</td>
          <td>0.018943</td>
          <td>0.014378</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.588094</td>
          <td>0.403797</td>
          <td>26.705860</td>
          <td>0.165340</td>
          <td>26.414092</td>
          <td>0.113505</td>
          <td>26.219186</td>
          <td>0.154771</td>
          <td>25.931047</td>
          <td>0.225303</td>
          <td>24.856054</td>
          <td>0.197811</td>
          <td>0.008151</td>
          <td>0.008125</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.650677</td>
          <td>0.485125</td>
          <td>26.098666</td>
          <td>0.118274</td>
          <td>25.860894</td>
          <td>0.086632</td>
          <td>25.945978</td>
          <td>0.152296</td>
          <td>25.713404</td>
          <td>0.230335</td>
          <td>25.611768</td>
          <td>0.443970</td>
          <td>0.152035</td>
          <td>0.131041</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.179430</td>
          <td>0.649979</td>
          <td>26.754201</td>
          <td>0.183284</td>
          <td>26.556423</td>
          <td>0.138025</td>
          <td>26.191121</td>
          <td>0.162765</td>
          <td>25.681781</td>
          <td>0.196073</td>
          <td>26.044040</td>
          <td>0.540758</td>
          <td>0.097289</td>
          <td>0.050924</td>
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
