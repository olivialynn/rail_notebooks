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

    <pzflow.flow.Flow at 0x7f5ed63d6b90>



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
          <td>26.594438</td>
          <td>0.150204</td>
          <td>26.152914</td>
          <td>0.090220</td>
          <td>25.107410</td>
          <td>0.058397</td>
          <td>24.813240</td>
          <td>0.086100</td>
          <td>24.028468</td>
          <td>0.096903</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.847090</td>
          <td>0.490656</td>
          <td>27.434840</td>
          <td>0.302471</td>
          <td>26.404810</td>
          <td>0.112490</td>
          <td>26.536994</td>
          <td>0.202470</td>
          <td>25.876028</td>
          <td>0.215032</td>
          <td>25.456268</td>
          <td>0.323264</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.801344</td>
          <td>0.939120</td>
          <td>29.958212</td>
          <td>1.578442</td>
          <td>27.771868</td>
          <td>0.352618</td>
          <td>26.173231</td>
          <td>0.148652</td>
          <td>25.128170</td>
          <td>0.113464</td>
          <td>24.407269</td>
          <td>0.134775</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.215897</td>
          <td>2.842053</td>
          <td>27.596211</td>
          <td>0.343923</td>
          <td>26.986832</td>
          <td>0.185530</td>
          <td>26.258234</td>
          <td>0.159882</td>
          <td>25.420986</td>
          <td>0.146205</td>
          <td>25.016786</td>
          <td>0.226047</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.887553</td>
          <td>0.230481</td>
          <td>26.218090</td>
          <td>0.108453</td>
          <td>25.898924</td>
          <td>0.072109</td>
          <td>25.648028</td>
          <td>0.094163</td>
          <td>25.478215</td>
          <td>0.153566</td>
          <td>25.248137</td>
          <td>0.273403</td>
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
          <td>27.535128</td>
          <td>0.793114</td>
          <td>26.189233</td>
          <td>0.105755</td>
          <td>25.354001</td>
          <td>0.044468</td>
          <td>25.098716</td>
          <td>0.057949</td>
          <td>24.747193</td>
          <td>0.081231</td>
          <td>24.627851</td>
          <td>0.162886</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.692680</td>
          <td>0.437086</td>
          <td>26.722217</td>
          <td>0.167533</td>
          <td>26.011565</td>
          <td>0.079656</td>
          <td>25.251327</td>
          <td>0.066347</td>
          <td>24.848216</td>
          <td>0.088791</td>
          <td>24.141471</td>
          <td>0.106980</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.973918</td>
          <td>0.538445</td>
          <td>26.456114</td>
          <td>0.133346</td>
          <td>26.321201</td>
          <td>0.104569</td>
          <td>26.254314</td>
          <td>0.159347</td>
          <td>25.808549</td>
          <td>0.203229</td>
          <td>26.130728</td>
          <td>0.540583</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.911193</td>
          <td>0.235028</td>
          <td>26.127482</td>
          <td>0.100198</td>
          <td>26.017954</td>
          <td>0.080106</td>
          <td>25.873446</td>
          <td>0.114688</td>
          <td>25.901004</td>
          <td>0.219556</td>
          <td>25.083529</td>
          <td>0.238893</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.982693</td>
          <td>0.541880</td>
          <td>26.388330</td>
          <td>0.125753</td>
          <td>26.675706</td>
          <td>0.142260</td>
          <td>26.490157</td>
          <td>0.194654</td>
          <td>25.808783</td>
          <td>0.203269</td>
          <td>25.887743</td>
          <td>0.451686</td>
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
          <td>26.375502</td>
          <td>0.379693</td>
          <td>26.400496</td>
          <td>0.146245</td>
          <td>26.081827</td>
          <td>0.099624</td>
          <td>25.149965</td>
          <td>0.071893</td>
          <td>24.823481</td>
          <td>0.102099</td>
          <td>24.107879</td>
          <td>0.122607</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.479724</td>
          <td>0.411501</td>
          <td>27.348596</td>
          <td>0.321569</td>
          <td>26.762956</td>
          <td>0.179386</td>
          <td>25.931488</td>
          <td>0.142423</td>
          <td>25.908479</td>
          <td>0.257153</td>
          <td>25.421509</td>
          <td>0.365414</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.488202</td>
          <td>1.367336</td>
          <td>28.987436</td>
          <td>0.961558</td>
          <td>25.792171</td>
          <td>0.129152</td>
          <td>24.946033</td>
          <td>0.116170</td>
          <td>24.410569</td>
          <td>0.162723</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>34.284531</td>
          <td>7.022280</td>
          <td>28.243888</td>
          <td>0.660462</td>
          <td>27.373888</td>
          <td>0.316136</td>
          <td>26.378457</td>
          <td>0.222333</td>
          <td>25.391193</td>
          <td>0.177977</td>
          <td>25.257854</td>
          <td>0.341668</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.988533</td>
          <td>0.279374</td>
          <td>26.258118</td>
          <td>0.129396</td>
          <td>25.913267</td>
          <td>0.085944</td>
          <td>25.715813</td>
          <td>0.118188</td>
          <td>25.343023</td>
          <td>0.160127</td>
          <td>25.681906</td>
          <td>0.446405</td>
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
          <td>26.417828</td>
          <td>0.151212</td>
          <td>25.410337</td>
          <td>0.056247</td>
          <td>25.100474</td>
          <td>0.070338</td>
          <td>24.659694</td>
          <td>0.090312</td>
          <td>24.885072</td>
          <td>0.242074</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.452699</td>
          <td>0.821746</td>
          <td>26.914578</td>
          <td>0.226643</td>
          <td>26.055053</td>
          <td>0.097720</td>
          <td>25.192743</td>
          <td>0.074989</td>
          <td>24.751860</td>
          <td>0.096290</td>
          <td>24.141072</td>
          <td>0.126722</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.211715</td>
          <td>0.704136</td>
          <td>26.880823</td>
          <td>0.222005</td>
          <td>26.341428</td>
          <td>0.126513</td>
          <td>26.292358</td>
          <td>0.196119</td>
          <td>26.082586</td>
          <td>0.299649</td>
          <td>26.647664</td>
          <td>0.882433</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.993935</td>
          <td>0.612698</td>
          <td>26.182878</td>
          <td>0.124648</td>
          <td>25.950801</td>
          <td>0.091640</td>
          <td>25.840985</td>
          <td>0.135975</td>
          <td>25.845168</td>
          <td>0.251350</td>
          <td>25.182853</td>
          <td>0.311412</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.288995</td>
          <td>0.357348</td>
          <td>26.676471</td>
          <td>0.186627</td>
          <td>26.676660</td>
          <td>0.168300</td>
          <td>26.001587</td>
          <td>0.152776</td>
          <td>26.913283</td>
          <td>0.563788</td>
          <td>28.004330</td>
          <td>1.823251</td>
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
          <td>26.812859</td>
          <td>0.180952</td>
          <td>26.127710</td>
          <td>0.088253</td>
          <td>25.169373</td>
          <td>0.061706</td>
          <td>24.675791</td>
          <td>0.076279</td>
          <td>23.933308</td>
          <td>0.089144</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.283098</td>
          <td>2.012765</td>
          <td>27.470989</td>
          <td>0.311594</td>
          <td>26.519528</td>
          <td>0.124410</td>
          <td>26.364201</td>
          <td>0.175156</td>
          <td>25.866694</td>
          <td>0.213555</td>
          <td>25.282941</td>
          <td>0.281496</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.197833</td>
          <td>1.996363</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.669252</td>
          <td>0.349985</td>
          <td>26.228606</td>
          <td>0.169387</td>
          <td>25.207189</td>
          <td>0.131752</td>
          <td>24.331072</td>
          <td>0.137156</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.962389</td>
          <td>0.611352</td>
          <td>27.776484</td>
          <td>0.470707</td>
          <td>27.285601</td>
          <td>0.293559</td>
          <td>26.443248</td>
          <td>0.233796</td>
          <td>25.351425</td>
          <td>0.171476</td>
          <td>26.427566</td>
          <td>0.795905</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.294241</td>
          <td>0.320980</td>
          <td>25.972997</td>
          <td>0.087609</td>
          <td>25.962296</td>
          <td>0.076373</td>
          <td>25.781230</td>
          <td>0.105977</td>
          <td>25.643062</td>
          <td>0.176988</td>
          <td>24.755604</td>
          <td>0.181832</td>
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
          <td>26.286455</td>
          <td>0.123240</td>
          <td>25.366124</td>
          <td>0.048682</td>
          <td>25.142257</td>
          <td>0.065462</td>
          <td>24.694003</td>
          <td>0.083882</td>
          <td>24.689441</td>
          <td>0.185717</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.671502</td>
          <td>0.434407</td>
          <td>26.257934</td>
          <td>0.113893</td>
          <td>25.945345</td>
          <td>0.076387</td>
          <td>25.282598</td>
          <td>0.069412</td>
          <td>24.704922</td>
          <td>0.079563</td>
          <td>24.106931</td>
          <td>0.105582</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.907909</td>
          <td>0.528048</td>
          <td>26.863695</td>
          <td>0.196768</td>
          <td>26.677801</td>
          <td>0.149506</td>
          <td>26.263888</td>
          <td>0.168849</td>
          <td>25.408291</td>
          <td>0.151694</td>
          <td>25.083930</td>
          <td>0.250609</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.694200</td>
          <td>0.469466</td>
          <td>26.301695</td>
          <td>0.128874</td>
          <td>26.115472</td>
          <td>0.097912</td>
          <td>25.805802</td>
          <td>0.121754</td>
          <td>25.761294</td>
          <td>0.217847</td>
          <td>24.724769</td>
          <td>0.198261</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.839685</td>
          <td>0.499293</td>
          <td>26.873944</td>
          <td>0.196805</td>
          <td>26.566019</td>
          <td>0.134458</td>
          <td>26.186193</td>
          <td>0.156417</td>
          <td>25.929396</td>
          <td>0.233202</td>
          <td>26.980075</td>
          <td>0.982391</td>
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
