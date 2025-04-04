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

    <pzflow.flow.Flow at 0x7f86c666a620>



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
          <td>27.301600</td>
          <td>0.678420</td>
          <td>26.643823</td>
          <td>0.156693</td>
          <td>26.085815</td>
          <td>0.085046</td>
          <td>25.163086</td>
          <td>0.061355</td>
          <td>24.699852</td>
          <td>0.077907</td>
          <td>24.137002</td>
          <td>0.106563</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.638778</td>
          <td>1.497689</td>
          <td>27.603208</td>
          <td>0.345826</td>
          <td>26.669144</td>
          <td>0.141459</td>
          <td>26.378906</td>
          <td>0.177186</td>
          <td>25.583032</td>
          <td>0.167953</td>
          <td>25.702513</td>
          <td>0.392155</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.147268</td>
          <td>1.018059</td>
          <td>28.985998</td>
          <td>0.842910</td>
          <td>26.092457</td>
          <td>0.138668</td>
          <td>24.897140</td>
          <td>0.092694</td>
          <td>24.269096</td>
          <td>0.119566</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.306364</td>
          <td>0.242288</td>
          <td>26.479262</td>
          <td>0.192876</td>
          <td>25.564634</td>
          <td>0.165340</td>
          <td>25.299264</td>
          <td>0.284984</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.784510</td>
          <td>0.468344</td>
          <td>26.123784</td>
          <td>0.099875</td>
          <td>25.898601</td>
          <td>0.072088</td>
          <td>25.666504</td>
          <td>0.095702</td>
          <td>25.583344</td>
          <td>0.167997</td>
          <td>25.120406</td>
          <td>0.246268</td>
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
          <td>26.302314</td>
          <td>0.322761</td>
          <td>26.319665</td>
          <td>0.118480</td>
          <td>25.441573</td>
          <td>0.048063</td>
          <td>24.987299</td>
          <td>0.052491</td>
          <td>24.921601</td>
          <td>0.094706</td>
          <td>24.528349</td>
          <td>0.149587</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.500212</td>
          <td>1.395727</td>
          <td>26.606817</td>
          <td>0.151807</td>
          <td>26.131893</td>
          <td>0.088567</td>
          <td>25.230252</td>
          <td>0.065120</td>
          <td>24.808084</td>
          <td>0.085709</td>
          <td>24.434295</td>
          <td>0.137956</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.124676</td>
          <td>0.279871</td>
          <td>26.597899</td>
          <td>0.150651</td>
          <td>26.315075</td>
          <td>0.104011</td>
          <td>25.944816</td>
          <td>0.122033</td>
          <td>25.836795</td>
          <td>0.208096</td>
          <td>25.714831</td>
          <td>0.395902</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.364843</td>
          <td>0.339155</td>
          <td>26.203964</td>
          <td>0.107124</td>
          <td>26.096973</td>
          <td>0.085886</td>
          <td>25.791006</td>
          <td>0.106728</td>
          <td>25.961857</td>
          <td>0.230941</td>
          <td>26.329983</td>
          <td>0.623111</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.671310</td>
          <td>0.160417</td>
          <td>26.358879</td>
          <td>0.108070</td>
          <td>26.287114</td>
          <td>0.163874</td>
          <td>25.977889</td>
          <td>0.234027</td>
          <td>26.530690</td>
          <td>0.715323</td>
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
          <td>26.341918</td>
          <td>0.139060</td>
          <td>26.087270</td>
          <td>0.100100</td>
          <td>25.022668</td>
          <td>0.064231</td>
          <td>24.720432</td>
          <td>0.093281</td>
          <td>24.155624</td>
          <td>0.127788</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.988340</td>
          <td>1.135945</td>
          <td>27.407493</td>
          <td>0.336955</td>
          <td>26.775550</td>
          <td>0.181310</td>
          <td>26.298641</td>
          <td>0.194715</td>
          <td>26.878434</td>
          <td>0.545389</td>
          <td>25.006279</td>
          <td>0.262119</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.613919</td>
          <td>0.919227</td>
          <td>29.972894</td>
          <td>1.734927</td>
          <td>27.572036</td>
          <td>0.355197</td>
          <td>26.020616</td>
          <td>0.157216</td>
          <td>25.026777</td>
          <td>0.124611</td>
          <td>24.461392</td>
          <td>0.169925</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.964746</td>
          <td>0.226504</td>
          <td>26.120595</td>
          <td>0.179043</td>
          <td>25.747324</td>
          <td>0.239783</td>
          <td>25.012436</td>
          <td>0.280737</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.887772</td>
          <td>0.557295</td>
          <td>26.138395</td>
          <td>0.116639</td>
          <td>25.940361</td>
          <td>0.088018</td>
          <td>25.703318</td>
          <td>0.116911</td>
          <td>25.407851</td>
          <td>0.169228</td>
          <td>24.731336</td>
          <td>0.208820</td>
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
          <td>30.207813</td>
          <td>2.975682</td>
          <td>26.433038</td>
          <td>0.153195</td>
          <td>25.493733</td>
          <td>0.060564</td>
          <td>25.114026</td>
          <td>0.071186</td>
          <td>24.832266</td>
          <td>0.105060</td>
          <td>25.183969</td>
          <td>0.308669</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.701657</td>
          <td>0.487645</td>
          <td>26.574393</td>
          <td>0.170295</td>
          <td>25.961846</td>
          <td>0.090043</td>
          <td>25.127184</td>
          <td>0.070766</td>
          <td>24.881299</td>
          <td>0.107839</td>
          <td>24.205037</td>
          <td>0.133933</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.537453</td>
          <td>0.433590</td>
          <td>26.703361</td>
          <td>0.191363</td>
          <td>26.461538</td>
          <td>0.140350</td>
          <td>26.595420</td>
          <td>0.252318</td>
          <td>26.373668</td>
          <td>0.377247</td>
          <td>25.764660</td>
          <td>0.480177</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.871488</td>
          <td>0.259668</td>
          <td>26.383493</td>
          <td>0.148186</td>
          <td>25.978558</td>
          <td>0.093901</td>
          <td>25.857567</td>
          <td>0.137934</td>
          <td>25.473064</td>
          <td>0.184300</td>
          <td>24.721157</td>
          <td>0.213435</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.152239</td>
          <td>0.320776</td>
          <td>27.336504</td>
          <td>0.321057</td>
          <td>26.605402</td>
          <td>0.158372</td>
          <td>26.470189</td>
          <td>0.226929</td>
          <td>25.259816</td>
          <td>0.150535</td>
          <td>25.800241</td>
          <td>0.491853</td>
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
          <td>29.426852</td>
          <td>2.134656</td>
          <td>26.679538</td>
          <td>0.161566</td>
          <td>26.116970</td>
          <td>0.087423</td>
          <td>25.145468</td>
          <td>0.060412</td>
          <td>24.669174</td>
          <td>0.075834</td>
          <td>24.014118</td>
          <td>0.095704</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.957049</td>
          <td>1.032375</td>
          <td>27.475807</td>
          <td>0.312797</td>
          <td>26.671174</td>
          <td>0.141838</td>
          <td>26.249901</td>
          <td>0.158901</td>
          <td>25.494435</td>
          <td>0.155858</td>
          <td>26.153325</td>
          <td>0.549942</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.351060</td>
          <td>1.200539</td>
          <td>27.959343</td>
          <td>0.437900</td>
          <td>25.993749</td>
          <td>0.138508</td>
          <td>25.089053</td>
          <td>0.118924</td>
          <td>24.100840</td>
          <td>0.112326</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.195958</td>
          <td>0.718017</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.515982</td>
          <td>0.352664</td>
          <td>26.525117</td>
          <td>0.250124</td>
          <td>25.599096</td>
          <td>0.211295</td>
          <td>25.869015</td>
          <td>0.541488</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.949573</td>
          <td>0.242807</td>
          <td>26.249679</td>
          <td>0.111617</td>
          <td>25.891758</td>
          <td>0.071756</td>
          <td>25.682934</td>
          <td>0.097237</td>
          <td>25.189504</td>
          <td>0.119853</td>
          <td>24.995401</td>
          <td>0.222375</td>
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
          <td>27.620552</td>
          <td>0.872808</td>
          <td>26.294768</td>
          <td>0.124132</td>
          <td>25.401345</td>
          <td>0.050228</td>
          <td>25.163796</td>
          <td>0.066723</td>
          <td>24.727019</td>
          <td>0.086357</td>
          <td>24.707067</td>
          <td>0.188502</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.809522</td>
          <td>0.481820</td>
          <td>26.583580</td>
          <td>0.150908</td>
          <td>25.966049</td>
          <td>0.077797</td>
          <td>25.103290</td>
          <td>0.059211</td>
          <td>24.771974</td>
          <td>0.084409</td>
          <td>24.138820</td>
          <td>0.108564</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.927885</td>
          <td>1.038062</td>
          <td>26.460295</td>
          <td>0.139564</td>
          <td>26.475877</td>
          <td>0.125599</td>
          <td>26.245324</td>
          <td>0.166200</td>
          <td>26.144911</td>
          <td>0.280854</td>
          <td>25.919085</td>
          <td>0.482802</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.770797</td>
          <td>0.496938</td>
          <td>26.124783</td>
          <td>0.110524</td>
          <td>26.036598</td>
          <td>0.091363</td>
          <td>25.875217</td>
          <td>0.129308</td>
          <td>25.593210</td>
          <td>0.189204</td>
          <td>24.717326</td>
          <td>0.197024</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.970555</td>
          <td>0.549331</td>
          <td>26.613242</td>
          <td>0.157774</td>
          <td>26.758072</td>
          <td>0.158595</td>
          <td>26.192715</td>
          <td>0.157293</td>
          <td>26.161209</td>
          <td>0.281979</td>
          <td>25.595953</td>
          <td>0.374142</td>
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
