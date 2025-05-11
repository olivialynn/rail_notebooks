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

    <pzflow.flow.Flow at 0x7fd7fa9932e0>



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
          <td>26.706775</td>
          <td>0.165344</td>
          <td>26.018814</td>
          <td>0.080167</td>
          <td>25.222079</td>
          <td>0.064650</td>
          <td>24.571181</td>
          <td>0.069529</td>
          <td>24.100972</td>
          <td>0.103258</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.015417</td>
          <td>0.554842</td>
          <td>27.654388</td>
          <td>0.360014</td>
          <td>26.342120</td>
          <td>0.106499</td>
          <td>26.156592</td>
          <td>0.146542</td>
          <td>26.157461</td>
          <td>0.271204</td>
          <td>25.403118</td>
          <td>0.309835</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.630203</td>
          <td>0.843401</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.886892</td>
          <td>0.385735</td>
          <td>25.851248</td>
          <td>0.112491</td>
          <td>25.092012</td>
          <td>0.109942</td>
          <td>24.126901</td>
          <td>0.105626</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.519313</td>
          <td>0.680565</td>
          <td>27.743428</td>
          <td>0.344811</td>
          <td>26.614637</td>
          <td>0.216059</td>
          <td>25.523986</td>
          <td>0.159701</td>
          <td>24.905062</td>
          <td>0.205933</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.154054</td>
          <td>0.286601</td>
          <td>26.056298</td>
          <td>0.094141</td>
          <td>25.942718</td>
          <td>0.074956</td>
          <td>25.732480</td>
          <td>0.101401</td>
          <td>25.482177</td>
          <td>0.154088</td>
          <td>25.189209</td>
          <td>0.260572</td>
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
          <td>26.340952</td>
          <td>0.120691</td>
          <td>25.506182</td>
          <td>0.050901</td>
          <td>25.090202</td>
          <td>0.057512</td>
          <td>24.833688</td>
          <td>0.087663</td>
          <td>24.630659</td>
          <td>0.163277</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.049558</td>
          <td>0.568618</td>
          <td>26.933694</td>
          <td>0.200332</td>
          <td>25.976410</td>
          <td>0.077221</td>
          <td>25.261037</td>
          <td>0.066921</td>
          <td>24.848167</td>
          <td>0.088787</td>
          <td>24.023263</td>
          <td>0.096462</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.549909</td>
          <td>0.391884</td>
          <td>26.399018</td>
          <td>0.126922</td>
          <td>26.290399</td>
          <td>0.101788</td>
          <td>26.061482</td>
          <td>0.135010</td>
          <td>26.031040</td>
          <td>0.244527</td>
          <td>25.028526</td>
          <td>0.228261</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.826187</td>
          <td>0.219045</td>
          <td>26.122337</td>
          <td>0.099748</td>
          <td>26.147124</td>
          <td>0.089762</td>
          <td>25.676087</td>
          <td>0.096511</td>
          <td>25.845744</td>
          <td>0.209660</td>
          <td>25.323631</td>
          <td>0.290653</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.872364</td>
          <td>0.499902</td>
          <td>26.712064</td>
          <td>0.166091</td>
          <td>26.652648</td>
          <td>0.139462</td>
          <td>26.291935</td>
          <td>0.164550</td>
          <td>26.319816</td>
          <td>0.309203</td>
          <td>25.040824</td>
          <td>0.230600</td>
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
          <td>28.792825</td>
          <td>1.719565</td>
          <td>26.611859</td>
          <td>0.175162</td>
          <td>26.010442</td>
          <td>0.093579</td>
          <td>25.166639</td>
          <td>0.072960</td>
          <td>24.620960</td>
          <td>0.085470</td>
          <td>24.016416</td>
          <td>0.113233</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.824264</td>
          <td>0.464558</td>
          <td>26.482325</td>
          <td>0.141134</td>
          <td>26.233096</td>
          <td>0.184240</td>
          <td>25.962690</td>
          <td>0.268799</td>
          <td>25.359093</td>
          <td>0.347953</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.059381</td>
          <td>1.804457</td>
          <td>28.101908</td>
          <td>0.530671</td>
          <td>25.850643</td>
          <td>0.135848</td>
          <td>25.052964</td>
          <td>0.127471</td>
          <td>24.344485</td>
          <td>0.153784</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.759642</td>
          <td>0.529971</td>
          <td>28.188834</td>
          <td>0.635730</td>
          <td>28.222013</td>
          <td>0.600024</td>
          <td>26.475716</td>
          <td>0.240985</td>
          <td>25.262607</td>
          <td>0.159526</td>
          <td>25.393359</td>
          <td>0.379919</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.219038</td>
          <td>0.702480</td>
          <td>26.089935</td>
          <td>0.111824</td>
          <td>26.056072</td>
          <td>0.097432</td>
          <td>25.784357</td>
          <td>0.125436</td>
          <td>25.632436</td>
          <td>0.204581</td>
          <td>26.155028</td>
          <td>0.629798</td>
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
          <td>26.665734</td>
          <td>0.479999</td>
          <td>26.353772</td>
          <td>0.143123</td>
          <td>25.474993</td>
          <td>0.059566</td>
          <td>25.249248</td>
          <td>0.080217</td>
          <td>24.773335</td>
          <td>0.099781</td>
          <td>24.978566</td>
          <td>0.261389</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.882569</td>
          <td>0.556543</td>
          <td>26.708383</td>
          <td>0.190745</td>
          <td>26.139043</td>
          <td>0.105176</td>
          <td>25.237191</td>
          <td>0.077991</td>
          <td>24.953856</td>
          <td>0.114881</td>
          <td>24.510536</td>
          <td>0.174010</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.629359</td>
          <td>0.923299</td>
          <td>26.616690</td>
          <td>0.177851</td>
          <td>26.397296</td>
          <td>0.132781</td>
          <td>26.333754</td>
          <td>0.203058</td>
          <td>25.978409</td>
          <td>0.275446</td>
          <td>25.569492</td>
          <td>0.414411</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.505838</td>
          <td>0.428569</td>
          <td>26.191380</td>
          <td>0.125569</td>
          <td>26.111069</td>
          <td>0.105458</td>
          <td>25.941365</td>
          <td>0.148249</td>
          <td>25.806941</td>
          <td>0.243569</td>
          <td>25.475212</td>
          <td>0.391934</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.421493</td>
          <td>0.396090</td>
          <td>26.797247</td>
          <td>0.206567</td>
          <td>26.522852</td>
          <td>0.147554</td>
          <td>25.958050</td>
          <td>0.147173</td>
          <td>25.950543</td>
          <td>0.268587</td>
          <td>25.308825</td>
          <td>0.337467</td>
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
          <td>30.033864</td>
          <td>2.674490</td>
          <td>26.872006</td>
          <td>0.190222</td>
          <td>26.024577</td>
          <td>0.080586</td>
          <td>25.266297</td>
          <td>0.067242</td>
          <td>24.565175</td>
          <td>0.069170</td>
          <td>24.131374</td>
          <td>0.106054</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.240129</td>
          <td>0.258492</td>
          <td>26.683981</td>
          <td>0.143410</td>
          <td>26.463504</td>
          <td>0.190513</td>
          <td>26.155438</td>
          <td>0.270995</td>
          <td>25.743909</td>
          <td>0.405213</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.171792</td>
          <td>1.212850</td>
          <td>28.739512</td>
          <td>0.831634</td>
          <td>28.634049</td>
          <td>0.710853</td>
          <td>25.986410</td>
          <td>0.137633</td>
          <td>25.206476</td>
          <td>0.131671</td>
          <td>24.274935</td>
          <td>0.130662</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.578377</td>
          <td>0.823700</td>
          <td>27.075748</td>
          <td>0.247432</td>
          <td>26.063575</td>
          <td>0.169975</td>
          <td>25.700296</td>
          <td>0.229862</td>
          <td>25.442176</td>
          <td>0.393286</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.239528</td>
          <td>0.307266</td>
          <td>26.278767</td>
          <td>0.114480</td>
          <td>26.049525</td>
          <td>0.082486</td>
          <td>25.673180</td>
          <td>0.096408</td>
          <td>25.478263</td>
          <td>0.153785</td>
          <td>25.007451</td>
          <td>0.224614</td>
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
          <td>26.418825</td>
          <td>0.371904</td>
          <td>26.254521</td>
          <td>0.119873</td>
          <td>25.403236</td>
          <td>0.050313</td>
          <td>25.036294</td>
          <td>0.059592</td>
          <td>24.828045</td>
          <td>0.094377</td>
          <td>24.898133</td>
          <td>0.221244</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.449960</td>
          <td>0.756357</td>
          <td>26.747576</td>
          <td>0.173574</td>
          <td>26.100827</td>
          <td>0.087614</td>
          <td>25.210191</td>
          <td>0.065099</td>
          <td>24.788712</td>
          <td>0.085663</td>
          <td>24.311369</td>
          <td>0.126151</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.213003</td>
          <td>0.310126</td>
          <td>26.663944</td>
          <td>0.166165</td>
          <td>26.252687</td>
          <td>0.103405</td>
          <td>26.380141</td>
          <td>0.186347</td>
          <td>25.995246</td>
          <td>0.248543</td>
          <td>25.407986</td>
          <td>0.325699</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.940628</td>
          <td>0.260026</td>
          <td>26.108623</td>
          <td>0.108978</td>
          <td>26.094656</td>
          <td>0.096141</td>
          <td>25.889412</td>
          <td>0.130906</td>
          <td>26.041084</td>
          <td>0.274294</td>
          <td>25.401798</td>
          <td>0.344559</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.990577</td>
          <td>0.557317</td>
          <td>26.665400</td>
          <td>0.164956</td>
          <td>26.572833</td>
          <td>0.135252</td>
          <td>26.272089</td>
          <td>0.168319</td>
          <td>25.802438</td>
          <td>0.209822</td>
          <td>25.491524</td>
          <td>0.344743</td>
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
