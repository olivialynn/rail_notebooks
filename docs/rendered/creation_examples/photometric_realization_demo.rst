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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f137d84bd90>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>27.027442</td>
          <td>0.559665</td>
          <td>26.724305</td>
          <td>0.167831</td>
          <td>26.111878</td>
          <td>0.087020</td>
          <td>25.347995</td>
          <td>0.072276</td>
          <td>25.034425</td>
          <td>0.104547</td>
          <td>24.744778</td>
          <td>0.179916</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.930956</td>
          <td>0.445381</td>
          <td>27.259629</td>
          <td>0.233110</td>
          <td>27.752823</td>
          <td>0.528914</td>
          <td>26.347296</td>
          <td>0.316073</td>
          <td>27.630707</td>
          <td>1.381093</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.081239</td>
          <td>0.270176</td>
          <td>26.073357</td>
          <td>0.095560</td>
          <td>24.830202</td>
          <td>0.028007</td>
          <td>23.843563</td>
          <td>0.019288</td>
          <td>23.077119</td>
          <td>0.018762</td>
          <td>22.837877</td>
          <td>0.033793</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.012889</td>
          <td>1.066497</td>
          <td>28.020961</td>
          <td>0.476486</td>
          <td>27.547787</td>
          <td>0.295006</td>
          <td>26.741575</td>
          <td>0.240059</td>
          <td>25.532751</td>
          <td>0.160902</td>
          <td>24.909651</td>
          <td>0.206726</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.245228</td>
          <td>0.308394</td>
          <td>25.708908</td>
          <td>0.069327</td>
          <td>25.424190</td>
          <td>0.047327</td>
          <td>24.821730</td>
          <td>0.045315</td>
          <td>24.353106</td>
          <td>0.057308</td>
          <td>23.718848</td>
          <td>0.073771</td>
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
          <td>2.147172</td>
          <td>26.460965</td>
          <td>0.365739</td>
          <td>26.172952</td>
          <td>0.104262</td>
          <td>26.202846</td>
          <td>0.094267</td>
          <td>26.230876</td>
          <td>0.156184</td>
          <td>26.576234</td>
          <td>0.378562</td>
          <td>25.358332</td>
          <td>0.298896</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.164176</td>
          <td>0.242674</td>
          <td>26.693958</td>
          <td>0.144513</td>
          <td>26.262397</td>
          <td>0.160452</td>
          <td>27.094320</td>
          <td>0.558185</td>
          <td>26.098910</td>
          <td>0.528219</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.841577</td>
          <td>0.962582</td>
          <td>27.323378</td>
          <td>0.276434</td>
          <td>26.795322</td>
          <td>0.157643</td>
          <td>26.828032</td>
          <td>0.257747</td>
          <td>26.126878</td>
          <td>0.264524</td>
          <td>25.487128</td>
          <td>0.331289</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.383597</td>
          <td>0.290246</td>
          <td>26.589887</td>
          <td>0.132105</td>
          <td>25.854445</td>
          <td>0.112804</td>
          <td>25.660627</td>
          <td>0.179398</td>
          <td>26.510559</td>
          <td>0.705658</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.010536</td>
          <td>0.552894</td>
          <td>26.664359</td>
          <td>0.159468</td>
          <td>26.051294</td>
          <td>0.082497</td>
          <td>25.590822</td>
          <td>0.089545</td>
          <td>25.176510</td>
          <td>0.118341</td>
          <td>25.615551</td>
          <td>0.366535</td>
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
          <td>0.890625</td>
          <td>27.299263</td>
          <td>0.741268</td>
          <td>26.816064</td>
          <td>0.208043</td>
          <td>26.268104</td>
          <td>0.117218</td>
          <td>25.271832</td>
          <td>0.080063</td>
          <td>25.114791</td>
          <td>0.131556</td>
          <td>24.794638</td>
          <td>0.220078</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.916196</td>
          <td>0.497426</td>
          <td>27.550371</td>
          <td>0.342350</td>
          <td>27.261667</td>
          <td>0.423053</td>
          <td>26.698512</td>
          <td>0.477888</td>
          <td>26.284329</td>
          <td>0.688529</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.304812</td>
          <td>0.364841</td>
          <td>25.904482</td>
          <td>0.097006</td>
          <td>24.713691</td>
          <td>0.030413</td>
          <td>23.912941</td>
          <td>0.024703</td>
          <td>23.169337</td>
          <td>0.024300</td>
          <td>22.855014</td>
          <td>0.041570</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.220198</td>
          <td>1.332992</td>
          <td>28.005149</td>
          <td>0.558169</td>
          <td>27.291241</td>
          <td>0.295863</td>
          <td>26.808499</td>
          <td>0.315766</td>
          <td>25.906549</td>
          <td>0.273202</td>
          <td>25.295471</td>
          <td>0.351945</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.789150</td>
          <td>0.518815</td>
          <td>25.943803</td>
          <td>0.098431</td>
          <td>25.426798</td>
          <td>0.055892</td>
          <td>24.796194</td>
          <td>0.052561</td>
          <td>24.485431</td>
          <td>0.075867</td>
          <td>23.710539</td>
          <td>0.086649</td>
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
          <td>2.147172</td>
          <td>26.735103</td>
          <td>0.505262</td>
          <td>26.139682</td>
          <td>0.118949</td>
          <td>25.938381</td>
          <td>0.089704</td>
          <td>26.227607</td>
          <td>0.187221</td>
          <td>26.764949</td>
          <td>0.510803</td>
          <td>24.982502</td>
          <td>0.262232</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.952656</td>
          <td>0.233906</td>
          <td>26.961588</td>
          <td>0.212824</td>
          <td>26.104961</td>
          <td>0.165906</td>
          <td>26.120805</td>
          <td>0.306567</td>
          <td>25.881195</td>
          <td>0.519423</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.232479</td>
          <td>0.296067</td>
          <td>26.695892</td>
          <td>0.171532</td>
          <td>26.084861</td>
          <td>0.164504</td>
          <td>26.274055</td>
          <td>0.348964</td>
          <td>25.729077</td>
          <td>0.467599</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.501004</td>
          <td>0.371877</td>
          <td>26.577896</td>
          <td>0.157943</td>
          <td>25.923185</td>
          <td>0.145952</td>
          <td>25.223514</td>
          <td>0.148995</td>
          <td>26.165314</td>
          <td>0.650545</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.526986</td>
          <td>0.164403</td>
          <td>26.159604</td>
          <td>0.107709</td>
          <td>25.810547</td>
          <td>0.129593</td>
          <td>25.206505</td>
          <td>0.143796</td>
          <td>24.826804</td>
          <td>0.228255</td>
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
          <td>0.890625</td>
          <td>27.871017</td>
          <td>0.980041</td>
          <td>26.505806</td>
          <td>0.139201</td>
          <td>26.035701</td>
          <td>0.081381</td>
          <td>25.258077</td>
          <td>0.066755</td>
          <td>25.102354</td>
          <td>0.110953</td>
          <td>25.178287</td>
          <td>0.258286</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.702311</td>
          <td>1.546131</td>
          <td>30.694043</td>
          <td>2.182365</td>
          <td>27.850414</td>
          <td>0.375275</td>
          <td>27.175258</td>
          <td>0.341171</td>
          <td>26.337016</td>
          <td>0.313757</td>
          <td>27.226990</td>
          <td>1.107182</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.311594</td>
          <td>0.714642</td>
          <td>25.905553</td>
          <td>0.088634</td>
          <td>24.814437</td>
          <td>0.029987</td>
          <td>23.826220</td>
          <td>0.020656</td>
          <td>23.116755</td>
          <td>0.021007</td>
          <td>22.827294</td>
          <td>0.036470</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.633580</td>
          <td>0.853391</td>
          <td>26.908318</td>
          <td>0.215384</td>
          <td>26.636360</td>
          <td>0.273935</td>
          <td>25.664692</td>
          <td>0.223168</td>
          <td>25.313243</td>
          <td>0.355727</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.039060</td>
          <td>0.261286</td>
          <td>25.781831</td>
          <td>0.074031</td>
          <td>25.394403</td>
          <td>0.046158</td>
          <td>24.794501</td>
          <td>0.044299</td>
          <td>24.193700</td>
          <td>0.049818</td>
          <td>23.680727</td>
          <td>0.071432</td>
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
          <td>2.147172</td>
          <td>27.229554</td>
          <td>0.674473</td>
          <td>26.554902</td>
          <td>0.155308</td>
          <td>25.988688</td>
          <td>0.084491</td>
          <td>26.092002</td>
          <td>0.150257</td>
          <td>25.672356</td>
          <td>0.195407</td>
          <td>24.863046</td>
          <td>0.214869</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.457762</td>
          <td>0.135430</td>
          <td>26.767300</td>
          <td>0.156397</td>
          <td>26.474050</td>
          <td>0.195239</td>
          <td>26.201428</td>
          <td>0.285361</td>
          <td>26.473949</td>
          <td>0.697495</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.740645</td>
          <td>0.926713</td>
          <td>27.139478</td>
          <td>0.247498</td>
          <td>26.811750</td>
          <td>0.167651</td>
          <td>26.157371</td>
          <td>0.154166</td>
          <td>26.126350</td>
          <td>0.276655</td>
          <td>25.425609</td>
          <td>0.330290</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.735616</td>
          <td>0.955243</td>
          <td>27.409529</td>
          <td>0.324883</td>
          <td>26.511172</td>
          <td>0.138152</td>
          <td>25.873467</td>
          <td>0.129112</td>
          <td>25.406184</td>
          <td>0.161426</td>
          <td>25.913600</td>
          <td>0.509104</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.695742</td>
          <td>0.448487</td>
          <td>26.421794</td>
          <td>0.133844</td>
          <td>26.147299</td>
          <td>0.093342</td>
          <td>25.788688</td>
          <td>0.110921</td>
          <td>25.162982</td>
          <td>0.121544</td>
          <td>24.977942</td>
          <td>0.227385</td>
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
