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

    <pzflow.flow.Flow at 0x7fac2c277d00>



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
          <td>26.535738</td>
          <td>0.387617</td>
          <td>26.823786</td>
          <td>0.182613</td>
          <td>25.999248</td>
          <td>0.078794</td>
          <td>25.339866</td>
          <td>0.071758</td>
          <td>25.170415</td>
          <td>0.117715</td>
          <td>24.649899</td>
          <td>0.165978</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.763748</td>
          <td>0.391980</td>
          <td>27.202486</td>
          <td>0.222314</td>
          <td>27.677803</td>
          <td>0.500605</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.102241</td>
          <td>0.274826</td>
          <td>25.976751</td>
          <td>0.087791</td>
          <td>24.809698</td>
          <td>0.027510</td>
          <td>23.887608</td>
          <td>0.020020</td>
          <td>23.168327</td>
          <td>0.020263</td>
          <td>22.849815</td>
          <td>0.034151</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.661835</td>
          <td>0.426975</td>
          <td>27.774116</td>
          <td>0.395129</td>
          <td>27.998200</td>
          <td>0.420211</td>
          <td>26.333637</td>
          <td>0.170500</td>
          <td>26.134566</td>
          <td>0.266190</td>
          <td>25.709178</td>
          <td>0.394179</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.666418</td>
          <td>0.428465</td>
          <td>25.730440</td>
          <td>0.070659</td>
          <td>25.419389</td>
          <td>0.047126</td>
          <td>24.746960</td>
          <td>0.042406</td>
          <td>24.301064</td>
          <td>0.054721</td>
          <td>23.805673</td>
          <td>0.079652</td>
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
          <td>27.216812</td>
          <td>0.639875</td>
          <td>26.560840</td>
          <td>0.145935</td>
          <td>26.140254</td>
          <td>0.089221</td>
          <td>26.033740</td>
          <td>0.131811</td>
          <td>25.801769</td>
          <td>0.202076</td>
          <td>25.138963</td>
          <td>0.250055</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.529125</td>
          <td>0.385639</td>
          <td>27.407253</td>
          <td>0.295835</td>
          <td>26.994853</td>
          <td>0.186792</td>
          <td>26.569462</td>
          <td>0.208055</td>
          <td>25.918963</td>
          <td>0.222862</td>
          <td>25.831944</td>
          <td>0.433024</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.242539</td>
          <td>0.258804</td>
          <td>26.858379</td>
          <td>0.166364</td>
          <td>26.964515</td>
          <td>0.288023</td>
          <td>26.121493</td>
          <td>0.263363</td>
          <td>26.700306</td>
          <td>0.800462</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.226689</td>
          <td>1.964771</td>
          <td>26.887558</td>
          <td>0.192711</td>
          <td>26.506682</td>
          <td>0.122915</td>
          <td>25.721807</td>
          <td>0.100458</td>
          <td>25.442179</td>
          <td>0.148892</td>
          <td>25.523411</td>
          <td>0.340942</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.449129</td>
          <td>0.749417</td>
          <td>26.451693</td>
          <td>0.132838</td>
          <td>26.025621</td>
          <td>0.080650</td>
          <td>25.706741</td>
          <td>0.099140</td>
          <td>25.147152</td>
          <td>0.115356</td>
          <td>24.828050</td>
          <td>0.193031</td>
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
          <td>27.064101</td>
          <td>0.631356</td>
          <td>26.696431</td>
          <td>0.188152</td>
          <td>25.971590</td>
          <td>0.090439</td>
          <td>25.379160</td>
          <td>0.088002</td>
          <td>24.820248</td>
          <td>0.101811</td>
          <td>24.893181</td>
          <td>0.238814</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.446409</td>
          <td>0.347464</td>
          <td>27.110488</td>
          <td>0.239932</td>
          <td>26.863622</td>
          <td>0.309880</td>
          <td>28.515256</td>
          <td>1.490358</td>
          <td>26.702751</td>
          <td>0.905085</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.264080</td>
          <td>0.733492</td>
          <td>25.956027</td>
          <td>0.101481</td>
          <td>24.791423</td>
          <td>0.032563</td>
          <td>23.836347</td>
          <td>0.023123</td>
          <td>23.133050</td>
          <td>0.023551</td>
          <td>22.834329</td>
          <td>0.040816</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.298372</td>
          <td>0.326432</td>
          <td>27.225440</td>
          <td>0.280542</td>
          <td>26.594221</td>
          <td>0.265596</td>
          <td>26.591605</td>
          <td>0.467012</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.871005</td>
          <td>0.550602</td>
          <td>25.787484</td>
          <td>0.085819</td>
          <td>25.519218</td>
          <td>0.060666</td>
          <td>24.831288</td>
          <td>0.054224</td>
          <td>24.503908</td>
          <td>0.077114</td>
          <td>23.681952</td>
          <td>0.084495</td>
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
          <td>26.135905</td>
          <td>0.319049</td>
          <td>26.525546</td>
          <td>0.165788</td>
          <td>26.103807</td>
          <td>0.103709</td>
          <td>26.104706</td>
          <td>0.168693</td>
          <td>25.660237</td>
          <td>0.213553</td>
          <td>25.815457</td>
          <td>0.502138</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.763739</td>
          <td>1.699365</td>
          <td>27.242758</td>
          <td>0.296407</td>
          <td>26.804443</td>
          <td>0.186505</td>
          <td>26.757348</td>
          <td>0.285547</td>
          <td>26.096989</td>
          <td>0.300762</td>
          <td>25.392077</td>
          <td>0.358391</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.993797</td>
          <td>1.891316</td>
          <td>27.376346</td>
          <td>0.332127</td>
          <td>26.778800</td>
          <td>0.184025</td>
          <td>26.335313</td>
          <td>0.203324</td>
          <td>25.967612</td>
          <td>0.273039</td>
          <td>25.429424</td>
          <td>0.371922</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.015656</td>
          <td>0.252125</td>
          <td>26.960990</td>
          <td>0.218291</td>
          <td>26.013831</td>
          <td>0.157748</td>
          <td>25.464848</td>
          <td>0.183024</td>
          <td>26.254205</td>
          <td>0.691502</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.006898</td>
          <td>0.285493</td>
          <td>26.740759</td>
          <td>0.197010</td>
          <td>26.182326</td>
          <td>0.109867</td>
          <td>25.748911</td>
          <td>0.122852</td>
          <td>25.358421</td>
          <td>0.163784</td>
          <td>24.904657</td>
          <td>0.243431</td>
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
          <td>26.205114</td>
          <td>0.298659</td>
          <td>26.744191</td>
          <td>0.170713</td>
          <td>25.939197</td>
          <td>0.074733</td>
          <td>25.351281</td>
          <td>0.072496</td>
          <td>25.188085</td>
          <td>0.119553</td>
          <td>25.029665</td>
          <td>0.228506</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.524418</td>
          <td>0.325157</td>
          <td>27.749205</td>
          <td>0.346679</td>
          <td>27.012521</td>
          <td>0.299664</td>
          <td>26.746827</td>
          <td>0.431945</td>
          <td>25.282528</td>
          <td>0.281402</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.975967</td>
          <td>0.261814</td>
          <td>25.977857</td>
          <td>0.094440</td>
          <td>24.780203</td>
          <td>0.029100</td>
          <td>23.844566</td>
          <td>0.020981</td>
          <td>23.132815</td>
          <td>0.021296</td>
          <td>22.791780</td>
          <td>0.035344</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.619059</td>
          <td>1.494311</td>
          <td>27.273889</td>
          <td>0.290798</td>
          <td>26.352238</td>
          <td>0.216774</td>
          <td>25.827225</td>
          <td>0.255218</td>
          <td>25.345530</td>
          <td>0.364837</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.969759</td>
          <td>0.246870</td>
          <td>25.714762</td>
          <td>0.069773</td>
          <td>25.504308</td>
          <td>0.050889</td>
          <td>24.857735</td>
          <td>0.046857</td>
          <td>24.323694</td>
          <td>0.055911</td>
          <td>23.861691</td>
          <td>0.083810</td>
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
          <td>27.616274</td>
          <td>0.870448</td>
          <td>26.289985</td>
          <td>0.123618</td>
          <td>26.099081</td>
          <td>0.093107</td>
          <td>25.900459</td>
          <td>0.127378</td>
          <td>25.841456</td>
          <td>0.225089</td>
          <td>25.707161</td>
          <td>0.422605</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.219753</td>
          <td>0.647022</td>
          <td>26.927327</td>
          <td>0.202009</td>
          <td>26.928418</td>
          <td>0.179404</td>
          <td>26.826861</td>
          <td>0.261677</td>
          <td>26.146189</td>
          <td>0.272853</td>
          <td>25.134939</td>
          <td>0.253246</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.490347</td>
          <td>0.385768</td>
          <td>27.279333</td>
          <td>0.277459</td>
          <td>27.275473</td>
          <td>0.247279</td>
          <td>26.375066</td>
          <td>0.185549</td>
          <td>25.877314</td>
          <td>0.225460</td>
          <td>25.667309</td>
          <td>0.399022</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.376538</td>
          <td>0.759864</td>
          <td>27.234640</td>
          <td>0.282335</td>
          <td>26.461033</td>
          <td>0.132298</td>
          <td>25.834436</td>
          <td>0.124818</td>
          <td>25.712325</td>
          <td>0.209120</td>
          <td>25.434267</td>
          <td>0.353478</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.037601</td>
          <td>0.576422</td>
          <td>26.717540</td>
          <td>0.172438</td>
          <td>26.185588</td>
          <td>0.096533</td>
          <td>25.681493</td>
          <td>0.101000</td>
          <td>24.960140</td>
          <td>0.101838</td>
          <td>25.410421</td>
          <td>0.323288</td>
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
