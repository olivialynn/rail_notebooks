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

    <pzflow.flow.Flow at 0x7fe832992560>



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
          <td>27.325298</td>
          <td>0.689488</td>
          <td>26.725172</td>
          <td>0.167955</td>
          <td>26.167674</td>
          <td>0.091398</td>
          <td>25.287023</td>
          <td>0.068479</td>
          <td>25.021738</td>
          <td>0.103394</td>
          <td>24.879269</td>
          <td>0.201527</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.340604</td>
          <td>2.060823</td>
          <td>28.390283</td>
          <td>0.622398</td>
          <td>28.147985</td>
          <td>0.470545</td>
          <td>26.877163</td>
          <td>0.268306</td>
          <td>26.334222</td>
          <td>0.312788</td>
          <td>26.661827</td>
          <td>0.780567</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.682190</td>
          <td>0.433625</td>
          <td>25.897530</td>
          <td>0.081880</td>
          <td>24.771968</td>
          <td>0.026619</td>
          <td>23.850419</td>
          <td>0.019400</td>
          <td>23.120981</td>
          <td>0.019467</td>
          <td>22.793255</td>
          <td>0.032490</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.742214</td>
          <td>0.790043</td>
          <td>28.945288</td>
          <td>0.821124</td>
          <td>26.763457</td>
          <td>0.244430</td>
          <td>25.724449</td>
          <td>0.189347</td>
          <td>25.405891</td>
          <td>0.310523</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.427998</td>
          <td>0.356429</td>
          <td>25.799886</td>
          <td>0.075127</td>
          <td>25.408222</td>
          <td>0.046661</td>
          <td>24.810850</td>
          <td>0.044879</td>
          <td>24.419217</td>
          <td>0.060770</td>
          <td>23.665660</td>
          <td>0.070381</td>
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
          <td>26.346166</td>
          <td>0.334185</td>
          <td>26.473058</td>
          <td>0.135311</td>
          <td>26.126826</td>
          <td>0.088173</td>
          <td>26.545679</td>
          <td>0.203951</td>
          <td>25.649946</td>
          <td>0.177781</td>
          <td>25.465679</td>
          <td>0.325693</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.162840</td>
          <td>1.162658</td>
          <td>26.985461</td>
          <td>0.209211</td>
          <td>27.152715</td>
          <td>0.213280</td>
          <td>26.570016</td>
          <td>0.208151</td>
          <td>25.937392</td>
          <td>0.226301</td>
          <td>25.619027</td>
          <td>0.367532</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.888691</td>
          <td>0.505948</td>
          <td>27.034967</td>
          <td>0.218036</td>
          <td>26.773370</td>
          <td>0.154708</td>
          <td>26.361782</td>
          <td>0.174629</td>
          <td>25.916549</td>
          <td>0.222415</td>
          <td>25.591500</td>
          <td>0.359703</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.303530</td>
          <td>0.679317</td>
          <td>27.651256</td>
          <td>0.359131</td>
          <td>26.675561</td>
          <td>0.142242</td>
          <td>25.965512</td>
          <td>0.124246</td>
          <td>25.650968</td>
          <td>0.177935</td>
          <td>25.066660</td>
          <td>0.235586</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.567505</td>
          <td>0.397237</td>
          <td>26.723332</td>
          <td>0.167692</td>
          <td>26.203641</td>
          <td>0.094332</td>
          <td>25.625479</td>
          <td>0.092316</td>
          <td>25.269279</td>
          <td>0.128264</td>
          <td>25.034187</td>
          <td>0.229335</td>
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
          <td>29.946177</td>
          <td>2.716252</td>
          <td>26.788740</td>
          <td>0.203338</td>
          <td>26.063625</td>
          <td>0.098048</td>
          <td>25.397725</td>
          <td>0.089451</td>
          <td>24.850133</td>
          <td>0.104507</td>
          <td>24.934694</td>
          <td>0.247126</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.579687</td>
          <td>0.889041</td>
          <td>28.839868</td>
          <td>0.932348</td>
          <td>30.735850</td>
          <td>2.252972</td>
          <td>27.100596</td>
          <td>0.373668</td>
          <td>26.810875</td>
          <td>0.519222</td>
          <td>25.411166</td>
          <td>0.362471</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.791314</td>
          <td>0.526941</td>
          <td>26.024969</td>
          <td>0.107777</td>
          <td>24.852924</td>
          <td>0.034375</td>
          <td>23.876389</td>
          <td>0.023935</td>
          <td>23.149347</td>
          <td>0.023884</td>
          <td>22.805136</td>
          <td>0.039775</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.244397</td>
          <td>0.312690</td>
          <td>27.357950</td>
          <td>0.312136</td>
          <td>26.737786</td>
          <td>0.298366</td>
          <td>25.502040</td>
          <td>0.195445</td>
          <td>24.948183</td>
          <td>0.266444</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.412376</td>
          <td>0.390770</td>
          <td>25.826091</td>
          <td>0.088780</td>
          <td>25.479568</td>
          <td>0.058570</td>
          <td>24.749833</td>
          <td>0.050443</td>
          <td>24.406247</td>
          <td>0.070739</td>
          <td>23.656496</td>
          <td>0.082622</td>
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
          <td>26.113584</td>
          <td>0.313422</td>
          <td>26.380628</td>
          <td>0.146464</td>
          <td>25.914052</td>
          <td>0.087805</td>
          <td>25.951865</td>
          <td>0.148025</td>
          <td>25.865076</td>
          <td>0.253019</td>
          <td>24.933462</td>
          <td>0.251906</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.805909</td>
          <td>0.526480</td>
          <td>27.344958</td>
          <td>0.321679</td>
          <td>26.777590</td>
          <td>0.182319</td>
          <td>26.248244</td>
          <td>0.187352</td>
          <td>25.485969</td>
          <td>0.181502</td>
          <td>25.836929</td>
          <td>0.502811</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.386167</td>
          <td>0.790814</td>
          <td>27.160507</td>
          <td>0.279341</td>
          <td>26.605376</td>
          <td>0.158793</td>
          <td>26.892186</td>
          <td>0.320803</td>
          <td>26.073108</td>
          <td>0.297372</td>
          <td>25.557871</td>
          <td>0.410740</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.732790</td>
          <td>0.444269</td>
          <td>26.452135</td>
          <td>0.141785</td>
          <td>25.746441</td>
          <td>0.125295</td>
          <td>25.292476</td>
          <td>0.158063</td>
          <td>26.044109</td>
          <td>0.597600</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.362169</td>
          <td>0.378330</td>
          <td>26.524477</td>
          <td>0.164051</td>
          <td>26.137052</td>
          <td>0.105608</td>
          <td>25.530373</td>
          <td>0.101541</td>
          <td>25.093498</td>
          <td>0.130438</td>
          <td>24.688646</td>
          <td>0.203411</td>
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
          <td>25.995060</td>
          <td>0.251839</td>
          <td>26.692043</td>
          <td>0.163299</td>
          <td>25.933813</td>
          <td>0.074378</td>
          <td>25.344098</td>
          <td>0.072037</td>
          <td>24.904643</td>
          <td>0.093319</td>
          <td>25.372681</td>
          <td>0.302401</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.311634</td>
          <td>0.683438</td>
          <td>27.949886</td>
          <td>0.452101</td>
          <td>28.090384</td>
          <td>0.451008</td>
          <td>27.615539</td>
          <td>0.478431</td>
          <td>27.197462</td>
          <td>0.601272</td>
          <td>27.502635</td>
          <td>1.291250</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.584055</td>
          <td>0.423363</td>
          <td>26.000026</td>
          <td>0.096292</td>
          <td>24.776791</td>
          <td>0.029013</td>
          <td>23.855578</td>
          <td>0.021179</td>
          <td>23.117288</td>
          <td>0.021016</td>
          <td>22.838997</td>
          <td>0.036849</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.216490</td>
          <td>0.277595</td>
          <td>26.704765</td>
          <td>0.289550</td>
          <td>25.835660</td>
          <td>0.256988</td>
          <td>24.887577</td>
          <td>0.252691</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.754757</td>
          <td>0.206577</td>
          <td>25.663165</td>
          <td>0.066662</td>
          <td>25.370792</td>
          <td>0.045201</td>
          <td>24.848299</td>
          <td>0.046466</td>
          <td>24.377148</td>
          <td>0.058628</td>
          <td>23.719540</td>
          <td>0.073926</td>
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
          <td>27.094169</td>
          <td>0.613972</td>
          <td>26.464330</td>
          <td>0.143700</td>
          <td>26.086410</td>
          <td>0.092077</td>
          <td>26.016330</td>
          <td>0.140791</td>
          <td>25.997027</td>
          <td>0.255925</td>
          <td>25.429000</td>
          <td>0.340504</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.608424</td>
          <td>1.484769</td>
          <td>26.628723</td>
          <td>0.156855</td>
          <td>26.601273</td>
          <td>0.135591</td>
          <td>26.375470</td>
          <td>0.179641</td>
          <td>25.848776</td>
          <td>0.213507</td>
          <td>25.354903</td>
          <td>0.302773</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.706436</td>
          <td>0.454898</td>
          <td>26.958337</td>
          <td>0.213003</td>
          <td>27.083985</td>
          <td>0.210963</td>
          <td>26.793946</td>
          <td>0.262901</td>
          <td>25.967963</td>
          <td>0.243023</td>
          <td>25.100082</td>
          <td>0.253954</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.064783</td>
          <td>1.159620</td>
          <td>27.146996</td>
          <td>0.262910</td>
          <td>27.102464</td>
          <td>0.228005</td>
          <td>25.747386</td>
          <td>0.115725</td>
          <td>25.525856</td>
          <td>0.178728</td>
          <td>25.568863</td>
          <td>0.392556</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.820019</td>
          <td>0.188067</td>
          <td>26.166357</td>
          <td>0.094917</td>
          <td>25.649823</td>
          <td>0.098235</td>
          <td>25.139512</td>
          <td>0.119090</td>
          <td>24.873187</td>
          <td>0.208375</td>
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
