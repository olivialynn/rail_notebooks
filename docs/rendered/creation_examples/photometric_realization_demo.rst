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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f705f0e0580>



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
          <td>27.243560</td>
          <td>0.651858</td>
          <td>26.640920</td>
          <td>0.156305</td>
          <td>25.981589</td>
          <td>0.077575</td>
          <td>25.203145</td>
          <td>0.063574</td>
          <td>24.993378</td>
          <td>0.100858</td>
          <td>25.148230</td>
          <td>0.251966</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.017729</td>
          <td>0.555767</td>
          <td>28.583310</td>
          <td>0.710828</td>
          <td>26.833211</td>
          <td>0.162831</td>
          <td>27.178170</td>
          <td>0.341651</td>
          <td>26.874599</td>
          <td>0.475160</td>
          <td>26.412621</td>
          <td>0.659963</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.779024</td>
          <td>0.466428</td>
          <td>26.095917</td>
          <td>0.097467</td>
          <td>24.790754</td>
          <td>0.027059</td>
          <td>23.841865</td>
          <td>0.019260</td>
          <td>23.139804</td>
          <td>0.019780</td>
          <td>22.871959</td>
          <td>0.034825</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.415424</td>
          <td>0.732756</td>
          <td>28.467575</td>
          <td>0.656785</td>
          <td>27.259319</td>
          <td>0.233050</td>
          <td>26.622534</td>
          <td>0.217486</td>
          <td>25.783460</td>
          <td>0.198993</td>
          <td>25.163281</td>
          <td>0.255097</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.849332</td>
          <td>0.491470</td>
          <td>25.731434</td>
          <td>0.070721</td>
          <td>25.370766</td>
          <td>0.045135</td>
          <td>24.824237</td>
          <td>0.045415</td>
          <td>24.314620</td>
          <td>0.055383</td>
          <td>23.699325</td>
          <td>0.072509</td>
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
          <td>26.151248</td>
          <td>0.285952</td>
          <td>26.508178</td>
          <td>0.139471</td>
          <td>26.179036</td>
          <td>0.092315</td>
          <td>25.897188</td>
          <td>0.117083</td>
          <td>25.790943</td>
          <td>0.200248</td>
          <td>25.392312</td>
          <td>0.307164</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.439138</td>
          <td>0.744451</td>
          <td>27.590659</td>
          <td>0.342420</td>
          <td>27.166941</td>
          <td>0.215828</td>
          <td>26.550742</td>
          <td>0.204818</td>
          <td>25.563889</td>
          <td>0.165235</td>
          <td>25.729335</td>
          <td>0.400352</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.079889</td>
          <td>0.226334</td>
          <td>26.766083</td>
          <td>0.153745</td>
          <td>26.877554</td>
          <td>0.268391</td>
          <td>26.223100</td>
          <td>0.286044</td>
          <td>25.628048</td>
          <td>0.370128</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.767808</td>
          <td>0.462529</td>
          <td>27.223336</td>
          <td>0.254766</td>
          <td>26.605779</td>
          <td>0.133932</td>
          <td>25.976096</td>
          <td>0.125392</td>
          <td>25.274682</td>
          <td>0.128865</td>
          <td>25.150240</td>
          <td>0.252382</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.676724</td>
          <td>0.431831</td>
          <td>26.477714</td>
          <td>0.135856</td>
          <td>26.244278</td>
          <td>0.097757</td>
          <td>25.785042</td>
          <td>0.106173</td>
          <td>25.118791</td>
          <td>0.112540</td>
          <td>24.714162</td>
          <td>0.175305</td>
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
          <td>27.147568</td>
          <td>0.668918</td>
          <td>26.886417</td>
          <td>0.220617</td>
          <td>26.006734</td>
          <td>0.093275</td>
          <td>25.269150</td>
          <td>0.079874</td>
          <td>25.068314</td>
          <td>0.126368</td>
          <td>24.821663</td>
          <td>0.225080</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.477796</td>
          <td>0.833295</td>
          <td>27.877125</td>
          <td>0.483238</td>
          <td>27.170285</td>
          <td>0.252037</td>
          <td>27.760929</td>
          <td>0.610367</td>
          <td>26.372373</td>
          <td>0.372697</td>
          <td>27.743266</td>
          <td>1.607012</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.301769</td>
          <td>0.363975</td>
          <td>25.927765</td>
          <td>0.099003</td>
          <td>24.748647</td>
          <td>0.031361</td>
          <td>23.890676</td>
          <td>0.024232</td>
          <td>23.159682</td>
          <td>0.024098</td>
          <td>22.873284</td>
          <td>0.042248</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.919729</td>
          <td>1.020611</td>
          <td>27.831610</td>
          <td>0.451085</td>
          <td>26.889858</td>
          <td>0.336861</td>
          <td>26.051192</td>
          <td>0.307050</td>
          <td>25.022551</td>
          <td>0.283047</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.053705</td>
          <td>0.626911</td>
          <td>25.787425</td>
          <td>0.085815</td>
          <td>25.441967</td>
          <td>0.056649</td>
          <td>24.799088</td>
          <td>0.052697</td>
          <td>24.470436</td>
          <td>0.074868</td>
          <td>23.596354</td>
          <td>0.078354</td>
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
          <td>28.453880</td>
          <td>1.472876</td>
          <td>26.387328</td>
          <td>0.147309</td>
          <td>26.085839</td>
          <td>0.102091</td>
          <td>26.358957</td>
          <td>0.209076</td>
          <td>26.016258</td>
          <td>0.286187</td>
          <td>25.351392</td>
          <td>0.352525</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.089913</td>
          <td>0.644414</td>
          <td>26.963988</td>
          <td>0.236107</td>
          <td>26.664745</td>
          <td>0.165655</td>
          <td>26.398916</td>
          <td>0.212627</td>
          <td>26.106426</td>
          <td>0.303050</td>
          <td>27.607025</td>
          <td>1.506801</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.231909</td>
          <td>0.295931</td>
          <td>27.062848</td>
          <td>0.233413</td>
          <td>26.241726</td>
          <td>0.187927</td>
          <td>26.627443</td>
          <td>0.458022</td>
          <td>25.843887</td>
          <td>0.509149</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.655849</td>
          <td>0.947972</td>
          <td>27.741892</td>
          <td>0.447332</td>
          <td>26.243891</td>
          <td>0.118404</td>
          <td>25.831324</td>
          <td>0.134845</td>
          <td>25.669749</td>
          <td>0.217392</td>
          <td>25.610053</td>
          <td>0.434561</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.268272</td>
          <td>0.351587</td>
          <td>26.772245</td>
          <td>0.202286</td>
          <td>26.071579</td>
          <td>0.099728</td>
          <td>25.584087</td>
          <td>0.106425</td>
          <td>25.376667</td>
          <td>0.166351</td>
          <td>24.643859</td>
          <td>0.195902</td>
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
          <td>26.928828</td>
          <td>0.521095</td>
          <td>26.648056</td>
          <td>0.157279</td>
          <td>25.937506</td>
          <td>0.074621</td>
          <td>25.340114</td>
          <td>0.071784</td>
          <td>24.987505</td>
          <td>0.100354</td>
          <td>24.784058</td>
          <td>0.186021</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.869735</td>
          <td>0.425480</td>
          <td>27.465031</td>
          <td>0.276138</td>
          <td>27.807091</td>
          <td>0.550606</td>
          <td>26.805647</td>
          <td>0.451595</td>
          <td>25.425537</td>
          <td>0.315722</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.433374</td>
          <td>0.377029</td>
          <td>25.861817</td>
          <td>0.085291</td>
          <td>24.822208</td>
          <td>0.030192</td>
          <td>23.889631</td>
          <td>0.021804</td>
          <td>23.120468</td>
          <td>0.021074</td>
          <td>22.838840</td>
          <td>0.036844</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.768360</td>
          <td>0.467858</td>
          <td>27.585081</td>
          <td>0.372258</td>
          <td>26.420266</td>
          <td>0.229387</td>
          <td>25.863297</td>
          <td>0.262865</td>
          <td>25.282926</td>
          <td>0.347348</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.477594</td>
          <td>0.370836</td>
          <td>25.747994</td>
          <td>0.071852</td>
          <td>25.415071</td>
          <td>0.047013</td>
          <td>24.786621</td>
          <td>0.043991</td>
          <td>24.377630</td>
          <td>0.058653</td>
          <td>23.595059</td>
          <td>0.066214</td>
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
          <td>26.673105</td>
          <td>0.451841</td>
          <td>26.132304</td>
          <td>0.107777</td>
          <td>26.195813</td>
          <td>0.101350</td>
          <td>26.183218</td>
          <td>0.162459</td>
          <td>26.214981</td>
          <td>0.305411</td>
          <td>26.202411</td>
          <td>0.608049</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.122974</td>
          <td>1.145114</td>
          <td>27.336201</td>
          <td>0.283039</td>
          <td>26.748376</td>
          <td>0.153883</td>
          <td>26.255242</td>
          <td>0.162177</td>
          <td>25.796994</td>
          <td>0.204455</td>
          <td>25.486850</td>
          <td>0.336363</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.773904</td>
          <td>1.630637</td>
          <td>27.283900</td>
          <td>0.278489</td>
          <td>26.760314</td>
          <td>0.160453</td>
          <td>26.540456</td>
          <td>0.213213</td>
          <td>25.972905</td>
          <td>0.244014</td>
          <td>25.099465</td>
          <td>0.253825</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.013236</td>
          <td>0.592264</td>
          <td>27.453360</td>
          <td>0.336375</td>
          <td>26.467631</td>
          <td>0.133054</td>
          <td>26.017549</td>
          <td>0.146201</td>
          <td>25.290553</td>
          <td>0.146200</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.532926</td>
          <td>1.442181</td>
          <td>26.442271</td>
          <td>0.136230</td>
          <td>25.954474</td>
          <td>0.078765</td>
          <td>25.684637</td>
          <td>0.101278</td>
          <td>25.611403</td>
          <td>0.178641</td>
          <td>25.096675</td>
          <td>0.250808</td>
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
