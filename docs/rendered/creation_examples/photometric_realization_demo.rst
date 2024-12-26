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

    <pzflow.flow.Flow at 0x7f85ea5751b0>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.790424</td>
          <td>0.177527</td>
          <td>25.910238</td>
          <td>0.072834</td>
          <td>25.375117</td>
          <td>0.074031</td>
          <td>25.116446</td>
          <td>0.112310</td>
          <td>25.115118</td>
          <td>0.245198</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.659782</td>
          <td>0.859468</td>
          <td>28.345661</td>
          <td>0.603163</td>
          <td>28.245678</td>
          <td>0.505899</td>
          <td>27.621949</td>
          <td>0.480314</td>
          <td>27.311843</td>
          <td>0.650891</td>
          <td>25.848251</td>
          <td>0.438412</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.469850</td>
          <td>0.759790</td>
          <td>25.883112</td>
          <td>0.080847</td>
          <td>24.819716</td>
          <td>0.027752</td>
          <td>23.860091</td>
          <td>0.019559</td>
          <td>23.144446</td>
          <td>0.019857</td>
          <td>22.818353</td>
          <td>0.033216</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.750021</td>
          <td>1.581967</td>
          <td>27.706563</td>
          <td>0.374980</td>
          <td>27.303610</td>
          <td>0.241738</td>
          <td>26.506922</td>
          <td>0.197420</td>
          <td>26.180162</td>
          <td>0.276258</td>
          <td>25.423482</td>
          <td>0.314922</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.448400</td>
          <td>0.362167</td>
          <td>25.947117</td>
          <td>0.085533</td>
          <td>25.380353</td>
          <td>0.045521</td>
          <td>24.805747</td>
          <td>0.044676</td>
          <td>24.413274</td>
          <td>0.060450</td>
          <td>23.681331</td>
          <td>0.071364</td>
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
          <td>26.851440</td>
          <td>0.492237</td>
          <td>26.229869</td>
          <td>0.109572</td>
          <td>26.066720</td>
          <td>0.083627</td>
          <td>26.082215</td>
          <td>0.137449</td>
          <td>26.083605</td>
          <td>0.255322</td>
          <td>25.492647</td>
          <td>0.332742</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.580781</td>
          <td>0.401316</td>
          <td>27.262354</td>
          <td>0.263031</td>
          <td>26.753472</td>
          <td>0.152092</td>
          <td>26.143456</td>
          <td>0.144896</td>
          <td>26.417801</td>
          <td>0.334303</td>
          <td>25.359217</td>
          <td>0.299109</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.321449</td>
          <td>0.687681</td>
          <td>27.412200</td>
          <td>0.297016</td>
          <td>26.847224</td>
          <td>0.164790</td>
          <td>26.434954</td>
          <td>0.185798</td>
          <td>26.549471</td>
          <td>0.370757</td>
          <td>25.285466</td>
          <td>0.281817</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.501153</td>
          <td>0.318952</td>
          <td>26.459848</td>
          <td>0.118013</td>
          <td>25.735684</td>
          <td>0.101686</td>
          <td>25.544955</td>
          <td>0.162587</td>
          <td>25.485554</td>
          <td>0.330876</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.548703</td>
          <td>0.391520</td>
          <td>26.617424</td>
          <td>0.153193</td>
          <td>26.171512</td>
          <td>0.091707</td>
          <td>25.664771</td>
          <td>0.095557</td>
          <td>25.020550</td>
          <td>0.103286</td>
          <td>24.842452</td>
          <td>0.195386</td>
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
          <td>28.796241</td>
          <td>1.722281</td>
          <td>26.902690</td>
          <td>0.223621</td>
          <td>25.985766</td>
          <td>0.091572</td>
          <td>25.330479</td>
          <td>0.084311</td>
          <td>25.007429</td>
          <td>0.119865</td>
          <td>24.879526</td>
          <td>0.236135</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.820676</td>
          <td>0.530844</td>
          <td>27.569993</td>
          <td>0.382694</td>
          <td>27.449720</td>
          <td>0.316058</td>
          <td>26.918146</td>
          <td>0.323664</td>
          <td>26.688272</td>
          <td>0.474255</td>
          <td>25.240035</td>
          <td>0.316610</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.694860</td>
          <td>0.490907</td>
          <td>26.034835</td>
          <td>0.108708</td>
          <td>24.803950</td>
          <td>0.032924</td>
          <td>23.836448</td>
          <td>0.023125</td>
          <td>23.129546</td>
          <td>0.023480</td>
          <td>22.835078</td>
          <td>0.040843</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.742581</td>
          <td>0.460188</td>
          <td>27.769014</td>
          <td>0.430218</td>
          <td>26.583623</td>
          <td>0.263307</td>
          <td>25.851751</td>
          <td>0.261262</td>
          <td>25.245895</td>
          <td>0.338455</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.254255</td>
          <td>0.345440</td>
          <td>25.875976</td>
          <td>0.092752</td>
          <td>25.491442</td>
          <td>0.059190</td>
          <td>24.780183</td>
          <td>0.051820</td>
          <td>24.255067</td>
          <td>0.061877</td>
          <td>23.642799</td>
          <td>0.081630</td>
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
          <td>26.926467</td>
          <td>0.580367</td>
          <td>26.196405</td>
          <td>0.124947</td>
          <td>26.147689</td>
          <td>0.107763</td>
          <td>26.155825</td>
          <td>0.176184</td>
          <td>25.887728</td>
          <td>0.257761</td>
          <td>25.620024</td>
          <td>0.433839</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.175483</td>
          <td>0.325438</td>
          <td>27.063799</td>
          <td>0.256313</td>
          <td>26.656298</td>
          <td>0.164466</td>
          <td>26.086480</td>
          <td>0.163312</td>
          <td>26.176554</td>
          <td>0.320536</td>
          <td>25.405034</td>
          <td>0.362047</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.955003</td>
          <td>0.516711</td>
          <td>27.465791</td>
          <td>0.323805</td>
          <td>26.139570</td>
          <td>0.172348</td>
          <td>25.863888</td>
          <td>0.250842</td>
          <td>25.742958</td>
          <td>0.472475</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.785748</td>
          <td>1.735682</td>
          <td>27.390723</td>
          <td>0.341063</td>
          <td>26.743738</td>
          <td>0.181880</td>
          <td>25.940964</td>
          <td>0.148198</td>
          <td>25.844217</td>
          <td>0.251153</td>
          <td>25.749448</td>
          <td>0.482518</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.388105</td>
          <td>0.145985</td>
          <td>26.021687</td>
          <td>0.095460</td>
          <td>25.670759</td>
          <td>0.114782</td>
          <td>24.901708</td>
          <td>0.110418</td>
          <td>25.270101</td>
          <td>0.327264</td>
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
          <td>27.217960</td>
          <td>0.640432</td>
          <td>26.636938</td>
          <td>0.155790</td>
          <td>25.968344</td>
          <td>0.076683</td>
          <td>25.401685</td>
          <td>0.075800</td>
          <td>25.095603</td>
          <td>0.110301</td>
          <td>25.074529</td>
          <td>0.237154</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.674544</td>
          <td>0.366001</td>
          <td>27.291780</td>
          <td>0.239603</td>
          <td>26.654530</td>
          <td>0.223567</td>
          <td>26.280223</td>
          <td>0.299791</td>
          <td>26.004801</td>
          <td>0.493346</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.129570</td>
          <td>0.296510</td>
          <td>25.924469</td>
          <td>0.090118</td>
          <td>24.836395</td>
          <td>0.030570</td>
          <td>23.851049</td>
          <td>0.021097</td>
          <td>23.194176</td>
          <td>0.022444</td>
          <td>22.870184</td>
          <td>0.037879</td>
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
          <td>27.620222</td>
          <td>0.382568</td>
          <td>26.373488</td>
          <td>0.220645</td>
          <td>26.083656</td>
          <td>0.314109</td>
          <td>25.909077</td>
          <td>0.557390</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.898090</td>
          <td>0.232713</td>
          <td>25.607356</td>
          <td>0.063451</td>
          <td>25.412449</td>
          <td>0.046904</td>
          <td>24.762154</td>
          <td>0.043046</td>
          <td>24.356479</td>
          <td>0.057562</td>
          <td>23.768955</td>
          <td>0.077226</td>
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
          <td>26.854521</td>
          <td>0.516971</td>
          <td>26.282197</td>
          <td>0.122786</td>
          <td>26.027979</td>
          <td>0.087465</td>
          <td>25.892173</td>
          <td>0.126466</td>
          <td>25.489425</td>
          <td>0.167366</td>
          <td>25.434640</td>
          <td>0.342024</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.783042</td>
          <td>0.472415</td>
          <td>26.664094</td>
          <td>0.161667</td>
          <td>26.949090</td>
          <td>0.182573</td>
          <td>26.490991</td>
          <td>0.198041</td>
          <td>26.045319</td>
          <td>0.251255</td>
          <td>25.278022</td>
          <td>0.284574</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.014672</td>
          <td>0.223235</td>
          <td>27.005950</td>
          <td>0.197603</td>
          <td>26.302856</td>
          <td>0.174537</td>
          <td>26.606643</td>
          <td>0.404571</td>
          <td>25.740590</td>
          <td>0.422077</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.963694</td>
          <td>2.695668</td>
          <td>27.444601</td>
          <td>0.334051</td>
          <td>26.690539</td>
          <td>0.161148</td>
          <td>26.076525</td>
          <td>0.153792</td>
          <td>25.779887</td>
          <td>0.221246</td>
          <td>25.121304</td>
          <td>0.275227</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.618973</td>
          <td>0.158548</td>
          <td>25.945912</td>
          <td>0.078171</td>
          <td>25.824581</td>
          <td>0.114447</td>
          <td>25.247911</td>
          <td>0.130830</td>
          <td>24.781196</td>
          <td>0.192885</td>
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
