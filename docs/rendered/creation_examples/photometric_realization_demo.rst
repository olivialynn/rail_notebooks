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

    <pzflow.flow.Flow at 0x7fc401e83d90>



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
          <td>27.153800</td>
          <td>0.612289</td>
          <td>26.704510</td>
          <td>0.165025</td>
          <td>26.020928</td>
          <td>0.080317</td>
          <td>25.255818</td>
          <td>0.066612</td>
          <td>24.997773</td>
          <td>0.101247</td>
          <td>25.051559</td>
          <td>0.232660</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.702723</td>
          <td>0.373861</td>
          <td>27.642463</td>
          <td>0.318273</td>
          <td>28.675517</td>
          <td>0.980860</td>
          <td>26.377943</td>
          <td>0.323889</td>
          <td>25.777114</td>
          <td>0.415304</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.618728</td>
          <td>0.413166</td>
          <td>25.920808</td>
          <td>0.083575</td>
          <td>24.747163</td>
          <td>0.026050</td>
          <td>23.850870</td>
          <td>0.019407</td>
          <td>23.114797</td>
          <td>0.019366</td>
          <td>22.911624</td>
          <td>0.036066</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.159614</td>
          <td>1.025587</td>
          <td>27.626165</td>
          <td>0.314158</td>
          <td>26.815450</td>
          <td>0.255103</td>
          <td>26.480769</td>
          <td>0.351337</td>
          <td>25.170712</td>
          <td>0.256655</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.218547</td>
          <td>0.301872</td>
          <td>25.862338</td>
          <td>0.079381</td>
          <td>25.506646</td>
          <td>0.050922</td>
          <td>24.792478</td>
          <td>0.044153</td>
          <td>24.452752</td>
          <td>0.062604</td>
          <td>23.775108</td>
          <td>0.077531</td>
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
          <td>27.170361</td>
          <td>0.619452</td>
          <td>26.470556</td>
          <td>0.135019</td>
          <td>26.210260</td>
          <td>0.094882</td>
          <td>26.034316</td>
          <td>0.131877</td>
          <td>25.713083</td>
          <td>0.187539</td>
          <td>25.143216</td>
          <td>0.250930</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.507631</td>
          <td>1.401099</td>
          <td>27.198452</td>
          <td>0.249616</td>
          <td>27.021928</td>
          <td>0.191109</td>
          <td>26.274156</td>
          <td>0.162072</td>
          <td>26.240911</td>
          <td>0.290191</td>
          <td>25.677858</td>
          <td>0.384744</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.048711</td>
          <td>0.486419</td>
          <td>27.374075</td>
          <td>0.256158</td>
          <td>25.976223</td>
          <td>0.125405</td>
          <td>25.887576</td>
          <td>0.217114</td>
          <td>25.505302</td>
          <td>0.336094</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.990935</td>
          <td>1.052822</td>
          <td>27.601570</td>
          <td>0.345380</td>
          <td>26.597388</td>
          <td>0.132964</td>
          <td>25.812206</td>
          <td>0.108723</td>
          <td>25.669192</td>
          <td>0.180705</td>
          <td>25.342072</td>
          <td>0.295009</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.412536</td>
          <td>0.731340</td>
          <td>26.642506</td>
          <td>0.156517</td>
          <td>26.112423</td>
          <td>0.087062</td>
          <td>25.555547</td>
          <td>0.086808</td>
          <td>25.073859</td>
          <td>0.108213</td>
          <td>24.854157</td>
          <td>0.197319</td>
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
          <td>26.580576</td>
          <td>0.444225</td>
          <td>27.121573</td>
          <td>0.267764</td>
          <td>25.887239</td>
          <td>0.083969</td>
          <td>25.338424</td>
          <td>0.084903</td>
          <td>24.959259</td>
          <td>0.114947</td>
          <td>24.474291</td>
          <td>0.168030</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.865763</td>
          <td>0.548482</td>
          <td>28.643745</td>
          <td>0.823714</td>
          <td>27.617275</td>
          <td>0.360841</td>
          <td>26.783852</td>
          <td>0.290631</td>
          <td>26.349764</td>
          <td>0.366180</td>
          <td>25.192840</td>
          <td>0.304876</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>32.116020</td>
          <td>4.825420</td>
          <td>26.011167</td>
          <td>0.106487</td>
          <td>24.841564</td>
          <td>0.034033</td>
          <td>23.920077</td>
          <td>0.024856</td>
          <td>23.158552</td>
          <td>0.024074</td>
          <td>22.822109</td>
          <td>0.040377</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.989499</td>
          <td>1.927330</td>
          <td>27.669311</td>
          <td>0.435450</td>
          <td>27.359892</td>
          <td>0.312621</td>
          <td>26.280554</td>
          <td>0.204883</td>
          <td>26.279409</td>
          <td>0.367821</td>
          <td>25.855359</td>
          <td>0.537771</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.324293</td>
          <td>0.364938</td>
          <td>25.666152</td>
          <td>0.077126</td>
          <td>25.353671</td>
          <td>0.052380</td>
          <td>24.812981</td>
          <td>0.053350</td>
          <td>24.314731</td>
          <td>0.065235</td>
          <td>23.600150</td>
          <td>0.078617</td>
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
          <td>26.933301</td>
          <td>0.583199</td>
          <td>26.277399</td>
          <td>0.134010</td>
          <td>26.073372</td>
          <td>0.100983</td>
          <td>25.971256</td>
          <td>0.150510</td>
          <td>26.488200</td>
          <td>0.415065</td>
          <td>26.133978</td>
          <td>0.631162</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.653580</td>
          <td>0.932919</td>
          <td>27.571400</td>
          <td>0.384322</td>
          <td>26.836852</td>
          <td>0.191676</td>
          <td>26.257951</td>
          <td>0.188894</td>
          <td>27.030115</td>
          <td>0.609748</td>
          <td>25.404775</td>
          <td>0.361974</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.500631</td>
          <td>0.421623</td>
          <td>27.569378</td>
          <td>0.386360</td>
          <td>26.942494</td>
          <td>0.211179</td>
          <td>26.718983</td>
          <td>0.279092</td>
          <td>26.144547</td>
          <td>0.314905</td>
          <td>24.975873</td>
          <td>0.258775</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.732511</td>
          <td>0.993247</td>
          <td>27.330056</td>
          <td>0.325058</td>
          <td>26.445344</td>
          <td>0.140958</td>
          <td>25.967627</td>
          <td>0.151628</td>
          <td>25.826207</td>
          <td>0.247463</td>
          <td>25.083109</td>
          <td>0.287407</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.771298</td>
          <td>1.005361</td>
          <td>26.719204</td>
          <td>0.193471</td>
          <td>26.202734</td>
          <td>0.111840</td>
          <td>25.526687</td>
          <td>0.101214</td>
          <td>25.397488</td>
          <td>0.169327</td>
          <td>25.472751</td>
          <td>0.383702</td>
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
          <td>27.046717</td>
          <td>0.567504</td>
          <td>26.752469</td>
          <td>0.171918</td>
          <td>26.087872</td>
          <td>0.085211</td>
          <td>25.262608</td>
          <td>0.067023</td>
          <td>25.078074</td>
          <td>0.108626</td>
          <td>24.737870</td>
          <td>0.178890</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.248416</td>
          <td>0.563170</td>
          <td>28.113194</td>
          <td>0.458812</td>
          <td>27.475790</td>
          <td>0.430683</td>
          <td>27.051587</td>
          <td>0.541638</td>
          <td>25.680968</td>
          <td>0.386007</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.358121</td>
          <td>0.355525</td>
          <td>25.989286</td>
          <td>0.095390</td>
          <td>24.830015</td>
          <td>0.030399</td>
          <td>23.864263</td>
          <td>0.021337</td>
          <td>23.125881</td>
          <td>0.021171</td>
          <td>22.777416</td>
          <td>0.034899</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.242775</td>
          <td>0.740901</td>
          <td>28.228280</td>
          <td>0.651707</td>
          <td>27.433545</td>
          <td>0.330438</td>
          <td>26.580852</td>
          <td>0.261814</td>
          <td>25.781483</td>
          <td>0.245806</td>
          <td>25.205724</td>
          <td>0.326768</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.892291</td>
          <td>0.231600</td>
          <td>25.800665</td>
          <td>0.075272</td>
          <td>25.450492</td>
          <td>0.048515</td>
          <td>24.883789</td>
          <td>0.047954</td>
          <td>24.442040</td>
          <td>0.062101</td>
          <td>23.721516</td>
          <td>0.074055</td>
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
          <td>26.302834</td>
          <td>0.339579</td>
          <td>26.128925</td>
          <td>0.107460</td>
          <td>26.061868</td>
          <td>0.090112</td>
          <td>26.290220</td>
          <td>0.177944</td>
          <td>26.139269</td>
          <td>0.287346</td>
          <td>27.171088</td>
          <td>1.130152</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.061102</td>
          <td>1.838736</td>
          <td>27.215155</td>
          <td>0.256466</td>
          <td>27.342056</td>
          <td>0.253383</td>
          <td>26.261813</td>
          <td>0.163089</td>
          <td>26.586936</td>
          <td>0.387299</td>
          <td>25.769594</td>
          <td>0.419111</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.042523</td>
          <td>0.228456</td>
          <td>26.659288</td>
          <td>0.147147</td>
          <td>26.618545</td>
          <td>0.227535</td>
          <td>26.237950</td>
          <td>0.302753</td>
          <td>25.681784</td>
          <td>0.403492</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.593998</td>
          <td>0.772848</td>
          <td>26.714848</td>
          <td>0.164526</td>
          <td>25.717524</td>
          <td>0.112753</td>
          <td>25.274884</td>
          <td>0.144244</td>
          <td>25.459899</td>
          <td>0.360656</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.441352</td>
          <td>0.369014</td>
          <td>26.613408</td>
          <td>0.157796</td>
          <td>26.164848</td>
          <td>0.094792</td>
          <td>25.627740</td>
          <td>0.096351</td>
          <td>25.465515</td>
          <td>0.157770</td>
          <td>24.803928</td>
          <td>0.196612</td>
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
