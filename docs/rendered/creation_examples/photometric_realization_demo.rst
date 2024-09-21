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

    <pzflow.flow.Flow at 0x7f994e899a50>



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
          <td>26.975750</td>
          <td>0.539160</td>
          <td>26.947890</td>
          <td>0.202732</td>
          <td>25.873720</td>
          <td>0.070518</td>
          <td>25.383486</td>
          <td>0.074580</td>
          <td>25.002667</td>
          <td>0.101682</td>
          <td>25.394677</td>
          <td>0.307747</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.156703</td>
          <td>1.906563</td>
          <td>27.828399</td>
          <td>0.411965</td>
          <td>27.310754</td>
          <td>0.243166</td>
          <td>26.465657</td>
          <td>0.190677</td>
          <td>26.693282</td>
          <td>0.414327</td>
          <td>25.578289</td>
          <td>0.355996</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.606459</td>
          <td>0.409303</td>
          <td>26.063302</td>
          <td>0.094721</td>
          <td>24.744317</td>
          <td>0.025986</td>
          <td>23.861221</td>
          <td>0.019577</td>
          <td>23.139181</td>
          <td>0.019769</td>
          <td>22.875506</td>
          <td>0.034934</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.514373</td>
          <td>0.782412</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.376634</td>
          <td>0.256696</td>
          <td>26.339114</td>
          <td>0.171297</td>
          <td>26.053263</td>
          <td>0.249040</td>
          <td>25.337151</td>
          <td>0.293841</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.276728</td>
          <td>0.316252</td>
          <td>25.757928</td>
          <td>0.072395</td>
          <td>25.384175</td>
          <td>0.045675</td>
          <td>24.735082</td>
          <td>0.041961</td>
          <td>24.487692</td>
          <td>0.064573</td>
          <td>23.641198</td>
          <td>0.068873</td>
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
          <td>26.468797</td>
          <td>0.367981</td>
          <td>26.103794</td>
          <td>0.098142</td>
          <td>26.199835</td>
          <td>0.094018</td>
          <td>26.052997</td>
          <td>0.134024</td>
          <td>25.720014</td>
          <td>0.188639</td>
          <td>25.982806</td>
          <td>0.484966</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.417969</td>
          <td>0.734004</td>
          <td>27.048607</td>
          <td>0.220526</td>
          <td>26.679610</td>
          <td>0.142739</td>
          <td>26.906067</td>
          <td>0.274694</td>
          <td>26.547810</td>
          <td>0.370277</td>
          <td>25.782818</td>
          <td>0.417119</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.215787</td>
          <td>0.253193</td>
          <td>26.986823</td>
          <td>0.185529</td>
          <td>26.674094</td>
          <td>0.227017</td>
          <td>25.890495</td>
          <td>0.217642</td>
          <td>25.124430</td>
          <td>0.247084</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.488371</td>
          <td>0.769145</td>
          <td>27.059889</td>
          <td>0.222605</td>
          <td>26.447678</td>
          <td>0.116770</td>
          <td>25.902526</td>
          <td>0.117628</td>
          <td>25.430157</td>
          <td>0.147362</td>
          <td>25.082916</td>
          <td>0.238772</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.127726</td>
          <td>0.280563</td>
          <td>26.645512</td>
          <td>0.156920</td>
          <td>26.164156</td>
          <td>0.091116</td>
          <td>25.707741</td>
          <td>0.099227</td>
          <td>25.014695</td>
          <td>0.102758</td>
          <td>25.153610</td>
          <td>0.253081</td>
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
          <td>28.050285</td>
          <td>1.176399</td>
          <td>26.882543</td>
          <td>0.219907</td>
          <td>25.973106</td>
          <td>0.090559</td>
          <td>25.295011</td>
          <td>0.081716</td>
          <td>25.124115</td>
          <td>0.132621</td>
          <td>25.046429</td>
          <td>0.270794</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.559486</td>
          <td>0.437258</td>
          <td>28.482478</td>
          <td>0.740984</td>
          <td>27.015535</td>
          <td>0.221777</td>
          <td>28.937880</td>
          <td>1.273056</td>
          <td>26.351687</td>
          <td>0.366731</td>
          <td>36.203075</td>
          <td>9.786465</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.531360</td>
          <td>0.434324</td>
          <td>26.055063</td>
          <td>0.110641</td>
          <td>24.831822</td>
          <td>0.033742</td>
          <td>23.862055</td>
          <td>0.023640</td>
          <td>23.126883</td>
          <td>0.023426</td>
          <td>22.793199</td>
          <td>0.039357</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.268920</td>
          <td>2.016161</td>
          <td>26.922100</td>
          <td>0.218612</td>
          <td>26.531394</td>
          <td>0.252283</td>
          <td>27.185760</td>
          <td>0.713211</td>
          <td>25.466918</td>
          <td>0.402145</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.387223</td>
          <td>0.383242</td>
          <td>25.719823</td>
          <td>0.080860</td>
          <td>25.471911</td>
          <td>0.058173</td>
          <td>24.793335</td>
          <td>0.052428</td>
          <td>24.445965</td>
          <td>0.073267</td>
          <td>23.796949</td>
          <td>0.093486</td>
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
          <td>26.492007</td>
          <td>0.421150</td>
          <td>26.395342</td>
          <td>0.148325</td>
          <td>26.077648</td>
          <td>0.101362</td>
          <td>25.898761</td>
          <td>0.141417</td>
          <td>25.881446</td>
          <td>0.256438</td>
          <td>32.230406</td>
          <td>5.842952</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.447005</td>
          <td>0.818729</td>
          <td>26.835503</td>
          <td>0.212208</td>
          <td>26.989909</td>
          <td>0.217912</td>
          <td>26.907302</td>
          <td>0.322077</td>
          <td>26.206619</td>
          <td>0.328294</td>
          <td>25.935871</td>
          <td>0.540529</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.136786</td>
          <td>0.274013</td>
          <td>26.911535</td>
          <td>0.205779</td>
          <td>26.330345</td>
          <td>0.202478</td>
          <td>25.830697</td>
          <td>0.244086</td>
          <td>26.938997</td>
          <td>1.054011</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.032847</td>
          <td>1.937119</td>
          <td>27.426327</td>
          <td>0.350765</td>
          <td>26.833673</td>
          <td>0.196220</td>
          <td>26.038345</td>
          <td>0.161088</td>
          <td>25.482940</td>
          <td>0.185845</td>
          <td>25.563634</td>
          <td>0.419477</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>29.464225</td>
          <td>2.290567</td>
          <td>26.411301</td>
          <td>0.148920</td>
          <td>26.020731</td>
          <td>0.095380</td>
          <td>25.567747</td>
          <td>0.104916</td>
          <td>25.281231</td>
          <td>0.153324</td>
          <td>24.947786</td>
          <td>0.252220</td>
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
          <td>27.390153</td>
          <td>0.720485</td>
          <td>26.688201</td>
          <td>0.162764</td>
          <td>26.070445</td>
          <td>0.083913</td>
          <td>25.327746</td>
          <td>0.071002</td>
          <td>24.899036</td>
          <td>0.092860</td>
          <td>24.886095</td>
          <td>0.202710</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.773947</td>
          <td>1.600937</td>
          <td>27.437586</td>
          <td>0.303365</td>
          <td>27.441729</td>
          <td>0.270953</td>
          <td>26.975907</td>
          <td>0.290952</td>
          <td>25.755315</td>
          <td>0.194514</td>
          <td>25.777189</td>
          <td>0.415683</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.706354</td>
          <td>0.464316</td>
          <td>26.044839</td>
          <td>0.100145</td>
          <td>24.780373</td>
          <td>0.029104</td>
          <td>23.857144</td>
          <td>0.021207</td>
          <td>23.143508</td>
          <td>0.021492</td>
          <td>22.839035</td>
          <td>0.036850</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.455427</td>
          <td>0.368329</td>
          <td>27.117901</td>
          <td>0.256145</td>
          <td>26.245678</td>
          <td>0.198275</td>
          <td>25.813550</td>
          <td>0.252371</td>
          <td>25.769202</td>
          <td>0.503396</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.174234</td>
          <td>0.291568</td>
          <td>25.771743</td>
          <td>0.073375</td>
          <td>25.412683</td>
          <td>0.046913</td>
          <td>24.765209</td>
          <td>0.043163</td>
          <td>24.375753</td>
          <td>0.058555</td>
          <td>23.554831</td>
          <td>0.063896</td>
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
          <td>25.940025</td>
          <td>0.253599</td>
          <td>26.229291</td>
          <td>0.117274</td>
          <td>26.170134</td>
          <td>0.099096</td>
          <td>26.128650</td>
          <td>0.155053</td>
          <td>25.626728</td>
          <td>0.188035</td>
          <td>26.451910</td>
          <td>0.722086</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.465842</td>
          <td>2.179463</td>
          <td>27.060136</td>
          <td>0.225686</td>
          <td>27.157268</td>
          <td>0.217466</td>
          <td>26.364152</td>
          <td>0.177926</td>
          <td>25.728143</td>
          <td>0.192960</td>
          <td>25.438620</td>
          <td>0.323731</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.390946</td>
          <td>0.739989</td>
          <td>27.779810</td>
          <td>0.411964</td>
          <td>26.689095</td>
          <td>0.150962</td>
          <td>26.858480</td>
          <td>0.277093</td>
          <td>26.279028</td>
          <td>0.312884</td>
          <td>25.811603</td>
          <td>0.445449</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.675832</td>
          <td>2.435010</td>
          <td>27.640124</td>
          <td>0.389298</td>
          <td>26.626091</td>
          <td>0.152501</td>
          <td>25.766387</td>
          <td>0.117654</td>
          <td>25.549834</td>
          <td>0.182395</td>
          <td>24.914645</td>
          <td>0.232297</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.724947</td>
          <td>0.913085</td>
          <td>26.464063</td>
          <td>0.138813</td>
          <td>26.129987</td>
          <td>0.091933</td>
          <td>25.648452</td>
          <td>0.098118</td>
          <td>25.246471</td>
          <td>0.130667</td>
          <td>24.439899</td>
          <td>0.144225</td>
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
