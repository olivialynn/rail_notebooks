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

    <pzflow.flow.Flow at 0x7f0d503733d0>



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
          <td>28.545094</td>
          <td>1.428375</td>
          <td>26.743832</td>
          <td>0.170642</td>
          <td>25.895211</td>
          <td>0.071872</td>
          <td>25.345981</td>
          <td>0.072147</td>
          <td>24.982949</td>
          <td>0.099941</td>
          <td>25.101106</td>
          <td>0.242383</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.109444</td>
          <td>0.593409</td>
          <td>28.571405</td>
          <td>0.705127</td>
          <td>27.776943</td>
          <td>0.354027</td>
          <td>26.825052</td>
          <td>0.257118</td>
          <td>28.285975</td>
          <td>1.197812</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.535939</td>
          <td>0.387677</td>
          <td>25.879892</td>
          <td>0.080618</td>
          <td>24.786875</td>
          <td>0.026967</td>
          <td>23.872361</td>
          <td>0.019763</td>
          <td>23.101420</td>
          <td>0.019149</td>
          <td>22.817421</td>
          <td>0.033189</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.294600</td>
          <td>1.251009</td>
          <td>28.412259</td>
          <td>0.632037</td>
          <td>27.941058</td>
          <td>0.402208</td>
          <td>26.572588</td>
          <td>0.208600</td>
          <td>26.123632</td>
          <td>0.263824</td>
          <td>24.952292</td>
          <td>0.214229</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.252556</td>
          <td>0.310207</td>
          <td>25.908072</td>
          <td>0.082644</td>
          <td>25.442233</td>
          <td>0.048091</td>
          <td>24.773554</td>
          <td>0.043418</td>
          <td>24.314605</td>
          <td>0.055383</td>
          <td>23.556728</td>
          <td>0.063908</td>
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
          <td>26.264325</td>
          <td>0.313138</td>
          <td>26.384450</td>
          <td>0.125331</td>
          <td>26.028221</td>
          <td>0.080835</td>
          <td>26.037967</td>
          <td>0.132294</td>
          <td>26.002529</td>
          <td>0.238843</td>
          <td>25.573878</td>
          <td>0.354765</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>30.250461</td>
          <td>2.874130</td>
          <td>27.084134</td>
          <td>0.227132</td>
          <td>27.472770</td>
          <td>0.277637</td>
          <td>26.248935</td>
          <td>0.158616</td>
          <td>28.828665</td>
          <td>1.588617</td>
          <td>25.319033</td>
          <td>0.289576</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>36.455094</td>
          <td>8.999195</td>
          <td>28.029106</td>
          <td>0.479384</td>
          <td>26.861352</td>
          <td>0.166786</td>
          <td>26.321029</td>
          <td>0.168681</td>
          <td>25.956472</td>
          <td>0.229912</td>
          <td>24.887145</td>
          <td>0.202863</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.564535</td>
          <td>0.335423</td>
          <td>26.675722</td>
          <td>0.142262</td>
          <td>25.928614</td>
          <td>0.120328</td>
          <td>25.478840</td>
          <td>0.153649</td>
          <td>25.471781</td>
          <td>0.327277</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.101997</td>
          <td>0.274772</td>
          <td>26.450095</td>
          <td>0.132655</td>
          <td>26.165770</td>
          <td>0.091245</td>
          <td>25.792040</td>
          <td>0.106824</td>
          <td>25.285161</td>
          <td>0.130040</td>
          <td>25.469099</td>
          <td>0.326580</td>
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
          <td>27.066530</td>
          <td>0.632427</td>
          <td>27.273734</td>
          <td>0.302837</td>
          <td>26.055256</td>
          <td>0.097331</td>
          <td>25.458208</td>
          <td>0.094333</td>
          <td>25.072412</td>
          <td>0.126818</td>
          <td>24.964711</td>
          <td>0.253297</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.733635</td>
          <td>0.977730</td>
          <td>27.923760</td>
          <td>0.500211</td>
          <td>27.786712</td>
          <td>0.411457</td>
          <td>26.687688</td>
          <td>0.268821</td>
          <td>26.821268</td>
          <td>0.523183</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.371407</td>
          <td>0.384229</td>
          <td>25.949627</td>
          <td>0.100915</td>
          <td>24.756058</td>
          <td>0.031566</td>
          <td>23.911735</td>
          <td>0.024677</td>
          <td>23.114415</td>
          <td>0.023176</td>
          <td>22.861641</td>
          <td>0.041815</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.821348</td>
          <td>2.498757</td>
          <td>26.905318</td>
          <td>0.215575</td>
          <td>26.756639</td>
          <td>0.302922</td>
          <td>25.894399</td>
          <td>0.270514</td>
          <td>25.791786</td>
          <td>0.513391</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.059110</td>
          <td>0.295748</td>
          <td>25.795892</td>
          <td>0.086456</td>
          <td>25.377930</td>
          <td>0.053520</td>
          <td>24.735142</td>
          <td>0.049790</td>
          <td>24.246572</td>
          <td>0.061412</td>
          <td>23.617846</td>
          <td>0.079854</td>
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
          <td>27.515164</td>
          <td>0.863318</td>
          <td>26.249538</td>
          <td>0.130825</td>
          <td>26.165154</td>
          <td>0.109418</td>
          <td>26.230224</td>
          <td>0.187635</td>
          <td>25.867520</td>
          <td>0.253527</td>
          <td>25.521363</td>
          <td>0.402339</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.980699</td>
          <td>1.133216</td>
          <td>26.661783</td>
          <td>0.183389</td>
          <td>26.935724</td>
          <td>0.208272</td>
          <td>26.090278</td>
          <td>0.163842</td>
          <td>25.955046</td>
          <td>0.268113</td>
          <td>24.932270</td>
          <td>0.247626</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.998671</td>
          <td>0.244739</td>
          <td>26.940976</td>
          <td>0.210911</td>
          <td>26.726329</td>
          <td>0.280759</td>
          <td>26.603199</td>
          <td>0.449743</td>
          <td>25.137864</td>
          <td>0.295163</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.506447</td>
          <td>0.373457</td>
          <td>26.654425</td>
          <td>0.168601</td>
          <td>26.006733</td>
          <td>0.156793</td>
          <td>25.609946</td>
          <td>0.206798</td>
          <td>25.476588</td>
          <td>0.392351</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.750167</td>
          <td>0.507317</td>
          <td>26.496005</td>
          <td>0.160115</td>
          <td>26.061119</td>
          <td>0.098818</td>
          <td>25.658790</td>
          <td>0.113592</td>
          <td>24.999682</td>
          <td>0.120248</td>
          <td>24.856100</td>
          <td>0.233863</td>
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

.. parsed-literal::

    




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
          <td>27.287620</td>
          <td>0.671999</td>
          <td>26.631622</td>
          <td>0.155083</td>
          <td>26.103062</td>
          <td>0.086359</td>
          <td>25.309902</td>
          <td>0.069890</td>
          <td>24.771111</td>
          <td>0.082973</td>
          <td>25.138522</td>
          <td>0.249996</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.319661</td>
          <td>0.592554</td>
          <td>28.364862</td>
          <td>0.552246</td>
          <td>27.329497</td>
          <td>0.384938</td>
          <td>27.177340</td>
          <td>0.592764</td>
          <td>26.093618</td>
          <td>0.526614</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.894990</td>
          <td>0.245012</td>
          <td>25.804074</td>
          <td>0.081065</td>
          <td>24.775035</td>
          <td>0.028969</td>
          <td>23.877570</td>
          <td>0.021581</td>
          <td>23.131912</td>
          <td>0.021280</td>
          <td>22.847485</td>
          <td>0.037127</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.267988</td>
          <td>0.753435</td>
          <td>27.368167</td>
          <td>0.343962</td>
          <td>27.025456</td>
          <td>0.237383</td>
          <td>26.872281</td>
          <td>0.331103</td>
          <td>27.444057</td>
          <td>0.843036</td>
          <td>25.630373</td>
          <td>0.453961</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.854923</td>
          <td>0.493915</td>
          <td>25.812604</td>
          <td>0.076069</td>
          <td>25.405588</td>
          <td>0.046619</td>
          <td>24.816752</td>
          <td>0.045183</td>
          <td>24.333796</td>
          <td>0.056415</td>
          <td>23.665217</td>
          <td>0.070458</td>
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
          <td>27.322206</td>
          <td>0.718293</td>
          <td>26.352045</td>
          <td>0.130441</td>
          <td>26.049738</td>
          <td>0.089156</td>
          <td>26.106041</td>
          <td>0.152078</td>
          <td>25.905251</td>
          <td>0.237306</td>
          <td>25.266982</td>
          <td>0.299243</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.050857</td>
          <td>0.574489</td>
          <td>26.785724</td>
          <td>0.179281</td>
          <td>26.803181</td>
          <td>0.161270</td>
          <td>26.405885</td>
          <td>0.184327</td>
          <td>26.905179</td>
          <td>0.492882</td>
          <td>25.531426</td>
          <td>0.348410</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.549872</td>
          <td>0.403882</td>
          <td>27.005409</td>
          <td>0.221523</td>
          <td>26.900668</td>
          <td>0.180803</td>
          <td>26.112619</td>
          <td>0.148360</td>
          <td>26.777165</td>
          <td>0.460511</td>
          <td>25.633606</td>
          <td>0.388774</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.195100</td>
          <td>2.016701</td>
          <td>27.438591</td>
          <td>0.332465</td>
          <td>26.847879</td>
          <td>0.184206</td>
          <td>25.701515</td>
          <td>0.111190</td>
          <td>25.545319</td>
          <td>0.181699</td>
          <td>25.606252</td>
          <td>0.404030</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.383080</td>
          <td>0.352571</td>
          <td>26.197991</td>
          <td>0.110222</td>
          <td>26.021047</td>
          <td>0.083528</td>
          <td>25.585073</td>
          <td>0.092809</td>
          <td>24.964677</td>
          <td>0.102243</td>
          <td>24.848018</td>
          <td>0.204028</td>
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
