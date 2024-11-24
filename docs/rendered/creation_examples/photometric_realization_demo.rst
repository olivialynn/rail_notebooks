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

    <pzflow.flow.Flow at 0x7fc8eeebe590>



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
          <td>27.201089</td>
          <td>0.632907</td>
          <td>26.778208</td>
          <td>0.175697</td>
          <td>26.035520</td>
          <td>0.081357</td>
          <td>25.311398</td>
          <td>0.069973</td>
          <td>25.092387</td>
          <td>0.109978</td>
          <td>25.028323</td>
          <td>0.228222</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.310149</td>
          <td>0.243045</td>
          <td>27.310289</td>
          <td>0.378910</td>
          <td>26.668584</td>
          <td>0.406556</td>
          <td>25.869949</td>
          <td>0.445666</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.909387</td>
          <td>0.234678</td>
          <td>25.932884</td>
          <td>0.084468</td>
          <td>24.732736</td>
          <td>0.025725</td>
          <td>23.877275</td>
          <td>0.019845</td>
          <td>23.128274</td>
          <td>0.019588</td>
          <td>22.863703</td>
          <td>0.034572</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.818704</td>
          <td>0.830246</td>
          <td>27.435756</td>
          <td>0.269402</td>
          <td>26.734531</td>
          <td>0.238667</td>
          <td>26.284578</td>
          <td>0.300583</td>
          <td>26.061075</td>
          <td>0.513805</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.935507</td>
          <td>0.239789</td>
          <td>25.746718</td>
          <td>0.071682</td>
          <td>25.473120</td>
          <td>0.049428</td>
          <td>24.783806</td>
          <td>0.043815</td>
          <td>24.420777</td>
          <td>0.060854</td>
          <td>23.623097</td>
          <td>0.067778</td>
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
          <td>26.502622</td>
          <td>0.377796</td>
          <td>26.855192</td>
          <td>0.187523</td>
          <td>25.932661</td>
          <td>0.074293</td>
          <td>26.058954</td>
          <td>0.134716</td>
          <td>26.153031</td>
          <td>0.270227</td>
          <td>26.236253</td>
          <td>0.583184</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.951363</td>
          <td>0.203324</td>
          <td>26.818631</td>
          <td>0.160816</td>
          <td>26.850683</td>
          <td>0.262568</td>
          <td>26.625660</td>
          <td>0.393338</td>
          <td>25.278848</td>
          <td>0.280309</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.348896</td>
          <td>1.288432</td>
          <td>27.442902</td>
          <td>0.304434</td>
          <td>27.224332</td>
          <td>0.226387</td>
          <td>26.668488</td>
          <td>0.225963</td>
          <td>26.308217</td>
          <td>0.306342</td>
          <td>25.907984</td>
          <td>0.458614</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.598590</td>
          <td>0.826450</td>
          <td>27.847966</td>
          <td>0.418177</td>
          <td>26.534719</td>
          <td>0.125942</td>
          <td>25.737715</td>
          <td>0.101867</td>
          <td>25.646554</td>
          <td>0.177270</td>
          <td>25.423038</td>
          <td>0.314811</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.617786</td>
          <td>0.412868</td>
          <td>26.685802</td>
          <td>0.162414</td>
          <td>26.057602</td>
          <td>0.082957</td>
          <td>25.720846</td>
          <td>0.100373</td>
          <td>25.219046</td>
          <td>0.122797</td>
          <td>24.965963</td>
          <td>0.216686</td>
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
          <td>26.651697</td>
          <td>0.468593</td>
          <td>27.268507</td>
          <td>0.301568</td>
          <td>26.101226</td>
          <td>0.101331</td>
          <td>25.284608</td>
          <td>0.080970</td>
          <td>24.795835</td>
          <td>0.099658</td>
          <td>25.254595</td>
          <td>0.320247</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.738862</td>
          <td>1.677044</td>
          <td>28.193365</td>
          <td>0.607619</td>
          <td>28.191530</td>
          <td>0.556052</td>
          <td>27.487031</td>
          <td>0.500970</td>
          <td>26.367266</td>
          <td>0.371216</td>
          <td>25.306749</td>
          <td>0.333861</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.169365</td>
          <td>0.688043</td>
          <td>25.948663</td>
          <td>0.100829</td>
          <td>24.780426</td>
          <td>0.032249</td>
          <td>23.882340</td>
          <td>0.024058</td>
          <td>23.178863</td>
          <td>0.024500</td>
          <td>22.859670</td>
          <td>0.041742</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.362693</td>
          <td>1.435612</td>
          <td>29.668408</td>
          <td>1.534205</td>
          <td>27.372085</td>
          <td>0.315681</td>
          <td>26.748415</td>
          <td>0.300927</td>
          <td>25.832371</td>
          <td>0.257151</td>
          <td>26.295434</td>
          <td>0.731317</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>29.382518</td>
          <td>2.211801</td>
          <td>25.727203</td>
          <td>0.081387</td>
          <td>25.489977</td>
          <td>0.059113</td>
          <td>24.777257</td>
          <td>0.051686</td>
          <td>24.355858</td>
          <td>0.067654</td>
          <td>23.645417</td>
          <td>0.081819</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.275104</td>
          <td>0.133745</td>
          <td>26.228322</td>
          <td>0.115611</td>
          <td>26.043223</td>
          <td>0.160075</td>
          <td>25.569630</td>
          <td>0.197944</td>
          <td>25.713364</td>
          <td>0.465462</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.320462</td>
          <td>0.753602</td>
          <td>26.997492</td>
          <td>0.242725</td>
          <td>26.690623</td>
          <td>0.169347</td>
          <td>26.563316</td>
          <td>0.243702</td>
          <td>26.130000</td>
          <td>0.308834</td>
          <td>25.149330</td>
          <td>0.295496</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.793898</td>
          <td>1.020513</td>
          <td>27.610846</td>
          <td>0.398931</td>
          <td>26.888556</td>
          <td>0.201852</td>
          <td>26.321704</td>
          <td>0.201015</td>
          <td>27.125645</td>
          <td>0.656244</td>
          <td>25.121261</td>
          <td>0.291238</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.981905</td>
          <td>0.607532</td>
          <td>26.982702</td>
          <td>0.245389</td>
          <td>26.650501</td>
          <td>0.168038</td>
          <td>25.920500</td>
          <td>0.145615</td>
          <td>26.166762</td>
          <td>0.325983</td>
          <td>25.210150</td>
          <td>0.318276</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.842489</td>
          <td>1.048885</td>
          <td>26.666510</td>
          <td>0.185064</td>
          <td>26.237989</td>
          <td>0.115328</td>
          <td>25.669485</td>
          <td>0.114655</td>
          <td>25.423388</td>
          <td>0.173097</td>
          <td>25.013366</td>
          <td>0.266127</td>
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
          <td>26.137710</td>
          <td>0.282863</td>
          <td>26.860356</td>
          <td>0.188363</td>
          <td>25.950111</td>
          <td>0.075457</td>
          <td>25.251930</td>
          <td>0.066392</td>
          <td>25.077070</td>
          <td>0.108531</td>
          <td>25.026285</td>
          <td>0.227866</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.423244</td>
          <td>1.341135</td>
          <td>29.965146</td>
          <td>1.584475</td>
          <td>28.442280</td>
          <td>0.583765</td>
          <td>27.043998</td>
          <td>0.307333</td>
          <td>26.871101</td>
          <td>0.474303</td>
          <td>25.765027</td>
          <td>0.411832</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.987062</td>
          <td>0.264195</td>
          <td>26.090598</td>
          <td>0.104232</td>
          <td>24.743242</td>
          <td>0.028174</td>
          <td>23.868462</td>
          <td>0.021413</td>
          <td>23.103507</td>
          <td>0.020771</td>
          <td>22.847807</td>
          <td>0.037137</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.005245</td>
          <td>0.296151</td>
          <td>27.828454</td>
          <td>0.489260</td>
          <td>26.979747</td>
          <td>0.228569</td>
          <td>26.517056</td>
          <td>0.248472</td>
          <td>26.126793</td>
          <td>0.325098</td>
          <td>25.888422</td>
          <td>0.549147</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.421946</td>
          <td>0.355054</td>
          <td>25.817164</td>
          <td>0.076375</td>
          <td>25.401923</td>
          <td>0.046467</td>
          <td>24.785797</td>
          <td>0.043959</td>
          <td>24.310192</td>
          <td>0.055245</td>
          <td>23.730093</td>
          <td>0.074619</td>
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
          <td>26.555639</td>
          <td>0.413318</td>
          <td>26.187075</td>
          <td>0.113046</td>
          <td>26.127592</td>
          <td>0.095467</td>
          <td>26.052283</td>
          <td>0.145217</td>
          <td>26.201222</td>
          <td>0.302057</td>
          <td>25.480069</td>
          <td>0.354479</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.067105</td>
          <td>1.843637</td>
          <td>27.113179</td>
          <td>0.235823</td>
          <td>26.734057</td>
          <td>0.152006</td>
          <td>26.395099</td>
          <td>0.182653</td>
          <td>26.785456</td>
          <td>0.450727</td>
          <td>25.964826</td>
          <td>0.485499</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.996151</td>
          <td>0.484984</td>
          <td>26.985241</td>
          <td>0.194189</td>
          <td>26.518361</td>
          <td>0.209312</td>
          <td>26.249809</td>
          <td>0.305648</td>
          <td>25.993583</td>
          <td>0.510125</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.625148</td>
          <td>0.445756</td>
          <td>27.067265</td>
          <td>0.246276</td>
          <td>26.614477</td>
          <td>0.150990</td>
          <td>25.770462</td>
          <td>0.118072</td>
          <td>25.449011</td>
          <td>0.167431</td>
          <td>27.614285</td>
          <td>1.467655</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.571139</td>
          <td>0.407961</td>
          <td>26.777516</td>
          <td>0.181433</td>
          <td>25.952793</td>
          <td>0.078648</td>
          <td>25.647854</td>
          <td>0.098066</td>
          <td>24.976114</td>
          <td>0.103271</td>
          <td>25.170768</td>
          <td>0.266493</td>
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
