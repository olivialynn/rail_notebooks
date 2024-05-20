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
    from rail.creation.degraders.lsst_error_model import LSSTErrorModel
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

    <pzflow.flow.Flow at 0x7fcf5e74b430>



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
    0      0.890625  27.370831  26.712662  26.025223  25.327188  25.016500   
    1      1.978239  29.557049  28.361185  27.587231  27.238544  26.628109   
    2      0.974287  26.566015  25.937716  24.787413  23.872456  23.139563   
    3      1.317979  29.042730  28.274593  27.501106  26.648790  26.091450   
    4      1.386366  26.292624  25.774778  25.429958  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362207  27.036276  26.823139  26.420132  26.110037   
    99997  1.372992  27.736044  27.271955  26.887581  26.416138  26.043434   
    99998  0.855022  28.044552  27.327116  26.599014  25.862331  25.592169   
    99999  1.723768  27.049067  26.526745  26.094595  25.642971  25.197956   
    
                   y     major     minor  
    0      24.926821  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346500  0.147522  0.143359  
    4      23.700010  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524904  0.044537  0.022302  
    99997  25.456165  0.073146  0.047825  
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
          <td>26.506591</td>
          <td>0.139280</td>
          <td>26.100650</td>
          <td>0.086164</td>
          <td>25.346362</td>
          <td>0.072172</td>
          <td>25.024665</td>
          <td>0.103659</td>
          <td>25.113363</td>
          <td>0.244844</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.928961</td>
          <td>0.444710</td>
          <td>27.467202</td>
          <td>0.276384</td>
          <td>28.803780</td>
          <td>1.058954</td>
          <td>25.860873</td>
          <td>0.212329</td>
          <td>25.674650</td>
          <td>0.383788</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.008980</td>
          <td>0.552274</td>
          <td>25.860896</td>
          <td>0.079280</td>
          <td>24.800243</td>
          <td>0.027284</td>
          <td>23.873668</td>
          <td>0.019785</td>
          <td>23.123933</td>
          <td>0.019516</td>
          <td>22.875826</td>
          <td>0.034944</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317979</td>
          <td>27.687798</td>
          <td>0.874871</td>
          <td>27.536532</td>
          <td>0.328058</td>
          <td>27.139303</td>
          <td>0.210903</td>
          <td>26.725360</td>
          <td>0.236866</td>
          <td>25.856925</td>
          <td>0.211629</td>
          <td>26.117344</td>
          <td>0.535355</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.353214</td>
          <td>0.336053</td>
          <td>25.741134</td>
          <td>0.071329</td>
          <td>25.497343</td>
          <td>0.050503</td>
          <td>24.810265</td>
          <td>0.044856</td>
          <td>24.269658</td>
          <td>0.053216</td>
          <td>23.519449</td>
          <td>0.061830</td>
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
          <td>26.679169</td>
          <td>0.432633</td>
          <td>26.161369</td>
          <td>0.103212</td>
          <td>26.243683</td>
          <td>0.097706</td>
          <td>26.007170</td>
          <td>0.128814</td>
          <td>26.215576</td>
          <td>0.284307</td>
          <td>25.676887</td>
          <td>0.384455</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.445027</td>
          <td>0.361213</td>
          <td>26.960953</td>
          <td>0.204964</td>
          <td>26.481086</td>
          <td>0.120213</td>
          <td>26.457303</td>
          <td>0.189338</td>
          <td>25.888951</td>
          <td>0.217362</td>
          <td>25.431736</td>
          <td>0.317005</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372992</td>
          <td>26.461098</td>
          <td>0.365777</td>
          <td>27.482797</td>
          <td>0.314314</td>
          <td>27.086297</td>
          <td>0.201744</td>
          <td>26.506689</td>
          <td>0.197381</td>
          <td>26.234599</td>
          <td>0.288716</td>
          <td>24.703704</td>
          <td>0.173755</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.657543</td>
          <td>0.425583</td>
          <td>27.367879</td>
          <td>0.286584</td>
          <td>26.469686</td>
          <td>0.119027</td>
          <td>25.754511</td>
          <td>0.103376</td>
          <td>25.476906</td>
          <td>0.153394</td>
          <td>25.255740</td>
          <td>0.275099</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.817959</td>
          <td>0.948765</td>
          <td>26.409975</td>
          <td>0.128132</td>
          <td>26.250061</td>
          <td>0.098254</td>
          <td>25.737874</td>
          <td>0.101881</td>
          <td>25.155787</td>
          <td>0.116226</td>
          <td>24.752357</td>
          <td>0.181075</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.476736</td>
          <td>0.156119</td>
          <td>26.115093</td>
          <td>0.102568</td>
          <td>25.350082</td>
          <td>0.085779</td>
          <td>25.026172</td>
          <td>0.121832</td>
          <td>25.152932</td>
          <td>0.295192</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.872816</td>
          <td>0.481694</td>
          <td>27.446436</td>
          <td>0.315230</td>
          <td>29.863416</td>
          <td>1.986264</td>
          <td>25.763926</td>
          <td>0.228263</td>
          <td>25.592576</td>
          <td>0.417076</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.094905</td>
          <td>0.653758</td>
          <td>25.847218</td>
          <td>0.092259</td>
          <td>24.802902</td>
          <td>0.032894</td>
          <td>23.873921</td>
          <td>0.023884</td>
          <td>23.120837</td>
          <td>0.023305</td>
          <td>22.885501</td>
          <td>0.042708</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317979</td>
          <td>27.547362</td>
          <td>0.903876</td>
          <td>27.416083</td>
          <td>0.358206</td>
          <td>27.059113</td>
          <td>0.244885</td>
          <td>26.748012</td>
          <td>0.300830</td>
          <td>25.801382</td>
          <td>0.250697</td>
          <td>26.488470</td>
          <td>0.830188</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.361405</td>
          <td>0.375642</td>
          <td>25.735811</td>
          <td>0.082006</td>
          <td>25.510132</td>
          <td>0.060179</td>
          <td>24.810980</td>
          <td>0.053256</td>
          <td>24.252569</td>
          <td>0.061740</td>
          <td>23.488192</td>
          <td>0.071214</td>
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
          <td>26.699744</td>
          <td>0.492257</td>
          <td>26.129906</td>
          <td>0.117944</td>
          <td>26.267895</td>
          <td>0.119659</td>
          <td>25.991370</td>
          <td>0.153128</td>
          <td>26.292148</td>
          <td>0.356566</td>
          <td>25.704617</td>
          <td>0.462422</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.364814</td>
          <td>0.377613</td>
          <td>26.948947</td>
          <td>0.233190</td>
          <td>26.426909</td>
          <td>0.135072</td>
          <td>26.464822</td>
          <td>0.224628</td>
          <td>25.851746</td>
          <td>0.246359</td>
          <td>25.414347</td>
          <td>0.364694</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372992</td>
          <td>26.359293</td>
          <td>0.378188</td>
          <td>27.524415</td>
          <td>0.373104</td>
          <td>27.130836</td>
          <td>0.246884</td>
          <td>26.526642</td>
          <td>0.238426</td>
          <td>26.277226</td>
          <td>0.349836</td>
          <td>24.596556</td>
          <td>0.188758</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.539324</td>
          <td>0.439586</td>
          <td>27.376068</td>
          <td>0.337136</td>
          <td>26.442757</td>
          <td>0.140644</td>
          <td>25.730781</td>
          <td>0.123605</td>
          <td>25.452735</td>
          <td>0.181157</td>
          <td>25.205041</td>
          <td>0.316982</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.986759</td>
          <td>1.140443</td>
          <td>26.391068</td>
          <td>0.146357</td>
          <td>26.283366</td>
          <td>0.119970</td>
          <td>25.758437</td>
          <td>0.123872</td>
          <td>25.147697</td>
          <td>0.136691</td>
          <td>24.724489</td>
          <td>0.209609</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.506569</td>
          <td>0.139293</td>
          <td>26.100661</td>
          <td>0.086176</td>
          <td>25.346365</td>
          <td>0.072182</td>
          <td>25.024667</td>
          <td>0.103672</td>
          <td>25.113392</td>
          <td>0.244881</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.928655</td>
          <td>0.444921</td>
          <td>27.467089</td>
          <td>0.276600</td>
          <td>28.807436</td>
          <td>1.061938</td>
          <td>25.860329</td>
          <td>0.212423</td>
          <td>25.674190</td>
          <td>0.383985</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.044100</td>
          <td>0.593930</td>
          <td>25.855129</td>
          <td>0.084791</td>
          <td>24.801361</td>
          <td>0.029645</td>
          <td>23.873774</td>
          <td>0.021511</td>
          <td>23.122631</td>
          <td>0.021112</td>
          <td>22.879893</td>
          <td>0.038206</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317979</td>
          <td>27.549735</td>
          <td>0.903422</td>
          <td>27.418105</td>
          <td>0.357734</td>
          <td>27.060469</td>
          <td>0.244340</td>
          <td>26.747612</td>
          <td>0.299722</td>
          <td>25.802332</td>
          <td>0.250057</td>
          <td>26.480765</td>
          <td>0.823863</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.353279</td>
          <td>0.336367</td>
          <td>25.741092</td>
          <td>0.071415</td>
          <td>25.497446</td>
          <td>0.050580</td>
          <td>24.810271</td>
          <td>0.044924</td>
          <td>24.269520</td>
          <td>0.053286</td>
          <td>23.519192</td>
          <td>0.061908</td>
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
          <td>26.687391</td>
          <td>0.456718</td>
          <td>26.148506</td>
          <td>0.109311</td>
          <td>26.253492</td>
          <td>0.106595</td>
          <td>26.000683</td>
          <td>0.138905</td>
          <td>26.246158</td>
          <td>0.313132</td>
          <td>25.688133</td>
          <td>0.416510</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.437567</td>
          <td>0.362800</td>
          <td>26.959855</td>
          <td>0.207590</td>
          <td>26.475997</td>
          <td>0.121649</td>
          <td>26.457994</td>
          <td>0.192616</td>
          <td>25.885481</td>
          <td>0.220142</td>
          <td>25.430124</td>
          <td>0.321549</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372992</td>
          <td>26.433898</td>
          <td>0.369221</td>
          <td>27.493489</td>
          <td>0.329507</td>
          <td>27.097809</td>
          <td>0.213414</td>
          <td>26.511903</td>
          <td>0.208184</td>
          <td>26.245624</td>
          <td>0.304623</td>
          <td>24.674489</td>
          <td>0.178014</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.589383</td>
          <td>0.433862</td>
          <td>27.372514</td>
          <td>0.315442</td>
          <td>26.454263</td>
          <td>0.131525</td>
          <td>25.740906</td>
          <td>0.115073</td>
          <td>25.463071</td>
          <td>0.169447</td>
          <td>25.226528</td>
          <td>0.299665</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.850839</td>
          <td>0.986425</td>
          <td>26.405986</td>
          <td>0.132029</td>
          <td>26.257012</td>
          <td>0.102767</td>
          <td>25.742199</td>
          <td>0.106509</td>
          <td>25.154073</td>
          <td>0.120607</td>
          <td>24.746395</td>
          <td>0.187305</td>
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
