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

    <pzflow.flow.Flow at 0x7f0494239a80>



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
          <td>26.538140</td>
          <td>0.388337</td>
          <td>26.754148</td>
          <td>0.172145</td>
          <td>26.019623</td>
          <td>0.080224</td>
          <td>25.431427</td>
          <td>0.077808</td>
          <td>25.065577</td>
          <td>0.107433</td>
          <td>25.235919</td>
          <td>0.270698</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.362325</td>
          <td>1.297777</td>
          <td>27.794269</td>
          <td>0.401312</td>
          <td>27.661560</td>
          <td>0.323154</td>
          <td>27.293970</td>
          <td>0.374130</td>
          <td>26.445462</td>
          <td>0.341697</td>
          <td>27.206038</td>
          <td>1.093126</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.722785</td>
          <td>0.447142</td>
          <td>25.971523</td>
          <td>0.087388</td>
          <td>24.795156</td>
          <td>0.027163</td>
          <td>23.896231</td>
          <td>0.020166</td>
          <td>23.121494</td>
          <td>0.019476</td>
          <td>22.790351</td>
          <td>0.032407</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.652563</td>
          <td>2.331344</td>
          <td>28.463020</td>
          <td>0.654721</td>
          <td>27.799624</td>
          <td>0.360381</td>
          <td>26.313548</td>
          <td>0.167609</td>
          <td>25.928651</td>
          <td>0.224664</td>
          <td>25.345991</td>
          <td>0.295942</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.597196</td>
          <td>0.406407</td>
          <td>25.663374</td>
          <td>0.066592</td>
          <td>25.427527</td>
          <td>0.047467</td>
          <td>24.830700</td>
          <td>0.045677</td>
          <td>24.380348</td>
          <td>0.058710</td>
          <td>23.647172</td>
          <td>0.069239</td>
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
          <td>26.536232</td>
          <td>0.387765</td>
          <td>26.603855</td>
          <td>0.151422</td>
          <td>26.099766</td>
          <td>0.086097</td>
          <td>25.852003</td>
          <td>0.112565</td>
          <td>25.855587</td>
          <td>0.211393</td>
          <td>25.735926</td>
          <td>0.402388</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.654574</td>
          <td>1.509528</td>
          <td>27.165153</td>
          <td>0.242869</td>
          <td>26.745609</td>
          <td>0.151070</td>
          <td>26.601462</td>
          <td>0.213696</td>
          <td>26.023407</td>
          <td>0.242993</td>
          <td>25.637996</td>
          <td>0.373010</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.842782</td>
          <td>0.963291</td>
          <td>27.416380</td>
          <td>0.298016</td>
          <td>26.823072</td>
          <td>0.161427</td>
          <td>26.341466</td>
          <td>0.171640</td>
          <td>26.297031</td>
          <td>0.303605</td>
          <td>25.242348</td>
          <td>0.272119</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.229660</td>
          <td>1.967256</td>
          <td>26.960226</td>
          <td>0.204839</td>
          <td>26.900108</td>
          <td>0.172380</td>
          <td>25.912032</td>
          <td>0.118605</td>
          <td>25.573546</td>
          <td>0.166601</td>
          <td>25.147617</td>
          <td>0.251839</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.353251</td>
          <td>0.336063</td>
          <td>26.308987</td>
          <td>0.117386</td>
          <td>26.254062</td>
          <td>0.098599</td>
          <td>25.587813</td>
          <td>0.089309</td>
          <td>25.099715</td>
          <td>0.110683</td>
          <td>25.487797</td>
          <td>0.331465</td>
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
          <td>27.182220</td>
          <td>0.684979</td>
          <td>26.846746</td>
          <td>0.213444</td>
          <td>26.016031</td>
          <td>0.094039</td>
          <td>25.349139</td>
          <td>0.085708</td>
          <td>24.988777</td>
          <td>0.117937</td>
          <td>24.962260</td>
          <td>0.252788</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.940290</td>
          <td>0.506338</td>
          <td>27.708510</td>
          <td>0.387407</td>
          <td>27.218838</td>
          <td>0.409424</td>
          <td>26.698334</td>
          <td>0.477825</td>
          <td>29.274231</td>
          <td>2.932938</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.686514</td>
          <td>0.487882</td>
          <td>26.159538</td>
          <td>0.121160</td>
          <td>24.803608</td>
          <td>0.032914</td>
          <td>23.841191</td>
          <td>0.023219</td>
          <td>23.170625</td>
          <td>0.024327</td>
          <td>22.890452</td>
          <td>0.042895</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.657085</td>
          <td>0.431429</td>
          <td>27.889298</td>
          <td>0.471032</td>
          <td>26.965698</td>
          <td>0.357598</td>
          <td>26.169691</td>
          <td>0.337434</td>
          <td>24.876776</td>
          <td>0.251318</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.644571</td>
          <td>0.466201</td>
          <td>25.753157</td>
          <td>0.083267</td>
          <td>25.373475</td>
          <td>0.053309</td>
          <td>24.732143</td>
          <td>0.049657</td>
          <td>24.371976</td>
          <td>0.068626</td>
          <td>23.619775</td>
          <td>0.079990</td>
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
          <td>26.112102</td>
          <td>0.313052</td>
          <td>26.295606</td>
          <td>0.136132</td>
          <td>26.195683</td>
          <td>0.112371</td>
          <td>25.945485</td>
          <td>0.147216</td>
          <td>25.826206</td>
          <td>0.245062</td>
          <td>25.623513</td>
          <td>0.434989</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.753049</td>
          <td>0.991340</td>
          <td>26.904232</td>
          <td>0.224706</td>
          <td>26.560683</td>
          <td>0.151553</td>
          <td>26.242514</td>
          <td>0.186448</td>
          <td>26.073580</td>
          <td>0.295149</td>
          <td>25.517324</td>
          <td>0.395061</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.942827</td>
          <td>0.584187</td>
          <td>27.521502</td>
          <td>0.372258</td>
          <td>27.264550</td>
          <td>0.275413</td>
          <td>26.694699</td>
          <td>0.273640</td>
          <td>25.976457</td>
          <td>0.275010</td>
          <td>26.626721</td>
          <td>0.870834</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.789143</td>
          <td>0.463506</td>
          <td>26.573527</td>
          <td>0.157354</td>
          <td>25.827723</td>
          <td>0.134426</td>
          <td>25.580095</td>
          <td>0.201688</td>
          <td>25.276592</td>
          <td>0.335532</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.545272</td>
          <td>1.534220</td>
          <td>26.404449</td>
          <td>0.148048</td>
          <td>26.122127</td>
          <td>0.104239</td>
          <td>25.561662</td>
          <td>0.104359</td>
          <td>25.078003</td>
          <td>0.128700</td>
          <td>24.891441</td>
          <td>0.240792</td>
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
          <td>26.329146</td>
          <td>0.329737</td>
          <td>26.701842</td>
          <td>0.164669</td>
          <td>26.153432</td>
          <td>0.090273</td>
          <td>25.398804</td>
          <td>0.075608</td>
          <td>24.951388</td>
          <td>0.097227</td>
          <td>24.738521</td>
          <td>0.178989</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.033595</td>
          <td>0.950880</td>
          <td>28.153354</td>
          <td>0.472816</td>
          <td>27.537989</td>
          <td>0.451435</td>
          <td>26.486737</td>
          <td>0.353288</td>
          <td>26.038378</td>
          <td>0.505723</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.183086</td>
          <td>0.309510</td>
          <td>26.047036</td>
          <td>0.100338</td>
          <td>24.776515</td>
          <td>0.029006</td>
          <td>23.905940</td>
          <td>0.022111</td>
          <td>23.170835</td>
          <td>0.022000</td>
          <td>22.854221</td>
          <td>0.037348</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.986872</td>
          <td>0.549392</td>
          <td>27.032372</td>
          <td>0.238743</td>
          <td>26.767574</td>
          <td>0.304566</td>
          <td>26.038215</td>
          <td>0.302882</td>
          <td>27.234675</td>
          <td>1.287689</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.229172</td>
          <td>0.304728</td>
          <td>25.826285</td>
          <td>0.076992</td>
          <td>25.479011</td>
          <td>0.049759</td>
          <td>24.719821</td>
          <td>0.041460</td>
          <td>24.412571</td>
          <td>0.060499</td>
          <td>23.588254</td>
          <td>0.065816</td>
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
          <td>26.994948</td>
          <td>0.572264</td>
          <td>26.282478</td>
          <td>0.122816</td>
          <td>26.137555</td>
          <td>0.096306</td>
          <td>25.895290</td>
          <td>0.126808</td>
          <td>25.629486</td>
          <td>0.188473</td>
          <td>24.826955</td>
          <td>0.208486</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.908881</td>
          <td>0.518439</td>
          <td>26.563907</td>
          <td>0.148383</td>
          <td>27.016544</td>
          <td>0.193274</td>
          <td>26.193621</td>
          <td>0.153850</td>
          <td>26.577700</td>
          <td>0.384538</td>
          <td>25.359031</td>
          <td>0.303778</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.479039</td>
          <td>0.784384</td>
          <td>26.839570</td>
          <td>0.192815</td>
          <td>27.259581</td>
          <td>0.244065</td>
          <td>26.766979</td>
          <td>0.257164</td>
          <td>25.880571</td>
          <td>0.226071</td>
          <td>25.025557</td>
          <td>0.238845</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.389382</td>
          <td>0.766343</td>
          <td>27.040844</td>
          <td>0.240974</td>
          <td>26.654097</td>
          <td>0.156204</td>
          <td>26.166277</td>
          <td>0.166052</td>
          <td>25.655025</td>
          <td>0.199312</td>
          <td>26.836109</td>
          <td>0.950012</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.633649</td>
          <td>0.427900</td>
          <td>26.483959</td>
          <td>0.141212</td>
          <td>26.220736</td>
          <td>0.099554</td>
          <td>25.641790</td>
          <td>0.097546</td>
          <td>24.996238</td>
          <td>0.105105</td>
          <td>25.320241</td>
          <td>0.300789</td>
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
