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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fb52c89fe50>



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
          <td>27.115624</td>
          <td>0.233137</td>
          <td>26.043501</td>
          <td>0.081932</td>
          <td>25.385357</td>
          <td>0.074704</td>
          <td>25.184212</td>
          <td>0.119136</td>
          <td>24.964700</td>
          <td>0.216458</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>31.687404</td>
          <td>4.253067</td>
          <td>27.760188</td>
          <td>0.390903</td>
          <td>27.356359</td>
          <td>0.252462</td>
          <td>27.332576</td>
          <td>0.385519</td>
          <td>26.911965</td>
          <td>0.488547</td>
          <td>26.962725</td>
          <td>0.945173</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.422513</td>
          <td>0.354900</td>
          <td>25.900117</td>
          <td>0.082067</td>
          <td>24.815380</td>
          <td>0.027647</td>
          <td>23.875221</td>
          <td>0.019811</td>
          <td>23.156517</td>
          <td>0.020061</td>
          <td>22.837905</td>
          <td>0.033794</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.531246</td>
          <td>3.137231</td>
          <td>28.096336</td>
          <td>0.503847</td>
          <td>27.452755</td>
          <td>0.273157</td>
          <td>26.560677</td>
          <td>0.206530</td>
          <td>25.815757</td>
          <td>0.204461</td>
          <td>25.863290</td>
          <td>0.443429</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.252652</td>
          <td>0.310231</td>
          <td>25.738094</td>
          <td>0.071138</td>
          <td>25.408925</td>
          <td>0.046690</td>
          <td>24.817600</td>
          <td>0.045149</td>
          <td>24.302254</td>
          <td>0.054779</td>
          <td>23.632048</td>
          <td>0.068318</td>
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
          <td>27.774901</td>
          <td>0.923897</td>
          <td>26.212107</td>
          <td>0.107888</td>
          <td>26.231845</td>
          <td>0.096697</td>
          <td>26.054671</td>
          <td>0.134218</td>
          <td>25.927131</td>
          <td>0.224381</td>
          <td>25.194994</td>
          <td>0.261807</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.849348</td>
          <td>0.186601</td>
          <td>26.914123</td>
          <td>0.174445</td>
          <td>26.380718</td>
          <td>0.177458</td>
          <td>25.984859</td>
          <td>0.235381</td>
          <td>26.738081</td>
          <td>0.820324</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.840716</td>
          <td>0.185245</td>
          <td>27.216847</td>
          <td>0.224984</td>
          <td>26.493694</td>
          <td>0.195235</td>
          <td>26.348634</td>
          <td>0.316410</td>
          <td>26.106878</td>
          <td>0.531294</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.602238</td>
          <td>2.287022</td>
          <td>26.900685</td>
          <td>0.194852</td>
          <td>26.475581</td>
          <td>0.119639</td>
          <td>25.707756</td>
          <td>0.099228</td>
          <td>25.979321</td>
          <td>0.234305</td>
          <td>25.456977</td>
          <td>0.323447</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.569709</td>
          <td>0.397912</td>
          <td>26.869478</td>
          <td>0.189797</td>
          <td>26.159767</td>
          <td>0.090765</td>
          <td>25.689023</td>
          <td>0.097612</td>
          <td>25.179459</td>
          <td>0.118645</td>
          <td>24.700424</td>
          <td>0.173271</td>
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
          <td>27.178128</td>
          <td>0.683068</td>
          <td>27.033385</td>
          <td>0.249121</td>
          <td>26.128400</td>
          <td>0.103769</td>
          <td>25.410710</td>
          <td>0.090478</td>
          <td>24.892145</td>
          <td>0.108414</td>
          <td>24.886883</td>
          <td>0.237575</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.833919</td>
          <td>1.752489</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.753093</td>
          <td>0.400971</td>
          <td>28.197232</td>
          <td>0.819677</td>
          <td>27.089140</td>
          <td>0.633502</td>
          <td>26.060347</td>
          <td>0.589094</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.483729</td>
          <td>0.418874</td>
          <td>25.692602</td>
          <td>0.080538</td>
          <td>24.785684</td>
          <td>0.032399</td>
          <td>23.826957</td>
          <td>0.022937</td>
          <td>23.118247</td>
          <td>0.023253</td>
          <td>22.932289</td>
          <td>0.044516</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.254683</td>
          <td>1.357480</td>
          <td>27.812270</td>
          <td>0.484753</td>
          <td>27.633019</td>
          <td>0.387599</td>
          <td>27.014827</td>
          <td>0.371600</td>
          <td>26.455737</td>
          <td>0.421449</td>
          <td>24.857539</td>
          <td>0.247376</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.733746</td>
          <td>0.977856</td>
          <td>25.793170</td>
          <td>0.086249</td>
          <td>25.550373</td>
          <td>0.062364</td>
          <td>24.775183</td>
          <td>0.051591</td>
          <td>24.514977</td>
          <td>0.077872</td>
          <td>23.710146</td>
          <td>0.086619</td>
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
          <td>28.472791</td>
          <td>1.486944</td>
          <td>26.468431</td>
          <td>0.157904</td>
          <td>26.168207</td>
          <td>0.109710</td>
          <td>25.720360</td>
          <td>0.121195</td>
          <td>25.689975</td>
          <td>0.218915</td>
          <td>25.296714</td>
          <td>0.337652</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.376395</td>
          <td>0.329817</td>
          <td>26.818552</td>
          <td>0.188740</td>
          <td>26.395143</td>
          <td>0.211958</td>
          <td>26.083431</td>
          <td>0.297500</td>
          <td>26.272446</td>
          <td>0.685132</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.157851</td>
          <td>0.678787</td>
          <td>27.328407</td>
          <td>0.319712</td>
          <td>27.112474</td>
          <td>0.243179</td>
          <td>26.591393</td>
          <td>0.251485</td>
          <td>26.079969</td>
          <td>0.299018</td>
          <td>25.858116</td>
          <td>0.514495</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.891099</td>
          <td>0.500001</td>
          <td>27.154342</td>
          <td>0.256115</td>
          <td>25.868220</td>
          <td>0.139207</td>
          <td>26.034409</td>
          <td>0.293204</td>
          <td>25.589614</td>
          <td>0.427866</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.285015</td>
          <td>0.356235</td>
          <td>26.802769</td>
          <td>0.207524</td>
          <td>25.956119</td>
          <td>0.090119</td>
          <td>25.669758</td>
          <td>0.114682</td>
          <td>25.169919</td>
          <td>0.139336</td>
          <td>24.952398</td>
          <td>0.253177</td>
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
          <td>27.525849</td>
          <td>0.788371</td>
          <td>26.833817</td>
          <td>0.184188</td>
          <td>25.878139</td>
          <td>0.070804</td>
          <td>25.452018</td>
          <td>0.079246</td>
          <td>24.936990</td>
          <td>0.096007</td>
          <td>25.050660</td>
          <td>0.232517</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.354952</td>
          <td>0.703873</td>
          <td>27.918865</td>
          <td>0.441642</td>
          <td>28.317325</td>
          <td>0.533548</td>
          <td>27.102403</td>
          <td>0.322015</td>
          <td>28.429380</td>
          <td>1.296442</td>
          <td>25.975684</td>
          <td>0.482809</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.180301</td>
          <td>0.653355</td>
          <td>25.938731</td>
          <td>0.091254</td>
          <td>24.760802</td>
          <td>0.028610</td>
          <td>23.931835</td>
          <td>0.022608</td>
          <td>23.090798</td>
          <td>0.020548</td>
          <td>22.843442</td>
          <td>0.036994</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.993888</td>
          <td>0.625012</td>
          <td>28.413434</td>
          <td>0.739157</td>
          <td>27.391607</td>
          <td>0.319599</td>
          <td>26.444452</td>
          <td>0.234029</td>
          <td>25.672459</td>
          <td>0.224613</td>
          <td>26.447237</td>
          <td>0.806167</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.055578</td>
          <td>0.264831</td>
          <td>25.744261</td>
          <td>0.071616</td>
          <td>25.410207</td>
          <td>0.046810</td>
          <td>24.877690</td>
          <td>0.047695</td>
          <td>24.388202</td>
          <td>0.059205</td>
          <td>23.682911</td>
          <td>0.071570</td>
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
          <td>27.117220</td>
          <td>0.623979</td>
          <td>26.211043</td>
          <td>0.115428</td>
          <td>26.124220</td>
          <td>0.095185</td>
          <td>26.129035</td>
          <td>0.155104</td>
          <td>26.056612</td>
          <td>0.268697</td>
          <td>25.887269</td>
          <td>0.483969</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.131318</td>
          <td>0.239383</td>
          <td>26.656346</td>
          <td>0.142186</td>
          <td>26.163865</td>
          <td>0.149973</td>
          <td>25.790034</td>
          <td>0.203265</td>
          <td>24.972863</td>
          <td>0.221498</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.968651</td>
          <td>0.551820</td>
          <td>27.307308</td>
          <td>0.283823</td>
          <td>26.586367</td>
          <td>0.138193</td>
          <td>26.421455</td>
          <td>0.192957</td>
          <td>26.237232</td>
          <td>0.302578</td>
          <td>25.432852</td>
          <td>0.332193</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.636699</td>
          <td>2.400102</td>
          <td>27.566754</td>
          <td>0.367728</td>
          <td>26.573956</td>
          <td>0.145826</td>
          <td>25.757378</td>
          <td>0.116735</td>
          <td>25.457926</td>
          <td>0.168707</td>
          <td>25.168199</td>
          <td>0.285894</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.896915</td>
          <td>1.014149</td>
          <td>26.630005</td>
          <td>0.160050</td>
          <td>26.043739</td>
          <td>0.085215</td>
          <td>25.595601</td>
          <td>0.093671</td>
          <td>25.150493</td>
          <td>0.120233</td>
          <td>24.917354</td>
          <td>0.216207</td>
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
