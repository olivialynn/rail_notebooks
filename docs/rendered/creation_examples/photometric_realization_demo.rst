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

    <pzflow.flow.Flow at 0x7fd90941b040>



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
          <td>26.903856</td>
          <td>0.511615</td>
          <td>27.029406</td>
          <td>0.217028</td>
          <td>26.099721</td>
          <td>0.086094</td>
          <td>25.341398</td>
          <td>0.071855</td>
          <td>24.953697</td>
          <td>0.097411</td>
          <td>25.075965</td>
          <td>0.237405</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.795316</td>
          <td>0.401635</td>
          <td>28.238552</td>
          <td>0.503252</td>
          <td>26.635822</td>
          <td>0.219907</td>
          <td>26.777571</td>
          <td>0.441768</td>
          <td>25.410246</td>
          <td>0.311607</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.304972</td>
          <td>0.323443</td>
          <td>26.004040</td>
          <td>0.089921</td>
          <td>24.801257</td>
          <td>0.027308</td>
          <td>23.858060</td>
          <td>0.019525</td>
          <td>23.129808</td>
          <td>0.019613</td>
          <td>22.869482</td>
          <td>0.034749</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.724042</td>
          <td>0.380110</td>
          <td>28.095554</td>
          <td>0.452401</td>
          <td>27.305072</td>
          <td>0.377376</td>
          <td>25.857799</td>
          <td>0.211784</td>
          <td>25.371210</td>
          <td>0.302007</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.059717</td>
          <td>0.572767</td>
          <td>25.807644</td>
          <td>0.075643</td>
          <td>25.396781</td>
          <td>0.046189</td>
          <td>24.826287</td>
          <td>0.045498</td>
          <td>24.324279</td>
          <td>0.055860</td>
          <td>23.842030</td>
          <td>0.082247</td>
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
          <td>26.621365</td>
          <td>0.414000</td>
          <td>26.240931</td>
          <td>0.110634</td>
          <td>26.099321</td>
          <td>0.086063</td>
          <td>26.291182</td>
          <td>0.164444</td>
          <td>25.919106</td>
          <td>0.222889</td>
          <td>25.982693</td>
          <td>0.484926</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.901095</td>
          <td>0.510580</td>
          <td>27.075098</td>
          <td>0.225435</td>
          <td>27.256791</td>
          <td>0.232563</td>
          <td>26.125960</td>
          <td>0.142730</td>
          <td>27.044128</td>
          <td>0.538295</td>
          <td>25.106068</td>
          <td>0.243376</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.464267</td>
          <td>0.756985</td>
          <td>26.940565</td>
          <td>0.201491</td>
          <td>26.747213</td>
          <td>0.151278</td>
          <td>26.255769</td>
          <td>0.159545</td>
          <td>26.591722</td>
          <td>0.383142</td>
          <td>25.026123</td>
          <td>0.227806</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.289954</td>
          <td>0.673028</td>
          <td>27.376044</td>
          <td>0.288481</td>
          <td>27.007550</td>
          <td>0.188805</td>
          <td>25.911667</td>
          <td>0.118568</td>
          <td>25.408227</td>
          <td>0.144610</td>
          <td>25.721762</td>
          <td>0.398023</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.763134</td>
          <td>0.460912</td>
          <td>26.780209</td>
          <td>0.175995</td>
          <td>26.097956</td>
          <td>0.085960</td>
          <td>25.637293</td>
          <td>0.093279</td>
          <td>25.200696</td>
          <td>0.120855</td>
          <td>24.784245</td>
          <td>0.186026</td>
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
          <td>26.334069</td>
          <td>0.367652</td>
          <td>27.199544</td>
          <td>0.285264</td>
          <td>25.994795</td>
          <td>0.092302</td>
          <td>25.419961</td>
          <td>0.091217</td>
          <td>25.224475</td>
          <td>0.144608</td>
          <td>24.929003</td>
          <td>0.245972</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.060735</td>
          <td>1.183439</td>
          <td>27.488871</td>
          <td>0.359246</td>
          <td>27.060099</td>
          <td>0.230137</td>
          <td>27.100634</td>
          <td>0.373679</td>
          <td>27.812551</td>
          <td>1.013061</td>
          <td>28.148004</td>
          <td>1.931969</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>28.209549</td>
          <td>1.297781</td>
          <td>25.976107</td>
          <td>0.103277</td>
          <td>24.770593</td>
          <td>0.031972</td>
          <td>23.904589</td>
          <td>0.024525</td>
          <td>23.168269</td>
          <td>0.024277</td>
          <td>22.904917</td>
          <td>0.043449</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.678433</td>
          <td>0.880082</td>
          <td>27.021118</td>
          <td>0.237329</td>
          <td>26.381538</td>
          <td>0.222903</td>
          <td>25.742047</td>
          <td>0.238741</td>
          <td>24.492564</td>
          <td>0.182402</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.823046</td>
          <td>0.531800</td>
          <td>25.719675</td>
          <td>0.080850</td>
          <td>25.539205</td>
          <td>0.061750</td>
          <td>24.762883</td>
          <td>0.051030</td>
          <td>24.213440</td>
          <td>0.059635</td>
          <td>23.707855</td>
          <td>0.086444</td>
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
          <td>26.590831</td>
          <td>0.453862</td>
          <td>26.677496</td>
          <td>0.188575</td>
          <td>26.157705</td>
          <td>0.108709</td>
          <td>26.063505</td>
          <td>0.162872</td>
          <td>25.906897</td>
          <td>0.261836</td>
          <td>25.074608</td>
          <td>0.282641</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.950264</td>
          <td>1.849431</td>
          <td>27.131699</td>
          <td>0.270924</td>
          <td>26.645149</td>
          <td>0.162910</td>
          <td>26.161369</td>
          <td>0.174063</td>
          <td>26.425278</td>
          <td>0.389681</td>
          <td>25.092787</td>
          <td>0.282301</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.403030</td>
          <td>0.799564</td>
          <td>27.377458</td>
          <td>0.332420</td>
          <td>26.882147</td>
          <td>0.200769</td>
          <td>26.997996</td>
          <td>0.348845</td>
          <td>26.117375</td>
          <td>0.308133</td>
          <td>25.052057</td>
          <td>0.275364</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.842489</td>
          <td>2.646349</td>
          <td>28.924162</td>
          <td>1.000910</td>
          <td>26.772807</td>
          <td>0.186406</td>
          <td>25.987146</td>
          <td>0.154186</td>
          <td>25.745345</td>
          <td>0.231484</td>
          <td>24.869267</td>
          <td>0.241349</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.473218</td>
          <td>0.157029</td>
          <td>25.928234</td>
          <td>0.087936</td>
          <td>25.693980</td>
          <td>0.117126</td>
          <td>24.961473</td>
          <td>0.116319</td>
          <td>25.092637</td>
          <td>0.283839</td>
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
          <td>27.676581</td>
          <td>0.868739</td>
          <td>26.625180</td>
          <td>0.154230</td>
          <td>26.142133</td>
          <td>0.089380</td>
          <td>25.466117</td>
          <td>0.080239</td>
          <td>24.936697</td>
          <td>0.095982</td>
          <td>24.745148</td>
          <td>0.179996</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.485798</td>
          <td>0.665518</td>
          <td>27.484272</td>
          <td>0.280485</td>
          <td>27.243913</td>
          <td>0.360103</td>
          <td>25.816149</td>
          <td>0.204713</td>
          <td>25.826055</td>
          <td>0.431459</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.891390</td>
          <td>0.087538</td>
          <td>24.784589</td>
          <td>0.029212</td>
          <td>23.872828</td>
          <td>0.021493</td>
          <td>23.092939</td>
          <td>0.020586</td>
          <td>22.865102</td>
          <td>0.037709</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.453858</td>
          <td>1.500819</td>
          <td>27.655433</td>
          <td>0.429676</td>
          <td>28.256092</td>
          <td>0.612866</td>
          <td>26.568022</td>
          <td>0.259081</td>
          <td>26.065251</td>
          <td>0.309520</td>
          <td>25.785854</td>
          <td>0.509600</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.474799</td>
          <td>0.370029</td>
          <td>25.876328</td>
          <td>0.080465</td>
          <td>25.420708</td>
          <td>0.047249</td>
          <td>24.830606</td>
          <td>0.045742</td>
          <td>24.369631</td>
          <td>0.058238</td>
          <td>23.699305</td>
          <td>0.072615</td>
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
          <td>27.039244</td>
          <td>0.590611</td>
          <td>26.222585</td>
          <td>0.116592</td>
          <td>25.968842</td>
          <td>0.083026</td>
          <td>25.997338</td>
          <td>0.138505</td>
          <td>26.695885</td>
          <td>0.444358</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.110269</td>
          <td>1.136853</td>
          <td>27.205339</td>
          <td>0.254411</td>
          <td>26.745916</td>
          <td>0.153559</td>
          <td>26.350139</td>
          <td>0.175823</td>
          <td>26.045776</td>
          <td>0.251349</td>
          <td>27.380144</td>
          <td>1.219880</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.507424</td>
          <td>0.333168</td>
          <td>27.020170</td>
          <td>0.199978</td>
          <td>26.176169</td>
          <td>0.156667</td>
          <td>25.395023</td>
          <td>0.149978</td>
          <td>25.409563</td>
          <td>0.326107</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.281173</td>
          <td>0.293153</td>
          <td>26.493936</td>
          <td>0.136112</td>
          <td>25.753222</td>
          <td>0.116314</td>
          <td>25.858125</td>
          <td>0.236079</td>
          <td>26.037602</td>
          <td>0.557172</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.568859</td>
          <td>0.827130</td>
          <td>26.673816</td>
          <td>0.166143</td>
          <td>26.032115</td>
          <td>0.084347</td>
          <td>25.459495</td>
          <td>0.083098</td>
          <td>25.180004</td>
          <td>0.123354</td>
          <td>24.694964</td>
          <td>0.179330</td>
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
