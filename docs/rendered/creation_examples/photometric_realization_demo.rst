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

    <pzflow.flow.Flow at 0x7f3ae6485ea0>



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
          <td>26.734328</td>
          <td>0.451047</td>
          <td>26.997786</td>
          <td>0.211378</td>
          <td>26.246414</td>
          <td>0.097940</td>
          <td>25.293053</td>
          <td>0.068845</td>
          <td>24.852605</td>
          <td>0.089135</td>
          <td>25.283784</td>
          <td>0.281433</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.795381</td>
          <td>0.178274</td>
          <td>27.952287</td>
          <td>0.405694</td>
          <td>27.474023</td>
          <td>0.429735</td>
          <td>27.279463</td>
          <td>0.636414</td>
          <td>25.986798</td>
          <td>0.486406</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.247156</td>
          <td>0.308870</td>
          <td>25.986455</td>
          <td>0.088542</td>
          <td>24.776745</td>
          <td>0.026730</td>
          <td>23.894793</td>
          <td>0.020142</td>
          <td>23.166729</td>
          <td>0.020236</td>
          <td>22.783282</td>
          <td>0.032206</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.585604</td>
          <td>1.458156</td>
          <td>28.849836</td>
          <td>0.846993</td>
          <td>26.984660</td>
          <td>0.185190</td>
          <td>26.752803</td>
          <td>0.242294</td>
          <td>26.117923</td>
          <td>0.262596</td>
          <td>25.715120</td>
          <td>0.395990</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.369086</td>
          <td>0.340292</td>
          <td>25.684878</td>
          <td>0.067870</td>
          <td>25.464216</td>
          <td>0.049039</td>
          <td>24.761562</td>
          <td>0.042958</td>
          <td>24.475474</td>
          <td>0.063878</td>
          <td>23.781932</td>
          <td>0.078000</td>
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
          <td>26.688292</td>
          <td>0.435636</td>
          <td>26.183035</td>
          <td>0.105185</td>
          <td>26.109520</td>
          <td>0.086840</td>
          <td>26.229699</td>
          <td>0.156027</td>
          <td>25.894473</td>
          <td>0.218365</td>
          <td>25.468664</td>
          <td>0.326468</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.717028</td>
          <td>1.556753</td>
          <td>27.329847</td>
          <td>0.277890</td>
          <td>26.613140</td>
          <td>0.134787</td>
          <td>26.353791</td>
          <td>0.173447</td>
          <td>26.324478</td>
          <td>0.310359</td>
          <td>25.439405</td>
          <td>0.318950</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>30.549807</td>
          <td>3.154767</td>
          <td>27.403156</td>
          <td>0.294860</td>
          <td>26.861445</td>
          <td>0.166800</td>
          <td>26.628180</td>
          <td>0.218512</td>
          <td>25.832573</td>
          <td>0.207362</td>
          <td>25.826061</td>
          <td>0.431094</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.878609</td>
          <td>0.984497</td>
          <td>26.744136</td>
          <td>0.170686</td>
          <td>26.346888</td>
          <td>0.106944</td>
          <td>25.858029</td>
          <td>0.113157</td>
          <td>25.358536</td>
          <td>0.138551</td>
          <td>25.769723</td>
          <td>0.412961</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.536456</td>
          <td>0.793802</td>
          <td>26.248127</td>
          <td>0.111330</td>
          <td>26.203268</td>
          <td>0.094302</td>
          <td>25.632772</td>
          <td>0.092909</td>
          <td>25.439734</td>
          <td>0.148579</td>
          <td>24.837211</td>
          <td>0.194526</td>
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
          <td>27.047947</td>
          <td>0.624269</td>
          <td>26.623022</td>
          <td>0.176828</td>
          <td>26.047639</td>
          <td>0.096683</td>
          <td>25.362063</td>
          <td>0.086688</td>
          <td>24.957806</td>
          <td>0.114801</td>
          <td>25.410931</td>
          <td>0.362336</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.669572</td>
          <td>0.837518</td>
          <td>27.311330</td>
          <td>0.282765</td>
          <td>27.899150</td>
          <td>0.671955</td>
          <td>28.195406</td>
          <td>1.260923</td>
          <td>26.555706</td>
          <td>0.824403</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.016706</td>
          <td>0.290262</td>
          <td>25.898015</td>
          <td>0.096459</td>
          <td>24.816343</td>
          <td>0.033285</td>
          <td>23.865134</td>
          <td>0.023703</td>
          <td>23.136206</td>
          <td>0.023615</td>
          <td>22.842287</td>
          <td>0.041104</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.935121</td>
          <td>1.030014</td>
          <td>27.607769</td>
          <td>0.380085</td>
          <td>26.753199</td>
          <td>0.302086</td>
          <td>25.707479</td>
          <td>0.232014</td>
          <td>26.132592</td>
          <td>0.654575</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.885112</td>
          <td>0.556229</td>
          <td>25.680917</td>
          <td>0.078136</td>
          <td>25.416322</td>
          <td>0.055374</td>
          <td>24.805176</td>
          <td>0.052982</td>
          <td>24.295786</td>
          <td>0.064150</td>
          <td>23.660893</td>
          <td>0.082942</td>
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
          <td>26.003043</td>
          <td>0.286809</td>
          <td>26.323959</td>
          <td>0.139498</td>
          <td>26.131316</td>
          <td>0.106233</td>
          <td>25.981758</td>
          <td>0.151871</td>
          <td>26.076917</td>
          <td>0.300533</td>
          <td>25.486934</td>
          <td>0.391801</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.470472</td>
          <td>0.409672</td>
          <td>27.523436</td>
          <td>0.370258</td>
          <td>26.838424</td>
          <td>0.191930</td>
          <td>26.262189</td>
          <td>0.189570</td>
          <td>27.829195</td>
          <td>1.025939</td>
          <td>25.449789</td>
          <td>0.374917</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.468623</td>
          <td>0.834219</td>
          <td>27.915272</td>
          <td>0.501845</td>
          <td>26.987193</td>
          <td>0.219203</td>
          <td>26.120470</td>
          <td>0.169571</td>
          <td>25.755510</td>
          <td>0.229380</td>
          <td>25.196280</td>
          <td>0.309342</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.087622</td>
          <td>0.267409</td>
          <td>26.583532</td>
          <td>0.158706</td>
          <td>25.809114</td>
          <td>0.132282</td>
          <td>25.779784</td>
          <td>0.238173</td>
          <td>25.516642</td>
          <td>0.404647</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.504228</td>
          <td>0.422010</td>
          <td>26.427309</td>
          <td>0.150978</td>
          <td>26.076788</td>
          <td>0.100184</td>
          <td>25.489377</td>
          <td>0.097959</td>
          <td>25.286156</td>
          <td>0.153972</td>
          <td>24.572315</td>
          <td>0.184431</td>
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
          <td>26.773607</td>
          <td>0.464577</td>
          <td>26.567121</td>
          <td>0.146740</td>
          <td>26.087329</td>
          <td>0.085170</td>
          <td>25.329233</td>
          <td>0.071096</td>
          <td>25.247124</td>
          <td>0.125841</td>
          <td>24.989581</td>
          <td>0.221020</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.984671</td>
          <td>0.464065</td>
          <td>27.956399</td>
          <td>0.407314</td>
          <td>27.212884</td>
          <td>0.351440</td>
          <td>26.402725</td>
          <td>0.330613</td>
          <td>26.360128</td>
          <td>0.636870</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.445627</td>
          <td>0.380631</td>
          <td>25.932992</td>
          <td>0.090795</td>
          <td>24.759272</td>
          <td>0.028572</td>
          <td>23.873836</td>
          <td>0.021512</td>
          <td>23.177223</td>
          <td>0.022121</td>
          <td>22.816136</td>
          <td>0.036112</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.461811</td>
          <td>0.423516</td>
          <td>28.460307</td>
          <td>0.762545</td>
          <td>27.340914</td>
          <td>0.306907</td>
          <td>26.673472</td>
          <td>0.282312</td>
          <td>26.129281</td>
          <td>0.325742</td>
          <td>25.007920</td>
          <td>0.278768</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.915737</td>
          <td>0.236130</td>
          <td>25.868188</td>
          <td>0.079890</td>
          <td>25.361895</td>
          <td>0.044845</td>
          <td>24.888023</td>
          <td>0.048134</td>
          <td>24.294438</td>
          <td>0.054478</td>
          <td>23.651990</td>
          <td>0.069638</td>
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
          <td>28.060182</td>
          <td>1.137089</td>
          <td>26.183810</td>
          <td>0.112725</td>
          <td>25.976512</td>
          <td>0.083589</td>
          <td>25.838250</td>
          <td>0.120684</td>
          <td>25.971012</td>
          <td>0.250519</td>
          <td>25.073532</td>
          <td>0.255737</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.355192</td>
          <td>0.340071</td>
          <td>26.757844</td>
          <td>0.175093</td>
          <td>26.765980</td>
          <td>0.156221</td>
          <td>26.228756</td>
          <td>0.158547</td>
          <td>26.146630</td>
          <td>0.272951</td>
          <td>25.063418</td>
          <td>0.238766</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.106230</td>
          <td>0.240814</td>
          <td>26.464720</td>
          <td>0.124389</td>
          <td>26.930765</td>
          <td>0.293785</td>
          <td>25.648081</td>
          <td>0.186056</td>
          <td>25.623308</td>
          <td>0.385687</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.915433</td>
          <td>0.217182</td>
          <td>27.209817</td>
          <td>0.249145</td>
          <td>26.065486</td>
          <td>0.152343</td>
          <td>25.666050</td>
          <td>0.201166</td>
          <td>25.720983</td>
          <td>0.440974</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.792708</td>
          <td>0.482232</td>
          <td>26.348510</td>
          <td>0.125624</td>
          <td>26.104375</td>
          <td>0.089887</td>
          <td>25.652172</td>
          <td>0.098438</td>
          <td>25.334765</td>
          <td>0.141018</td>
          <td>25.148037</td>
          <td>0.261591</td>
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
