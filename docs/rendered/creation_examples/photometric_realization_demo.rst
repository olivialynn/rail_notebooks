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

    <pzflow.flow.Flow at 0x7f85cfa4aad0>



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
          <td>27.818133</td>
          <td>0.948866</td>
          <td>26.910119</td>
          <td>0.196404</td>
          <td>26.109934</td>
          <td>0.086872</td>
          <td>25.365182</td>
          <td>0.073383</td>
          <td>25.004905</td>
          <td>0.101882</td>
          <td>25.146890</td>
          <td>0.251689</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.156782</td>
          <td>1.906629</td>
          <td>27.980830</td>
          <td>0.462408</td>
          <td>27.913783</td>
          <td>0.393841</td>
          <td>26.886120</td>
          <td>0.270271</td>
          <td>26.410786</td>
          <td>0.332450</td>
          <td>26.496081</td>
          <td>0.698764</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.146578</td>
          <td>0.609185</td>
          <td>25.928184</td>
          <td>0.084120</td>
          <td>24.764560</td>
          <td>0.026448</td>
          <td>23.840704</td>
          <td>0.019241</td>
          <td>23.119111</td>
          <td>0.019437</td>
          <td>22.817101</td>
          <td>0.033180</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.699756</td>
          <td>0.372998</td>
          <td>27.122870</td>
          <td>0.208024</td>
          <td>26.446896</td>
          <td>0.187682</td>
          <td>26.060588</td>
          <td>0.250544</td>
          <td>25.302220</td>
          <td>0.285666</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.325740</td>
          <td>0.328821</td>
          <td>25.689259</td>
          <td>0.068134</td>
          <td>25.364623</td>
          <td>0.044889</td>
          <td>24.740438</td>
          <td>0.042161</td>
          <td>24.403837</td>
          <td>0.059946</td>
          <td>23.643610</td>
          <td>0.069021</td>
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
          <td>27.132398</td>
          <td>0.603124</td>
          <td>26.261228</td>
          <td>0.112608</td>
          <td>26.086804</td>
          <td>0.085120</td>
          <td>26.095666</td>
          <td>0.139053</td>
          <td>26.035451</td>
          <td>0.245417</td>
          <td>25.121991</td>
          <td>0.246589</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.166366</td>
          <td>0.617719</td>
          <td>27.536083</td>
          <td>0.327941</td>
          <td>26.994843</td>
          <td>0.186790</td>
          <td>26.676679</td>
          <td>0.227505</td>
          <td>27.021509</td>
          <td>0.529512</td>
          <td>25.344582</td>
          <td>0.295606</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.716840</td>
          <td>0.891026</td>
          <td>27.169057</td>
          <td>0.243652</td>
          <td>26.964942</td>
          <td>0.182126</td>
          <td>26.263988</td>
          <td>0.160670</td>
          <td>26.400042</td>
          <td>0.329628</td>
          <td>25.554719</td>
          <td>0.349462</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.463746</td>
          <td>0.309562</td>
          <td>26.518249</td>
          <td>0.124156</td>
          <td>26.044640</td>
          <td>0.133059</td>
          <td>25.667919</td>
          <td>0.180510</td>
          <td>25.206326</td>
          <td>0.264243</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.678773</td>
          <td>0.432503</td>
          <td>26.442344</td>
          <td>0.131769</td>
          <td>26.115315</td>
          <td>0.087284</td>
          <td>25.464381</td>
          <td>0.080105</td>
          <td>25.269922</td>
          <td>0.128335</td>
          <td>24.640619</td>
          <td>0.164670</td>
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
          <td>26.746378</td>
          <td>0.196235</td>
          <td>25.974277</td>
          <td>0.090653</td>
          <td>25.250983</td>
          <td>0.078604</td>
          <td>25.011926</td>
          <td>0.120334</td>
          <td>24.927785</td>
          <td>0.245725</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.970970</td>
          <td>1.863582</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.078230</td>
          <td>0.233619</td>
          <td>27.677897</td>
          <td>0.575448</td>
          <td>26.663283</td>
          <td>0.465481</td>
          <td>27.306787</td>
          <td>1.286436</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.682951</td>
          <td>0.486596</td>
          <td>25.996313</td>
          <td>0.105115</td>
          <td>24.790733</td>
          <td>0.032543</td>
          <td>23.835039</td>
          <td>0.023097</td>
          <td>23.160589</td>
          <td>0.024117</td>
          <td>22.893591</td>
          <td>0.043015</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.193119</td>
          <td>2.838532</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.766643</td>
          <td>0.305364</td>
          <td>26.316460</td>
          <td>0.378590</td>
          <td>25.413138</td>
          <td>0.385792</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.171029</td>
          <td>0.323432</td>
          <td>25.794746</td>
          <td>0.086369</td>
          <td>25.375668</td>
          <td>0.053413</td>
          <td>24.868066</td>
          <td>0.056022</td>
          <td>24.317202</td>
          <td>0.065378</td>
          <td>23.793705</td>
          <td>0.093220</td>
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
          <td>26.749437</td>
          <td>0.510610</td>
          <td>26.560859</td>
          <td>0.170844</td>
          <td>26.059306</td>
          <td>0.099747</td>
          <td>25.930175</td>
          <td>0.145292</td>
          <td>25.685134</td>
          <td>0.218033</td>
          <td>26.050180</td>
          <td>0.595028</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.517985</td>
          <td>0.424805</td>
          <td>27.628614</td>
          <td>0.401679</td>
          <td>26.600495</td>
          <td>0.156810</td>
          <td>26.405079</td>
          <td>0.213725</td>
          <td>26.325704</td>
          <td>0.360623</td>
          <td>26.893584</td>
          <td>1.019890</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.420367</td>
          <td>0.343889</td>
          <td>26.861303</td>
          <td>0.197284</td>
          <td>26.137341</td>
          <td>0.172021</td>
          <td>25.755840</td>
          <td>0.229443</td>
          <td>24.991352</td>
          <td>0.262072</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.327222</td>
          <td>1.386859</td>
          <td>26.715794</td>
          <td>0.196525</td>
          <td>26.820166</td>
          <td>0.194003</td>
          <td>25.691607</td>
          <td>0.119471</td>
          <td>25.464656</td>
          <td>0.182994</td>
          <td>25.291955</td>
          <td>0.339634</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.505884</td>
          <td>0.161471</td>
          <td>26.269790</td>
          <td>0.118563</td>
          <td>25.815524</td>
          <td>0.130153</td>
          <td>25.167301</td>
          <td>0.139022</td>
          <td>25.175429</td>
          <td>0.303432</td>
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
          <td>29.545610</td>
          <td>2.237538</td>
          <td>26.463559</td>
          <td>0.134221</td>
          <td>25.980677</td>
          <td>0.077523</td>
          <td>25.302607</td>
          <td>0.069440</td>
          <td>24.946832</td>
          <td>0.096839</td>
          <td>25.359123</td>
          <td>0.299124</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.247567</td>
          <td>1.982880</td>
          <td>27.943654</td>
          <td>0.449983</td>
          <td>27.276672</td>
          <td>0.236631</td>
          <td>26.861451</td>
          <td>0.265133</td>
          <td>26.722585</td>
          <td>0.424051</td>
          <td>25.645898</td>
          <td>0.375639</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.124534</td>
          <td>0.295311</td>
          <td>26.141460</td>
          <td>0.108963</td>
          <td>24.725142</td>
          <td>0.027732</td>
          <td>23.927714</td>
          <td>0.022528</td>
          <td>23.172712</td>
          <td>0.022035</td>
          <td>22.852932</td>
          <td>0.037306</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.537318</td>
          <td>0.896430</td>
          <td>27.643538</td>
          <td>0.425806</td>
          <td>27.109740</td>
          <td>0.254437</td>
          <td>26.869010</td>
          <td>0.330244</td>
          <td>26.213839</td>
          <td>0.348280</td>
          <td>25.678020</td>
          <td>0.470469</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.376268</td>
          <td>0.342528</td>
          <td>25.757557</td>
          <td>0.072461</td>
          <td>25.433873</td>
          <td>0.047804</td>
          <td>24.851547</td>
          <td>0.046600</td>
          <td>24.430217</td>
          <td>0.061453</td>
          <td>23.654190</td>
          <td>0.069774</td>
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
          <td>26.271458</td>
          <td>0.331258</td>
          <td>26.217651</td>
          <td>0.116093</td>
          <td>25.941158</td>
          <td>0.081024</td>
          <td>25.981215</td>
          <td>0.136592</td>
          <td>25.676888</td>
          <td>0.196154</td>
          <td>25.586664</td>
          <td>0.385216</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.039396</td>
          <td>0.488940</td>
          <td>26.994822</td>
          <td>0.189767</td>
          <td>26.389558</td>
          <td>0.181798</td>
          <td>26.659034</td>
          <td>0.409425</td>
          <td>25.150881</td>
          <td>0.256579</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.192542</td>
          <td>0.258510</td>
          <td>26.869827</td>
          <td>0.176137</td>
          <td>26.304893</td>
          <td>0.174840</td>
          <td>25.712780</td>
          <td>0.196487</td>
          <td>25.618862</td>
          <td>0.384360</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.709255</td>
          <td>0.410577</td>
          <td>26.832000</td>
          <td>0.181747</td>
          <td>25.791583</td>
          <td>0.120260</td>
          <td>25.673473</td>
          <td>0.202423</td>
          <td>25.773875</td>
          <td>0.458908</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.299507</td>
          <td>0.691943</td>
          <td>26.531389</td>
          <td>0.147089</td>
          <td>26.049866</td>
          <td>0.085676</td>
          <td>25.551640</td>
          <td>0.090121</td>
          <td>25.065587</td>
          <td>0.111666</td>
          <td>24.808289</td>
          <td>0.197334</td>
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
