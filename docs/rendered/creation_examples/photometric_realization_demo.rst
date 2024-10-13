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

    <pzflow.flow.Flow at 0x7fc8e2825e40>



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
          <td>28.419540</td>
          <td>1.337983</td>
          <td>26.821138</td>
          <td>0.182204</td>
          <td>26.099905</td>
          <td>0.086108</td>
          <td>25.340007</td>
          <td>0.071767</td>
          <td>25.033690</td>
          <td>0.104480</td>
          <td>24.734412</td>
          <td>0.178343</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.236347</td>
          <td>0.648610</td>
          <td>27.846594</td>
          <td>0.417739</td>
          <td>27.750783</td>
          <td>0.346816</td>
          <td>26.977516</td>
          <td>0.291064</td>
          <td>26.841678</td>
          <td>0.463611</td>
          <td>26.343298</td>
          <td>0.628944</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.594466</td>
          <td>0.405556</td>
          <td>25.824764</td>
          <td>0.076794</td>
          <td>24.804428</td>
          <td>0.027384</td>
          <td>23.871031</td>
          <td>0.019740</td>
          <td>23.122776</td>
          <td>0.019497</td>
          <td>22.755295</td>
          <td>0.031423</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.391236</td>
          <td>0.720960</td>
          <td>28.178958</td>
          <td>0.535244</td>
          <td>27.731039</td>
          <td>0.341456</td>
          <td>26.639202</td>
          <td>0.220527</td>
          <td>25.956524</td>
          <td>0.229922</td>
          <td>25.073392</td>
          <td>0.236901</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.322564</td>
          <td>0.327994</td>
          <td>25.815566</td>
          <td>0.076173</td>
          <td>25.414497</td>
          <td>0.046921</td>
          <td>24.871010</td>
          <td>0.047341</td>
          <td>24.376192</td>
          <td>0.058494</td>
          <td>23.688288</td>
          <td>0.071804</td>
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
          <td>26.511032</td>
          <td>0.380270</td>
          <td>26.304338</td>
          <td>0.116912</td>
          <td>26.139857</td>
          <td>0.089190</td>
          <td>26.125182</td>
          <td>0.142635</td>
          <td>25.612338</td>
          <td>0.172194</td>
          <td>25.115317</td>
          <td>0.245238</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.396839</td>
          <td>0.723680</td>
          <td>27.096470</td>
          <td>0.229468</td>
          <td>26.753943</td>
          <td>0.152154</td>
          <td>26.401453</td>
          <td>0.180605</td>
          <td>25.798341</td>
          <td>0.201496</td>
          <td>25.299345</td>
          <td>0.285003</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.708741</td>
          <td>0.886501</td>
          <td>27.513958</td>
          <td>0.322223</td>
          <td>26.856839</td>
          <td>0.166146</td>
          <td>26.538971</td>
          <td>0.202806</td>
          <td>26.118284</td>
          <td>0.262673</td>
          <td>26.004359</td>
          <td>0.492777</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.574513</td>
          <td>0.399386</td>
          <td>26.866139</td>
          <td>0.189263</td>
          <td>26.814645</td>
          <td>0.160269</td>
          <td>25.839520</td>
          <td>0.111346</td>
          <td>25.606256</td>
          <td>0.171305</td>
          <td>25.164135</td>
          <td>0.255276</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.581064</td>
          <td>0.148491</td>
          <td>26.072286</td>
          <td>0.084038</td>
          <td>25.563729</td>
          <td>0.087436</td>
          <td>25.098971</td>
          <td>0.110611</td>
          <td>25.441988</td>
          <td>0.319608</td>
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
          <td>26.937344</td>
          <td>0.577332</td>
          <td>26.246350</td>
          <td>0.128049</td>
          <td>26.142523</td>
          <td>0.105059</td>
          <td>25.247678</td>
          <td>0.078375</td>
          <td>25.006390</td>
          <td>0.119757</td>
          <td>24.970197</td>
          <td>0.254439</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.503241</td>
          <td>1.496211</td>
          <td>27.599358</td>
          <td>0.391492</td>
          <td>27.314999</td>
          <td>0.283606</td>
          <td>27.909754</td>
          <td>0.676859</td>
          <td>26.707851</td>
          <td>0.481221</td>
          <td>26.352232</td>
          <td>0.720937</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.450687</td>
          <td>0.408424</td>
          <td>25.846620</td>
          <td>0.092210</td>
          <td>24.772308</td>
          <td>0.032020</td>
          <td>23.931795</td>
          <td>0.025109</td>
          <td>23.136315</td>
          <td>0.023617</td>
          <td>22.762483</td>
          <td>0.038302</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.602820</td>
          <td>0.472133</td>
          <td>28.010102</td>
          <td>0.560162</td>
          <td>27.753321</td>
          <td>0.425112</td>
          <td>27.083846</td>
          <td>0.392045</td>
          <td>26.672632</td>
          <td>0.496015</td>
          <td>25.575694</td>
          <td>0.436971</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.704604</td>
          <td>0.221324</td>
          <td>26.002912</td>
          <td>0.103651</td>
          <td>25.407900</td>
          <td>0.054962</td>
          <td>24.828563</td>
          <td>0.054093</td>
          <td>24.385594</td>
          <td>0.069458</td>
          <td>23.737443</td>
          <td>0.088724</td>
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
          <td>26.518529</td>
          <td>0.429734</td>
          <td>26.579555</td>
          <td>0.173578</td>
          <td>26.119333</td>
          <td>0.105126</td>
          <td>25.827192</td>
          <td>0.132949</td>
          <td>25.866111</td>
          <td>0.253234</td>
          <td>26.221656</td>
          <td>0.670676</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.826584</td>
          <td>0.534461</td>
          <td>26.782810</td>
          <td>0.203057</td>
          <td>26.974167</td>
          <td>0.215071</td>
          <td>26.089448</td>
          <td>0.163726</td>
          <td>26.313850</td>
          <td>0.357288</td>
          <td>25.329179</td>
          <td>0.341086</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.081691</td>
          <td>0.261983</td>
          <td>27.870317</td>
          <td>0.443287</td>
          <td>26.561558</td>
          <td>0.245391</td>
          <td>25.847518</td>
          <td>0.247490</td>
          <td>25.583744</td>
          <td>0.418951</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.224559</td>
          <td>0.718108</td>
          <td>29.580410</td>
          <td>1.441026</td>
          <td>26.501025</td>
          <td>0.147873</td>
          <td>25.802356</td>
          <td>0.131511</td>
          <td>25.546744</td>
          <td>0.196115</td>
          <td>26.382265</td>
          <td>0.753694</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.056977</td>
          <td>1.186593</td>
          <td>26.425047</td>
          <td>0.150686</td>
          <td>25.991325</td>
          <td>0.092950</td>
          <td>25.614643</td>
          <td>0.109302</td>
          <td>25.126401</td>
          <td>0.134201</td>
          <td>24.782260</td>
          <td>0.219961</td>
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
          <td>26.804677</td>
          <td>0.179703</td>
          <td>25.860191</td>
          <td>0.069688</td>
          <td>25.422135</td>
          <td>0.077182</td>
          <td>24.977727</td>
          <td>0.099498</td>
          <td>24.789960</td>
          <td>0.186951</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.034142</td>
          <td>1.080305</td>
          <td>28.619746</td>
          <td>0.728934</td>
          <td>27.648587</td>
          <td>0.320106</td>
          <td>26.645404</td>
          <td>0.221877</td>
          <td>26.753051</td>
          <td>0.433991</td>
          <td>25.218839</td>
          <td>0.267199</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.272855</td>
          <td>0.332426</td>
          <td>25.977080</td>
          <td>0.094375</td>
          <td>24.755544</td>
          <td>0.028479</td>
          <td>23.847837</td>
          <td>0.021040</td>
          <td>23.122496</td>
          <td>0.021110</td>
          <td>22.838650</td>
          <td>0.036838</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.737900</td>
          <td>0.520426</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.908483</td>
          <td>0.476376</td>
          <td>27.041558</td>
          <td>0.378177</td>
          <td>26.430030</td>
          <td>0.411968</td>
          <td>26.208553</td>
          <td>0.687662</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.297697</td>
          <td>0.321864</td>
          <td>25.813371</td>
          <td>0.076120</td>
          <td>25.375769</td>
          <td>0.045401</td>
          <td>24.882061</td>
          <td>0.047880</td>
          <td>24.321586</td>
          <td>0.055807</td>
          <td>23.689726</td>
          <td>0.072003</td>
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
          <td>27.184710</td>
          <td>0.653971</td>
          <td>26.406937</td>
          <td>0.136772</td>
          <td>26.168491</td>
          <td>0.098954</td>
          <td>26.301349</td>
          <td>0.179631</td>
          <td>25.891610</td>
          <td>0.234645</td>
          <td>25.378557</td>
          <td>0.327163</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.994022</td>
          <td>1.062865</td>
          <td>26.912441</td>
          <td>0.199502</td>
          <td>27.050270</td>
          <td>0.198837</td>
          <td>26.007646</td>
          <td>0.131081</td>
          <td>26.838388</td>
          <td>0.468993</td>
          <td>25.652542</td>
          <td>0.383001</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.041350</td>
          <td>1.109273</td>
          <td>27.072319</td>
          <td>0.234163</td>
          <td>26.891729</td>
          <td>0.179439</td>
          <td>26.318407</td>
          <td>0.176857</td>
          <td>26.093164</td>
          <td>0.269286</td>
          <td>25.637342</td>
          <td>0.389899</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.748273</td>
          <td>0.488731</td>
          <td>27.053866</td>
          <td>0.243574</td>
          <td>26.635150</td>
          <td>0.153690</td>
          <td>26.013007</td>
          <td>0.145632</td>
          <td>25.497146</td>
          <td>0.174428</td>
          <td>26.302313</td>
          <td>0.671249</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.886340</td>
          <td>0.516705</td>
          <td>26.437934</td>
          <td>0.135721</td>
          <td>26.195466</td>
          <td>0.097373</td>
          <td>25.546894</td>
          <td>0.089746</td>
          <td>25.448194</td>
          <td>0.155448</td>
          <td>24.868976</td>
          <td>0.207642</td>
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
