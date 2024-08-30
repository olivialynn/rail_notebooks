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

    <pzflow.flow.Flow at 0x7ff9a84f2c20>



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
          <td>27.084527</td>
          <td>0.582997</td>
          <td>27.478696</td>
          <td>0.313286</td>
          <td>26.004752</td>
          <td>0.079178</td>
          <td>25.356296</td>
          <td>0.072809</td>
          <td>25.090534</td>
          <td>0.109800</td>
          <td>24.756904</td>
          <td>0.181774</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.952475</td>
          <td>1.740496</td>
          <td>28.118453</td>
          <td>0.512107</td>
          <td>27.470898</td>
          <td>0.277215</td>
          <td>27.081332</td>
          <td>0.316364</td>
          <td>26.110708</td>
          <td>0.261052</td>
          <td>26.411074</td>
          <td>0.659258</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.121360</td>
          <td>0.279121</td>
          <td>25.992205</td>
          <td>0.088991</td>
          <td>24.834548</td>
          <td>0.028114</td>
          <td>23.859516</td>
          <td>0.019549</td>
          <td>23.136251</td>
          <td>0.019720</td>
          <td>22.850917</td>
          <td>0.034184</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.732255</td>
          <td>0.382540</td>
          <td>27.647875</td>
          <td>0.319650</td>
          <td>27.212624</td>
          <td>0.351055</td>
          <td>25.962463</td>
          <td>0.231057</td>
          <td>25.160854</td>
          <td>0.254590</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.993325</td>
          <td>0.251460</td>
          <td>25.865654</td>
          <td>0.079613</td>
          <td>25.542646</td>
          <td>0.052576</td>
          <td>24.902105</td>
          <td>0.048666</td>
          <td>24.436171</td>
          <td>0.061690</td>
          <td>23.690653</td>
          <td>0.071954</td>
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
          <td>26.322861</td>
          <td>0.328071</td>
          <td>26.279253</td>
          <td>0.114389</td>
          <td>26.089173</td>
          <td>0.085298</td>
          <td>25.985207</td>
          <td>0.126386</td>
          <td>26.036843</td>
          <td>0.245698</td>
          <td>25.299295</td>
          <td>0.284991</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.992895</td>
          <td>0.545896</td>
          <td>27.347624</td>
          <td>0.281924</td>
          <td>26.789858</td>
          <td>0.156908</td>
          <td>26.500474</td>
          <td>0.196352</td>
          <td>26.365848</td>
          <td>0.320785</td>
          <td>25.700830</td>
          <td>0.391646</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.895662</td>
          <td>0.994691</td>
          <td>27.139401</td>
          <td>0.237764</td>
          <td>26.727975</td>
          <td>0.148801</td>
          <td>26.160458</td>
          <td>0.147029</td>
          <td>26.097962</td>
          <td>0.258343</td>
          <td>25.150768</td>
          <td>0.252492</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.056224</td>
          <td>1.093796</td>
          <td>26.934778</td>
          <td>0.200515</td>
          <td>26.603708</td>
          <td>0.133693</td>
          <td>25.967005</td>
          <td>0.124407</td>
          <td>25.740900</td>
          <td>0.191992</td>
          <td>25.799049</td>
          <td>0.422321</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.635302</td>
          <td>0.418431</td>
          <td>26.593278</td>
          <td>0.150055</td>
          <td>26.079247</td>
          <td>0.084555</td>
          <td>25.783640</td>
          <td>0.106043</td>
          <td>25.403428</td>
          <td>0.144014</td>
          <td>24.905226</td>
          <td>0.205961</td>
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
          <td>26.701442</td>
          <td>0.486265</td>
          <td>26.565804</td>
          <td>0.168441</td>
          <td>26.264129</td>
          <td>0.116813</td>
          <td>25.413903</td>
          <td>0.090733</td>
          <td>24.955098</td>
          <td>0.114531</td>
          <td>24.738463</td>
          <td>0.210001</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.573078</td>
          <td>0.441776</td>
          <td>27.732830</td>
          <td>0.433618</td>
          <td>27.972606</td>
          <td>0.473565</td>
          <td>27.013491</td>
          <td>0.349034</td>
          <td>26.479066</td>
          <td>0.404770</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.826318</td>
          <td>0.540518</td>
          <td>25.948139</td>
          <td>0.100783</td>
          <td>24.736093</td>
          <td>0.031017</td>
          <td>23.904044</td>
          <td>0.024513</td>
          <td>23.143531</td>
          <td>0.023765</td>
          <td>22.827546</td>
          <td>0.040571</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.076315</td>
          <td>0.663273</td>
          <td>28.403503</td>
          <td>0.736088</td>
          <td>27.228919</td>
          <td>0.281334</td>
          <td>26.209875</td>
          <td>0.193071</td>
          <td>26.464359</td>
          <td>0.424229</td>
          <td>25.253923</td>
          <td>0.340609</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.018821</td>
          <td>0.611762</td>
          <td>25.799978</td>
          <td>0.086767</td>
          <td>25.511793</td>
          <td>0.060268</td>
          <td>24.757026</td>
          <td>0.050766</td>
          <td>24.459061</td>
          <td>0.074120</td>
          <td>23.796433</td>
          <td>0.093444</td>
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
          <td>26.260881</td>
          <td>0.352183</td>
          <td>26.480879</td>
          <td>0.159592</td>
          <td>26.228437</td>
          <td>0.115623</td>
          <td>25.863147</td>
          <td>0.137141</td>
          <td>25.847028</td>
          <td>0.249296</td>
          <td>24.752314</td>
          <td>0.216843</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>30.198279</td>
          <td>2.953114</td>
          <td>27.936586</td>
          <td>0.506474</td>
          <td>26.593913</td>
          <td>0.155930</td>
          <td>26.178683</td>
          <td>0.176639</td>
          <td>26.012249</td>
          <td>0.280876</td>
          <td>25.166938</td>
          <td>0.299713</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.070834</td>
          <td>0.639239</td>
          <td>27.787040</td>
          <td>0.456163</td>
          <td>26.633760</td>
          <td>0.162689</td>
          <td>26.625728</td>
          <td>0.258665</td>
          <td>26.340822</td>
          <td>0.367715</td>
          <td>25.242063</td>
          <td>0.320863</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.063147</td>
          <td>1.962386</td>
          <td>27.663028</td>
          <td>0.421357</td>
          <td>26.567630</td>
          <td>0.156562</td>
          <td>25.997486</td>
          <td>0.155557</td>
          <td>25.461166</td>
          <td>0.182455</td>
          <td>25.927033</td>
          <td>0.549593</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.240345</td>
          <td>1.311827</td>
          <td>26.559459</td>
          <td>0.169011</td>
          <td>26.061255</td>
          <td>0.098830</td>
          <td>25.614831</td>
          <td>0.109320</td>
          <td>25.297672</td>
          <td>0.155498</td>
          <td>24.949513</td>
          <td>0.252578</td>
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
          <td>28.201202</td>
          <td>1.188084</td>
          <td>26.614730</td>
          <td>0.152856</td>
          <td>26.018256</td>
          <td>0.080138</td>
          <td>25.275873</td>
          <td>0.067815</td>
          <td>25.051936</td>
          <td>0.106174</td>
          <td>24.775562</td>
          <td>0.184690</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.732614</td>
          <td>0.785573</td>
          <td>27.369047</td>
          <td>0.255330</td>
          <td>26.760089</td>
          <td>0.243980</td>
          <td>27.711835</td>
          <td>0.850146</td>
          <td>27.001717</td>
          <td>0.968665</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>29.495357</td>
          <td>2.251664</td>
          <td>25.937260</td>
          <td>0.091136</td>
          <td>24.754178</td>
          <td>0.028445</td>
          <td>23.882423</td>
          <td>0.021670</td>
          <td>23.151658</td>
          <td>0.021642</td>
          <td>22.790132</td>
          <td>0.035292</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.797235</td>
          <td>1.630163</td>
          <td>27.123282</td>
          <td>0.257277</td>
          <td>26.261148</td>
          <td>0.200868</td>
          <td>26.516561</td>
          <td>0.440025</td>
          <td>24.966623</td>
          <td>0.269563</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.462904</td>
          <td>0.756862</td>
          <td>25.780154</td>
          <td>0.073922</td>
          <td>25.404060</td>
          <td>0.046556</td>
          <td>24.794131</td>
          <td>0.044285</td>
          <td>24.360020</td>
          <td>0.057743</td>
          <td>23.742692</td>
          <td>0.075455</td>
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
          <td>26.138227</td>
          <td>0.297849</td>
          <td>26.640606</td>
          <td>0.167094</td>
          <td>26.082448</td>
          <td>0.091757</td>
          <td>25.996169</td>
          <td>0.138365</td>
          <td>25.974592</td>
          <td>0.251257</td>
          <td>25.053504</td>
          <td>0.251569</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.648847</td>
          <td>0.427000</td>
          <td>26.898217</td>
          <td>0.197132</td>
          <td>26.623408</td>
          <td>0.138206</td>
          <td>26.428754</td>
          <td>0.187924</td>
          <td>26.382295</td>
          <td>0.329882</td>
          <td>26.080353</td>
          <td>0.528566</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.172541</td>
          <td>0.254310</td>
          <td>26.567005</td>
          <td>0.135903</td>
          <td>26.805208</td>
          <td>0.265330</td>
          <td>26.109758</td>
          <td>0.272949</td>
          <td>25.590867</td>
          <td>0.376094</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.073963</td>
          <td>1.915398</td>
          <td>27.505077</td>
          <td>0.350381</td>
          <td>26.360254</td>
          <td>0.121233</td>
          <td>25.755215</td>
          <td>0.116516</td>
          <td>25.908191</td>
          <td>0.246034</td>
          <td>25.616079</td>
          <td>0.407091</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.784536</td>
          <td>0.479312</td>
          <td>26.419461</td>
          <td>0.133574</td>
          <td>26.188682</td>
          <td>0.096795</td>
          <td>25.646812</td>
          <td>0.097977</td>
          <td>25.238068</td>
          <td>0.129721</td>
          <td>24.813140</td>
          <td>0.198141</td>
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
