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

    <pzflow.flow.Flow at 0x7f5ea4cd1660>



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
          <td>26.912556</td>
          <td>0.514889</td>
          <td>27.066157</td>
          <td>0.223767</td>
          <td>26.040164</td>
          <td>0.081691</td>
          <td>25.442512</td>
          <td>0.078573</td>
          <td>24.924738</td>
          <td>0.094967</td>
          <td>24.687311</td>
          <td>0.171350</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.104020</td>
          <td>0.506705</td>
          <td>27.829431</td>
          <td>0.368879</td>
          <td>27.605346</td>
          <td>0.474410</td>
          <td>26.960178</td>
          <td>0.506258</td>
          <td>25.029124</td>
          <td>0.228374</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.097011</td>
          <td>0.588196</td>
          <td>25.857214</td>
          <td>0.079023</td>
          <td>24.781038</td>
          <td>0.026830</td>
          <td>23.855031</td>
          <td>0.019475</td>
          <td>23.151507</td>
          <td>0.019976</td>
          <td>22.812362</td>
          <td>0.033041</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.812180</td>
          <td>1.629953</td>
          <td>27.540795</td>
          <td>0.329170</td>
          <td>27.522170</td>
          <td>0.288971</td>
          <td>26.615938</td>
          <td>0.216293</td>
          <td>26.155820</td>
          <td>0.270842</td>
          <td>26.338276</td>
          <td>0.626739</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.547484</td>
          <td>0.391151</td>
          <td>25.740306</td>
          <td>0.071277</td>
          <td>25.412961</td>
          <td>0.046857</td>
          <td>24.780007</td>
          <td>0.043667</td>
          <td>24.496861</td>
          <td>0.065100</td>
          <td>23.758061</td>
          <td>0.076372</td>
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
          <td>26.940475</td>
          <td>0.525505</td>
          <td>26.398695</td>
          <td>0.126887</td>
          <td>26.206257</td>
          <td>0.094549</td>
          <td>26.115508</td>
          <td>0.141451</td>
          <td>27.160090</td>
          <td>0.585094</td>
          <td>25.630054</td>
          <td>0.370708</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.293067</td>
          <td>0.674467</td>
          <td>26.819572</td>
          <td>0.181963</td>
          <td>26.952382</td>
          <td>0.180199</td>
          <td>26.439547</td>
          <td>0.186520</td>
          <td>26.637686</td>
          <td>0.397005</td>
          <td>25.751918</td>
          <td>0.407363</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.378175</td>
          <td>0.714646</td>
          <td>27.900737</td>
          <td>0.435313</td>
          <td>26.958850</td>
          <td>0.181189</td>
          <td>26.898923</td>
          <td>0.273103</td>
          <td>25.651479</td>
          <td>0.178012</td>
          <td>24.924471</td>
          <td>0.209306</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.699119</td>
          <td>0.881146</td>
          <td>27.034174</td>
          <td>0.217892</td>
          <td>26.659873</td>
          <td>0.140333</td>
          <td>25.722391</td>
          <td>0.100509</td>
          <td>25.723838</td>
          <td>0.189249</td>
          <td>25.437173</td>
          <td>0.318383</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.375337</td>
          <td>1.306865</td>
          <td>26.804800</td>
          <td>0.179702</td>
          <td>26.020616</td>
          <td>0.080294</td>
          <td>25.584015</td>
          <td>0.089011</td>
          <td>24.991901</td>
          <td>0.100728</td>
          <td>25.312109</td>
          <td>0.287960</td>
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
          <td>27.091157</td>
          <td>0.643358</td>
          <td>26.544947</td>
          <td>0.165477</td>
          <td>26.035591</td>
          <td>0.095667</td>
          <td>25.308115</td>
          <td>0.082666</td>
          <td>25.217737</td>
          <td>0.143773</td>
          <td>25.258068</td>
          <td>0.321135</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.015107</td>
          <td>0.610121</td>
          <td>27.560977</td>
          <td>0.380026</td>
          <td>27.145581</td>
          <td>0.246972</td>
          <td>26.533092</td>
          <td>0.236784</td>
          <td>25.738164</td>
          <td>0.223432</td>
          <td>25.983045</td>
          <td>0.557403</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.554783</td>
          <td>0.442090</td>
          <td>26.122336</td>
          <td>0.117309</td>
          <td>24.789232</td>
          <td>0.032500</td>
          <td>23.845311</td>
          <td>0.023302</td>
          <td>23.140578</td>
          <td>0.023704</td>
          <td>22.754611</td>
          <td>0.038037</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.079991</td>
          <td>0.588857</td>
          <td>27.086374</td>
          <td>0.250438</td>
          <td>26.411027</td>
          <td>0.228429</td>
          <td>25.909487</td>
          <td>0.273855</td>
          <td>25.185682</td>
          <td>0.322667</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.279212</td>
          <td>0.352284</td>
          <td>25.764808</td>
          <td>0.084125</td>
          <td>25.360628</td>
          <td>0.052705</td>
          <td>24.703382</td>
          <td>0.048406</td>
          <td>24.368709</td>
          <td>0.068428</td>
          <td>23.670067</td>
          <td>0.083615</td>
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
          <td>26.894481</td>
          <td>0.567252</td>
          <td>26.285345</td>
          <td>0.134932</td>
          <td>26.294316</td>
          <td>0.122437</td>
          <td>26.238077</td>
          <td>0.188883</td>
          <td>25.598055</td>
          <td>0.202725</td>
          <td>25.687004</td>
          <td>0.456349</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.761147</td>
          <td>0.996194</td>
          <td>27.381657</td>
          <td>0.331197</td>
          <td>26.631145</td>
          <td>0.160973</td>
          <td>26.379750</td>
          <td>0.209248</td>
          <td>25.869388</td>
          <td>0.249959</td>
          <td>25.746983</td>
          <td>0.470348</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.989831</td>
          <td>1.143995</td>
          <td>27.422339</td>
          <td>0.344424</td>
          <td>26.472252</td>
          <td>0.141651</td>
          <td>26.112717</td>
          <td>0.168455</td>
          <td>26.037387</td>
          <td>0.288929</td>
          <td>25.358187</td>
          <td>0.351751</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.314144</td>
          <td>0.762343</td>
          <td>27.180034</td>
          <td>0.288233</td>
          <td>26.697772</td>
          <td>0.174930</td>
          <td>26.069478</td>
          <td>0.165424</td>
          <td>25.978490</td>
          <td>0.280241</td>
          <td>25.615322</td>
          <td>0.436301</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.675841</td>
          <td>0.480200</td>
          <td>26.416454</td>
          <td>0.149580</td>
          <td>25.937073</td>
          <td>0.088622</td>
          <td>25.858767</td>
          <td>0.135110</td>
          <td>25.235444</td>
          <td>0.147418</td>
          <td>24.866893</td>
          <td>0.235960</td>
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
          <td>27.067288</td>
          <td>0.575917</td>
          <td>26.506815</td>
          <td>0.139323</td>
          <td>25.946517</td>
          <td>0.075218</td>
          <td>25.342498</td>
          <td>0.071935</td>
          <td>25.088779</td>
          <td>0.109646</td>
          <td>24.735488</td>
          <td>0.178529</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.352246</td>
          <td>0.283196</td>
          <td>27.451568</td>
          <td>0.273132</td>
          <td>26.933765</td>
          <td>0.281200</td>
          <td>26.525037</td>
          <td>0.364055</td>
          <td>26.083780</td>
          <td>0.522845</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.810458</td>
          <td>0.501634</td>
          <td>25.996394</td>
          <td>0.095986</td>
          <td>24.788611</td>
          <td>0.029315</td>
          <td>23.870149</td>
          <td>0.021444</td>
          <td>23.154044</td>
          <td>0.021686</td>
          <td>22.864673</td>
          <td>0.037695</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.377384</td>
          <td>0.397004</td>
          <td>27.853220</td>
          <td>0.498303</td>
          <td>28.173238</td>
          <td>0.577912</td>
          <td>27.378785</td>
          <td>0.488605</td>
          <td>26.330152</td>
          <td>0.381432</td>
          <td>26.586083</td>
          <td>0.881124</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.508138</td>
          <td>0.379747</td>
          <td>25.853888</td>
          <td>0.078889</td>
          <td>25.435833</td>
          <td>0.047887</td>
          <td>24.760168</td>
          <td>0.042970</td>
          <td>24.395524</td>
          <td>0.059591</td>
          <td>23.842731</td>
          <td>0.082420</td>
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
          <td>27.794262</td>
          <td>0.972127</td>
          <td>26.256805</td>
          <td>0.120110</td>
          <td>26.255134</td>
          <td>0.106748</td>
          <td>26.128704</td>
          <td>0.155060</td>
          <td>25.617800</td>
          <td>0.186622</td>
          <td>25.880284</td>
          <td>0.481463</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.791357</td>
          <td>0.940894</td>
          <td>27.056544</td>
          <td>0.225014</td>
          <td>26.750821</td>
          <td>0.154206</td>
          <td>26.636030</td>
          <td>0.223574</td>
          <td>27.033352</td>
          <td>0.541403</td>
          <td>25.906842</td>
          <td>0.464956</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.713907</td>
          <td>0.457458</td>
          <td>27.390433</td>
          <td>0.303490</td>
          <td>26.763886</td>
          <td>0.160944</td>
          <td>26.384718</td>
          <td>0.187069</td>
          <td>26.104459</td>
          <td>0.271775</td>
          <td>25.116315</td>
          <td>0.257356</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.625775</td>
          <td>1.558402</td>
          <td>28.379883</td>
          <td>0.669123</td>
          <td>26.576527</td>
          <td>0.146149</td>
          <td>26.104308</td>
          <td>0.157494</td>
          <td>25.710407</td>
          <td>0.208784</td>
          <td>25.621594</td>
          <td>0.408817</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.133349</td>
          <td>0.616852</td>
          <td>26.720431</td>
          <td>0.172862</td>
          <td>26.121068</td>
          <td>0.091216</td>
          <td>25.739568</td>
          <td>0.106264</td>
          <td>25.221170</td>
          <td>0.127836</td>
          <td>24.889894</td>
          <td>0.211307</td>
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
