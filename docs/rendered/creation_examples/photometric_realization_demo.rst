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

    <pzflow.flow.Flow at 0x7fc62a9611b0>



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
          <td>27.850195</td>
          <td>0.967655</td>
          <td>26.498706</td>
          <td>0.138337</td>
          <td>25.962288</td>
          <td>0.076264</td>
          <td>25.340619</td>
          <td>0.071806</td>
          <td>25.119827</td>
          <td>0.112642</td>
          <td>25.362842</td>
          <td>0.299982</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>31.197047</td>
          <td>3.774980</td>
          <td>29.056641</td>
          <td>0.963834</td>
          <td>27.323140</td>
          <td>0.245660</td>
          <td>26.840257</td>
          <td>0.260339</td>
          <td>32.483709</td>
          <td>4.968937</td>
          <td>26.771613</td>
          <td>0.838230</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.315761</td>
          <td>0.685018</td>
          <td>25.892838</td>
          <td>0.081543</td>
          <td>24.821992</td>
          <td>0.027807</td>
          <td>23.868635</td>
          <td>0.019700</td>
          <td>23.121837</td>
          <td>0.019482</td>
          <td>22.882762</td>
          <td>0.035158</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.884754</td>
          <td>0.988163</td>
          <td>29.897441</td>
          <td>1.532184</td>
          <td>26.950497</td>
          <td>0.179912</td>
          <td>26.547888</td>
          <td>0.204329</td>
          <td>26.160360</td>
          <td>0.271845</td>
          <td>27.082210</td>
          <td>1.016206</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.137302</td>
          <td>0.605215</td>
          <td>25.848477</td>
          <td>0.078417</td>
          <td>25.448447</td>
          <td>0.048357</td>
          <td>24.854821</td>
          <td>0.046665</td>
          <td>24.410679</td>
          <td>0.060311</td>
          <td>23.632669</td>
          <td>0.068355</td>
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
          <td>26.784749</td>
          <td>0.468428</td>
          <td>26.294848</td>
          <td>0.115951</td>
          <td>26.253754</td>
          <td>0.098572</td>
          <td>25.894228</td>
          <td>0.116782</td>
          <td>25.842082</td>
          <td>0.209019</td>
          <td>25.712307</td>
          <td>0.395132</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.682504</td>
          <td>0.433729</td>
          <td>27.217565</td>
          <td>0.253563</td>
          <td>26.825399</td>
          <td>0.161749</td>
          <td>26.659957</td>
          <td>0.224367</td>
          <td>25.697657</td>
          <td>0.185110</td>
          <td>24.970403</td>
          <td>0.217490</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.526910</td>
          <td>0.384979</td>
          <td>27.036348</td>
          <td>0.218287</td>
          <td>26.963747</td>
          <td>0.181942</td>
          <td>26.486843</td>
          <td>0.194112</td>
          <td>26.511720</td>
          <td>0.359977</td>
          <td>25.074306</td>
          <td>0.237079</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.909841</td>
          <td>0.513865</td>
          <td>27.383884</td>
          <td>0.290313</td>
          <td>26.539194</td>
          <td>0.126431</td>
          <td>25.846957</td>
          <td>0.112070</td>
          <td>25.315118</td>
          <td>0.133453</td>
          <td>25.545597</td>
          <td>0.346961</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.920549</td>
          <td>0.517911</td>
          <td>26.394311</td>
          <td>0.126406</td>
          <td>26.105599</td>
          <td>0.086541</td>
          <td>25.799861</td>
          <td>0.107557</td>
          <td>25.273792</td>
          <td>0.128766</td>
          <td>24.530294</td>
          <td>0.149837</td>
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
          <td>26.757918</td>
          <td>0.198147</td>
          <td>25.968483</td>
          <td>0.090192</td>
          <td>25.245098</td>
          <td>0.078196</td>
          <td>24.949390</td>
          <td>0.113963</td>
          <td>24.922664</td>
          <td>0.244691</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.411023</td>
          <td>1.428005</td>
          <td>29.296976</td>
          <td>1.218552</td>
          <td>27.848246</td>
          <td>0.431234</td>
          <td>26.998716</td>
          <td>0.344995</td>
          <td>26.115405</td>
          <td>0.304140</td>
          <td>26.290146</td>
          <td>0.691264</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.921363</td>
          <td>0.578742</td>
          <td>25.821061</td>
          <td>0.090166</td>
          <td>24.757900</td>
          <td>0.031617</td>
          <td>23.893806</td>
          <td>0.024297</td>
          <td>23.156121</td>
          <td>0.024024</td>
          <td>22.811591</td>
          <td>0.040003</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.912436</td>
          <td>1.016174</td>
          <td>27.334378</td>
          <td>0.306299</td>
          <td>26.983384</td>
          <td>0.362587</td>
          <td>25.820679</td>
          <td>0.254699</td>
          <td>25.273455</td>
          <td>0.345899</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.592632</td>
          <td>0.448377</td>
          <td>25.872385</td>
          <td>0.092461</td>
          <td>25.472405</td>
          <td>0.058199</td>
          <td>24.797794</td>
          <td>0.052636</td>
          <td>24.399572</td>
          <td>0.070322</td>
          <td>23.857587</td>
          <td>0.098592</td>
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
          <td>26.426226</td>
          <td>0.400467</td>
          <td>26.648319</td>
          <td>0.183987</td>
          <td>26.016623</td>
          <td>0.096084</td>
          <td>26.084690</td>
          <td>0.165841</td>
          <td>25.902330</td>
          <td>0.260860</td>
          <td>25.052835</td>
          <td>0.277694</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.521752</td>
          <td>0.426025</td>
          <td>26.989341</td>
          <td>0.241100</td>
          <td>26.858098</td>
          <td>0.195137</td>
          <td>26.149461</td>
          <td>0.172310</td>
          <td>26.043717</td>
          <td>0.288122</td>
          <td>25.681027</td>
          <td>0.447626</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.034824</td>
          <td>0.623378</td>
          <td>27.405548</td>
          <td>0.339891</td>
          <td>27.061634</td>
          <td>0.233179</td>
          <td>26.394528</td>
          <td>0.213652</td>
          <td>25.565531</td>
          <td>0.195720</td>
          <td>25.789224</td>
          <td>0.489017</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.852057</td>
          <td>1.066436</td>
          <td>27.139585</td>
          <td>0.278949</td>
          <td>26.545070</td>
          <td>0.153567</td>
          <td>25.965169</td>
          <td>0.151309</td>
          <td>25.222890</td>
          <td>0.148915</td>
          <td>26.488646</td>
          <td>0.808227</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.588566</td>
          <td>0.173243</td>
          <td>25.966958</td>
          <td>0.090981</td>
          <td>25.680141</td>
          <td>0.115724</td>
          <td>24.958513</td>
          <td>0.116020</td>
          <td>25.040828</td>
          <td>0.272150</td>
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
          <td>27.045485</td>
          <td>0.567003</td>
          <td>27.075178</td>
          <td>0.225475</td>
          <td>25.986315</td>
          <td>0.077910</td>
          <td>25.409457</td>
          <td>0.076323</td>
          <td>24.954167</td>
          <td>0.097464</td>
          <td>24.743423</td>
          <td>0.179734</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>31.654108</td>
          <td>4.221173</td>
          <td>27.884468</td>
          <td>0.430275</td>
          <td>27.097773</td>
          <td>0.203880</td>
          <td>26.875815</td>
          <td>0.268258</td>
          <td>26.301348</td>
          <td>0.304922</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.030802</td>
          <td>0.588353</td>
          <td>25.837341</td>
          <td>0.083474</td>
          <td>24.771789</td>
          <td>0.028887</td>
          <td>23.875764</td>
          <td>0.021547</td>
          <td>23.131761</td>
          <td>0.021277</td>
          <td>22.834107</td>
          <td>0.036690</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.205710</td>
          <td>3.008822</td>
          <td>28.204095</td>
          <td>0.640864</td>
          <td>27.360348</td>
          <td>0.311720</td>
          <td>26.904570</td>
          <td>0.339676</td>
          <td>26.067258</td>
          <td>0.310017</td>
          <td>26.723429</td>
          <td>0.959582</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.079282</td>
          <td>0.269993</td>
          <td>25.751593</td>
          <td>0.072081</td>
          <td>25.318236</td>
          <td>0.043141</td>
          <td>24.793811</td>
          <td>0.044272</td>
          <td>24.356073</td>
          <td>0.057541</td>
          <td>23.674927</td>
          <td>0.071066</td>
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
          <td>27.727427</td>
          <td>0.933109</td>
          <td>26.482373</td>
          <td>0.145946</td>
          <td>26.146421</td>
          <td>0.097057</td>
          <td>26.208158</td>
          <td>0.165952</td>
          <td>26.096544</td>
          <td>0.277569</td>
          <td>25.872118</td>
          <td>0.478547</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.023714</td>
          <td>1.081488</td>
          <td>27.004014</td>
          <td>0.215390</td>
          <td>26.683162</td>
          <td>0.145505</td>
          <td>26.250146</td>
          <td>0.161472</td>
          <td>26.221532</td>
          <td>0.290037</td>
          <td>25.819911</td>
          <td>0.435472</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.731782</td>
          <td>0.463629</td>
          <td>27.697139</td>
          <td>0.386554</td>
          <td>27.268962</td>
          <td>0.245958</td>
          <td>26.401060</td>
          <td>0.189667</td>
          <td>25.698878</td>
          <td>0.194201</td>
          <td>25.911540</td>
          <td>0.480101</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.689618</td>
          <td>0.467863</td>
          <td>26.877344</td>
          <td>0.210389</td>
          <td>26.517039</td>
          <td>0.138852</td>
          <td>25.774389</td>
          <td>0.118476</td>
          <td>25.423585</td>
          <td>0.163841</td>
          <td>24.999812</td>
          <td>0.249207</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.480786</td>
          <td>0.380504</td>
          <td>26.750232</td>
          <td>0.177289</td>
          <td>26.074617</td>
          <td>0.087564</td>
          <td>25.511911</td>
          <td>0.087025</td>
          <td>25.223393</td>
          <td>0.128083</td>
          <td>24.749614</td>
          <td>0.187815</td>
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
