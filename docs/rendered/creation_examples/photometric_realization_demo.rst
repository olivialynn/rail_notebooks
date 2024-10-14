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

    <pzflow.flow.Flow at 0x7f5934c8b4f0>



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
          <td>27.417524</td>
          <td>0.733786</td>
          <td>26.527414</td>
          <td>0.141800</td>
          <td>26.026396</td>
          <td>0.080705</td>
          <td>25.366756</td>
          <td>0.073485</td>
          <td>25.021002</td>
          <td>0.103327</td>
          <td>25.150200</td>
          <td>0.252374</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.298879</td>
          <td>0.583477</td>
          <td>27.368606</td>
          <td>0.255012</td>
          <td>27.456775</td>
          <td>0.424130</td>
          <td>26.903002</td>
          <td>0.485309</td>
          <td>28.234589</td>
          <td>1.846690</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.587518</td>
          <td>0.403399</td>
          <td>26.013786</td>
          <td>0.090694</td>
          <td>24.794496</td>
          <td>0.027147</td>
          <td>23.889978</td>
          <td>0.020060</td>
          <td>23.176253</td>
          <td>0.020400</td>
          <td>22.770936</td>
          <td>0.031858</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.085071</td>
          <td>0.583223</td>
          <td>28.645590</td>
          <td>0.741182</td>
          <td>27.559799</td>
          <td>0.297874</td>
          <td>26.640565</td>
          <td>0.220777</td>
          <td>25.656613</td>
          <td>0.178789</td>
          <td>25.311662</td>
          <td>0.287856</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.410290</td>
          <td>0.351512</td>
          <td>25.784714</td>
          <td>0.074128</td>
          <td>25.441517</td>
          <td>0.048061</td>
          <td>24.818171</td>
          <td>0.045172</td>
          <td>24.311990</td>
          <td>0.055254</td>
          <td>23.631045</td>
          <td>0.068257</td>
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
          <td>26.801040</td>
          <td>0.474157</td>
          <td>26.141151</td>
          <td>0.101404</td>
          <td>26.171767</td>
          <td>0.091728</td>
          <td>25.803350</td>
          <td>0.107885</td>
          <td>26.194473</td>
          <td>0.279486</td>
          <td>25.774951</td>
          <td>0.414617</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.788090</td>
          <td>0.469598</td>
          <td>26.648045</td>
          <td>0.157260</td>
          <td>26.728258</td>
          <td>0.148837</td>
          <td>26.175017</td>
          <td>0.148880</td>
          <td>25.793096</td>
          <td>0.200611</td>
          <td>25.224845</td>
          <td>0.268266</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.023716</td>
          <td>0.558168</td>
          <td>27.425538</td>
          <td>0.300219</td>
          <td>26.944567</td>
          <td>0.179010</td>
          <td>26.487712</td>
          <td>0.194254</td>
          <td>25.965122</td>
          <td>0.231567</td>
          <td>24.723207</td>
          <td>0.176656</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.628255</td>
          <td>0.842350</td>
          <td>27.634942</td>
          <td>0.354566</td>
          <td>26.758169</td>
          <td>0.152706</td>
          <td>25.815811</td>
          <td>0.109066</td>
          <td>25.616111</td>
          <td>0.172747</td>
          <td>25.392910</td>
          <td>0.307312</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.164083</td>
          <td>0.616730</td>
          <td>26.576033</td>
          <td>0.147851</td>
          <td>26.181936</td>
          <td>0.092551</td>
          <td>25.657998</td>
          <td>0.094991</td>
          <td>25.112957</td>
          <td>0.111969</td>
          <td>24.912071</td>
          <td>0.207145</td>
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
          <td>27.914411</td>
          <td>1.088469</td>
          <td>26.562135</td>
          <td>0.167916</td>
          <td>25.981507</td>
          <td>0.091230</td>
          <td>25.406290</td>
          <td>0.090127</td>
          <td>25.362610</td>
          <td>0.162777</td>
          <td>25.362878</td>
          <td>0.348926</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.726060</td>
          <td>0.431396</td>
          <td>27.094007</td>
          <td>0.236688</td>
          <td>27.336195</td>
          <td>0.447654</td>
          <td>26.735144</td>
          <td>0.491067</td>
          <td>25.623887</td>
          <td>0.427156</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.947412</td>
          <td>0.589570</td>
          <td>25.897412</td>
          <td>0.096408</td>
          <td>24.749813</td>
          <td>0.031393</td>
          <td>23.833413</td>
          <td>0.023065</td>
          <td>23.116290</td>
          <td>0.023214</td>
          <td>22.846683</td>
          <td>0.041265</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.219928</td>
          <td>0.731225</td>
          <td>28.464141</td>
          <td>0.766354</td>
          <td>27.399334</td>
          <td>0.322615</td>
          <td>26.865729</td>
          <td>0.330482</td>
          <td>25.885916</td>
          <td>0.268651</td>
          <td>25.442131</td>
          <td>0.394538</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.152207</td>
          <td>0.318625</td>
          <td>25.875770</td>
          <td>0.092736</td>
          <td>25.433095</td>
          <td>0.056205</td>
          <td>24.848270</td>
          <td>0.055047</td>
          <td>24.394372</td>
          <td>0.070000</td>
          <td>23.488065</td>
          <td>0.071206</td>
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
          <td>25.993872</td>
          <td>0.284693</td>
          <td>26.448264</td>
          <td>0.155205</td>
          <td>26.286677</td>
          <td>0.121628</td>
          <td>25.983103</td>
          <td>0.152047</td>
          <td>26.047655</td>
          <td>0.293536</td>
          <td>24.765200</td>
          <td>0.219183</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.607052</td>
          <td>0.454405</td>
          <td>26.993295</td>
          <td>0.241887</td>
          <td>26.570119</td>
          <td>0.152784</td>
          <td>26.131126</td>
          <td>0.169644</td>
          <td>26.090061</td>
          <td>0.299091</td>
          <td>25.323773</td>
          <td>0.339633</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.186811</td>
          <td>0.330330</td>
          <td>27.011110</td>
          <td>0.247256</td>
          <td>26.926464</td>
          <td>0.208367</td>
          <td>26.534307</td>
          <td>0.239939</td>
          <td>25.691604</td>
          <td>0.217513</td>
          <td>25.941085</td>
          <td>0.546543</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.930886</td>
          <td>0.585984</td>
          <td>27.485216</td>
          <td>0.367327</td>
          <td>26.421259</td>
          <td>0.138062</td>
          <td>25.824231</td>
          <td>0.134022</td>
          <td>25.470882</td>
          <td>0.183960</td>
          <td>25.680204</td>
          <td>0.458191</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.560559</td>
          <td>0.883121</td>
          <td>26.384708</td>
          <td>0.145560</td>
          <td>26.097239</td>
          <td>0.101994</td>
          <td>25.779907</td>
          <td>0.126200</td>
          <td>25.133160</td>
          <td>0.134987</td>
          <td>25.036234</td>
          <td>0.271134</td>
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
          <td>26.404536</td>
          <td>0.349955</td>
          <td>26.732165</td>
          <td>0.168976</td>
          <td>26.023477</td>
          <td>0.080508</td>
          <td>25.399523</td>
          <td>0.075656</td>
          <td>25.083840</td>
          <td>0.109175</td>
          <td>24.728914</td>
          <td>0.177536</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>32.357542</td>
          <td>4.913939</td>
          <td>31.257305</td>
          <td>2.685070</td>
          <td>27.911965</td>
          <td>0.393616</td>
          <td>27.207729</td>
          <td>0.350017</td>
          <td>26.838974</td>
          <td>0.463045</td>
          <td>27.793402</td>
          <td>1.501423</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.485585</td>
          <td>0.392581</td>
          <td>25.899364</td>
          <td>0.088153</td>
          <td>24.813367</td>
          <td>0.029958</td>
          <td>23.889980</td>
          <td>0.021811</td>
          <td>23.137723</td>
          <td>0.021386</td>
          <td>22.794496</td>
          <td>0.035429</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.030043</td>
          <td>1.958451</td>
          <td>28.257562</td>
          <td>0.665014</td>
          <td>27.369351</td>
          <td>0.313972</td>
          <td>26.462044</td>
          <td>0.237458</td>
          <td>29.628901</td>
          <td>2.475724</td>
          <td>25.526429</td>
          <td>0.419569</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.776737</td>
          <td>0.466020</td>
          <td>25.861255</td>
          <td>0.079403</td>
          <td>25.371181</td>
          <td>0.045216</td>
          <td>24.761966</td>
          <td>0.043039</td>
          <td>24.323678</td>
          <td>0.055911</td>
          <td>23.672131</td>
          <td>0.070890</td>
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
          <td>26.259613</td>
          <td>0.328162</td>
          <td>26.508414</td>
          <td>0.149245</td>
          <td>26.194380</td>
          <td>0.101223</td>
          <td>26.028782</td>
          <td>0.142309</td>
          <td>25.781941</td>
          <td>0.214205</td>
          <td>25.638664</td>
          <td>0.401001</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.128501</td>
          <td>0.607034</td>
          <td>27.791313</td>
          <td>0.405446</td>
          <td>26.908623</td>
          <td>0.176418</td>
          <td>26.653047</td>
          <td>0.226757</td>
          <td>26.053861</td>
          <td>0.253023</td>
          <td>26.725568</td>
          <td>0.824010</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.176900</td>
          <td>0.639550</td>
          <td>27.024256</td>
          <td>0.225019</td>
          <td>27.015144</td>
          <td>0.199136</td>
          <td>26.545137</td>
          <td>0.214048</td>
          <td>25.491367</td>
          <td>0.162869</td>
          <td>25.036327</td>
          <td>0.240977</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.678146</td>
          <td>1.598525</td>
          <td>27.423097</td>
          <td>0.328404</td>
          <td>26.447831</td>
          <td>0.130796</td>
          <td>25.977641</td>
          <td>0.141266</td>
          <td>26.159987</td>
          <td>0.301963</td>
          <td>25.757866</td>
          <td>0.453418</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.104511</td>
          <td>0.604458</td>
          <td>26.504819</td>
          <td>0.143769</td>
          <td>26.106278</td>
          <td>0.090037</td>
          <td>25.843461</td>
          <td>0.116344</td>
          <td>25.301933</td>
          <td>0.137082</td>
          <td>24.908885</td>
          <td>0.214685</td>
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
