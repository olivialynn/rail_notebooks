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

    <pzflow.flow.Flow at 0x7fc304b6aad0>



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
          <td>26.787263</td>
          <td>0.469309</td>
          <td>26.441899</td>
          <td>0.131719</td>
          <td>26.086759</td>
          <td>0.085116</td>
          <td>25.193885</td>
          <td>0.063054</td>
          <td>24.983818</td>
          <td>0.100017</td>
          <td>24.786435</td>
          <td>0.186371</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.766723</td>
          <td>0.392881</td>
          <td>27.576867</td>
          <td>0.301990</td>
          <td>26.961885</td>
          <td>0.287412</td>
          <td>26.704129</td>
          <td>0.417778</td>
          <td>25.102843</td>
          <td>0.242730</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.911046</td>
          <td>0.235000</td>
          <td>26.155097</td>
          <td>0.102648</td>
          <td>24.734676</td>
          <td>0.025769</td>
          <td>23.875434</td>
          <td>0.019814</td>
          <td>23.133237</td>
          <td>0.019670</td>
          <td>22.836784</td>
          <td>0.033760</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.022271</td>
          <td>1.072373</td>
          <td>31.060508</td>
          <td>2.505483</td>
          <td>27.796510</td>
          <td>0.359503</td>
          <td>27.558329</td>
          <td>0.458005</td>
          <td>26.314814</td>
          <td>0.307966</td>
          <td>25.232353</td>
          <td>0.269913</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.315238</td>
          <td>0.326092</td>
          <td>25.776308</td>
          <td>0.073580</td>
          <td>25.337826</td>
          <td>0.043835</td>
          <td>24.741163</td>
          <td>0.042188</td>
          <td>24.394258</td>
          <td>0.059439</td>
          <td>23.745441</td>
          <td>0.075526</td>
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
          <td>26.379143</td>
          <td>0.343002</td>
          <td>26.242201</td>
          <td>0.110757</td>
          <td>26.146772</td>
          <td>0.089734</td>
          <td>25.969274</td>
          <td>0.124652</td>
          <td>25.656422</td>
          <td>0.178760</td>
          <td>25.110871</td>
          <td>0.244342</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.579372</td>
          <td>2.266966</td>
          <td>26.615139</td>
          <td>0.152893</td>
          <td>26.785308</td>
          <td>0.156298</td>
          <td>26.440455</td>
          <td>0.186663</td>
          <td>26.119845</td>
          <td>0.263009</td>
          <td>25.438209</td>
          <td>0.318646</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.639394</td>
          <td>0.419739</td>
          <td>27.743810</td>
          <td>0.385981</td>
          <td>27.278249</td>
          <td>0.236729</td>
          <td>26.215748</td>
          <td>0.154173</td>
          <td>25.684256</td>
          <td>0.183024</td>
          <td>25.063805</td>
          <td>0.235030</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.076655</td>
          <td>0.225727</td>
          <td>26.453342</td>
          <td>0.117347</td>
          <td>25.836163</td>
          <td>0.111020</td>
          <td>25.455665</td>
          <td>0.150625</td>
          <td>25.051966</td>
          <td>0.232738</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.444173</td>
          <td>0.360972</td>
          <td>26.888613</td>
          <td>0.192882</td>
          <td>26.118298</td>
          <td>0.087514</td>
          <td>25.616153</td>
          <td>0.091562</td>
          <td>25.191774</td>
          <td>0.119922</td>
          <td>24.779504</td>
          <td>0.185282</td>
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
          <td>26.606212</td>
          <td>0.174325</td>
          <td>25.929313</td>
          <td>0.087138</td>
          <td>25.241126</td>
          <td>0.077923</td>
          <td>25.150890</td>
          <td>0.135724</td>
          <td>24.933874</td>
          <td>0.246960</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.983166</td>
          <td>0.596519</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.164849</td>
          <td>0.545441</td>
          <td>27.117965</td>
          <td>0.378751</td>
          <td>26.264473</td>
          <td>0.342457</td>
          <td>25.823286</td>
          <td>0.496074</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>28.036234</td>
          <td>1.179933</td>
          <td>26.126006</td>
          <td>0.117684</td>
          <td>24.785985</td>
          <td>0.032408</td>
          <td>23.956293</td>
          <td>0.025648</td>
          <td>23.163354</td>
          <td>0.024174</td>
          <td>22.812328</td>
          <td>0.040029</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.376275</td>
          <td>0.316739</td>
          <td>26.871755</td>
          <td>0.332066</td>
          <td>26.435335</td>
          <td>0.414932</td>
          <td>26.198440</td>
          <td>0.684875</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.668841</td>
          <td>0.474724</td>
          <td>25.700401</td>
          <td>0.079489</td>
          <td>25.408113</td>
          <td>0.054973</td>
          <td>24.816224</td>
          <td>0.053504</td>
          <td>24.303489</td>
          <td>0.064589</td>
          <td>23.859269</td>
          <td>0.098738</td>
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
          <td>27.205192</td>
          <td>0.704434</td>
          <td>26.492144</td>
          <td>0.161134</td>
          <td>26.179930</td>
          <td>0.110838</td>
          <td>25.824016</td>
          <td>0.132584</td>
          <td>25.768480</td>
          <td>0.233658</td>
          <td>25.351285</td>
          <td>0.352496</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.290144</td>
          <td>0.738546</td>
          <td>27.164290</td>
          <td>0.278195</td>
          <td>26.502913</td>
          <td>0.144217</td>
          <td>26.393253</td>
          <td>0.211624</td>
          <td>26.079816</td>
          <td>0.296635</td>
          <td>25.341863</td>
          <td>0.344518</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.483031</td>
          <td>2.309211</td>
          <td>27.646607</td>
          <td>0.410043</td>
          <td>26.687160</td>
          <td>0.170263</td>
          <td>26.361703</td>
          <td>0.207869</td>
          <td>25.928370</td>
          <td>0.264445</td>
          <td>25.047275</td>
          <td>0.274296</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.942619</td>
          <td>0.590888</td>
          <td>27.165725</td>
          <td>0.284918</td>
          <td>26.378066</td>
          <td>0.133009</td>
          <td>25.966876</td>
          <td>0.151530</td>
          <td>25.498166</td>
          <td>0.188250</td>
          <td>25.336121</td>
          <td>0.351665</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.513171</td>
          <td>0.424893</td>
          <td>26.373947</td>
          <td>0.144220</td>
          <td>26.035158</td>
          <td>0.096595</td>
          <td>25.853762</td>
          <td>0.134527</td>
          <td>25.247575</td>
          <td>0.148962</td>
          <td>25.219349</td>
          <td>0.314295</td>
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
          <td>27.150785</td>
          <td>0.611036</td>
          <td>26.869479</td>
          <td>0.189818</td>
          <td>26.169260</td>
          <td>0.091538</td>
          <td>25.218669</td>
          <td>0.064463</td>
          <td>24.986106</td>
          <td>0.100231</td>
          <td>24.642167</td>
          <td>0.164910</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.551566</td>
          <td>1.433659</td>
          <td>27.630641</td>
          <td>0.353629</td>
          <td>28.117133</td>
          <td>0.460171</td>
          <td>27.511634</td>
          <td>0.442545</td>
          <td>26.054022</td>
          <td>0.249416</td>
          <td>25.558290</td>
          <td>0.350754</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.882993</td>
          <td>0.242607</td>
          <td>25.900966</td>
          <td>0.088277</td>
          <td>24.799085</td>
          <td>0.029586</td>
          <td>23.886097</td>
          <td>0.021739</td>
          <td>23.159406</td>
          <td>0.021786</td>
          <td>22.889408</td>
          <td>0.038529</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.903165</td>
          <td>3.506163</td>
          <td>27.827624</td>
          <td>0.448357</td>
          <td>26.675643</td>
          <td>0.282809</td>
          <td>26.362991</td>
          <td>0.391257</td>
          <td>26.198286</td>
          <td>0.682859</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.943933</td>
          <td>0.527261</td>
          <td>25.819700</td>
          <td>0.076547</td>
          <td>25.463529</td>
          <td>0.049080</td>
          <td>24.833712</td>
          <td>0.045868</td>
          <td>24.279096</td>
          <td>0.053741</td>
          <td>23.744666</td>
          <td>0.075586</td>
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
          <td>25.748025</td>
          <td>0.216421</td>
          <td>26.669777</td>
          <td>0.171292</td>
          <td>26.273223</td>
          <td>0.108448</td>
          <td>26.323969</td>
          <td>0.183104</td>
          <td>25.792188</td>
          <td>0.216044</td>
          <td>25.848744</td>
          <td>0.470277</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.987866</td>
          <td>0.549073</td>
          <td>26.702052</td>
          <td>0.166984</td>
          <td>26.827647</td>
          <td>0.164673</td>
          <td>26.351780</td>
          <td>0.176068</td>
          <td>26.209406</td>
          <td>0.287208</td>
          <td>25.311569</td>
          <td>0.292394</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.555559</td>
          <td>1.464896</td>
          <td>27.386029</td>
          <td>0.302419</td>
          <td>27.108289</td>
          <td>0.215289</td>
          <td>26.424352</td>
          <td>0.193428</td>
          <td>25.939442</td>
          <td>0.237369</td>
          <td>25.773491</td>
          <td>0.432778</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.195873</td>
          <td>0.273592</td>
          <td>26.680153</td>
          <td>0.159724</td>
          <td>26.126229</td>
          <td>0.160473</td>
          <td>25.537847</td>
          <td>0.180553</td>
          <td>25.285174</td>
          <td>0.314089</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.564463</td>
          <td>0.405878</td>
          <td>26.642325</td>
          <td>0.161742</td>
          <td>26.316780</td>
          <td>0.108280</td>
          <td>25.631073</td>
          <td>0.096633</td>
          <td>25.325462</td>
          <td>0.139892</td>
          <td>25.216465</td>
          <td>0.276594</td>
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
