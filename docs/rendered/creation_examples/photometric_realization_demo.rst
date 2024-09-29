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

    <pzflow.flow.Flow at 0x7f700ea25000>



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
          <td>29.156110</td>
          <td>1.906073</td>
          <td>26.761053</td>
          <td>0.173157</td>
          <td>26.128935</td>
          <td>0.088337</td>
          <td>25.459260</td>
          <td>0.079743</td>
          <td>25.249443</td>
          <td>0.126078</td>
          <td>25.081804</td>
          <td>0.238553</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.705964</td>
          <td>0.374806</td>
          <td>27.174344</td>
          <td>0.217164</td>
          <td>26.957960</td>
          <td>0.286501</td>
          <td>27.265798</td>
          <td>0.630376</td>
          <td>25.965126</td>
          <td>0.478633</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.271620</td>
          <td>0.314966</td>
          <td>26.021369</td>
          <td>0.091300</td>
          <td>24.741420</td>
          <td>0.025920</td>
          <td>23.841219</td>
          <td>0.019250</td>
          <td>23.147078</td>
          <td>0.019902</td>
          <td>22.842079</td>
          <td>0.033918</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.137048</td>
          <td>0.605107</td>
          <td>27.827720</td>
          <td>0.411751</td>
          <td>27.654253</td>
          <td>0.321279</td>
          <td>27.509907</td>
          <td>0.441590</td>
          <td>25.838731</td>
          <td>0.208434</td>
          <td>25.184094</td>
          <td>0.259484</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.263936</td>
          <td>0.313040</td>
          <td>25.723515</td>
          <td>0.070228</td>
          <td>25.461012</td>
          <td>0.048900</td>
          <td>24.866819</td>
          <td>0.047165</td>
          <td>24.377614</td>
          <td>0.058568</td>
          <td>23.705421</td>
          <td>0.072900</td>
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
          <td>26.856037</td>
          <td>0.493913</td>
          <td>26.169424</td>
          <td>0.103941</td>
          <td>26.087171</td>
          <td>0.085147</td>
          <td>25.848173</td>
          <td>0.112189</td>
          <td>25.659778</td>
          <td>0.179269</td>
          <td>25.215982</td>
          <td>0.266334</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.026055</td>
          <td>0.559107</td>
          <td>26.882174</td>
          <td>0.191839</td>
          <td>27.052420</td>
          <td>0.196081</td>
          <td>26.013365</td>
          <td>0.129507</td>
          <td>26.551998</td>
          <td>0.371488</td>
          <td>25.361642</td>
          <td>0.299693</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.489604</td>
          <td>0.316027</td>
          <td>26.753671</td>
          <td>0.152118</td>
          <td>26.461489</td>
          <td>0.190008</td>
          <td>25.958345</td>
          <td>0.230270</td>
          <td>24.898710</td>
          <td>0.204839</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.647700</td>
          <td>0.422405</td>
          <td>27.327681</td>
          <td>0.277402</td>
          <td>26.491237</td>
          <td>0.121278</td>
          <td>25.691178</td>
          <td>0.097797</td>
          <td>25.173130</td>
          <td>0.117993</td>
          <td>25.234076</td>
          <td>0.270292</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.313513</td>
          <td>0.325646</td>
          <td>26.709992</td>
          <td>0.165798</td>
          <td>26.092367</td>
          <td>0.085538</td>
          <td>25.581471</td>
          <td>0.088812</td>
          <td>25.062264</td>
          <td>0.107123</td>
          <td>25.069550</td>
          <td>0.236149</td>
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
          <td>26.717487</td>
          <td>0.191522</td>
          <td>25.942375</td>
          <td>0.088145</td>
          <td>25.318558</td>
          <td>0.083430</td>
          <td>25.089895</td>
          <td>0.128753</td>
          <td>24.957543</td>
          <td>0.251811</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.931738</td>
          <td>1.099578</td>
          <td>27.763190</td>
          <td>0.443701</td>
          <td>27.347264</td>
          <td>0.291101</td>
          <td>27.181395</td>
          <td>0.397807</td>
          <td>28.023943</td>
          <td>1.146118</td>
          <td>27.968546</td>
          <td>1.784957</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.613460</td>
          <td>0.462040</td>
          <td>25.843412</td>
          <td>0.091951</td>
          <td>24.792156</td>
          <td>0.032584</td>
          <td>23.844526</td>
          <td>0.023286</td>
          <td>23.150544</td>
          <td>0.023909</td>
          <td>22.829775</td>
          <td>0.040652</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.564914</td>
          <td>0.818524</td>
          <td>26.846311</td>
          <td>0.205199</td>
          <td>26.869444</td>
          <td>0.331458</td>
          <td>26.143231</td>
          <td>0.330435</td>
          <td>25.665790</td>
          <td>0.467640</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.544985</td>
          <td>0.432514</td>
          <td>25.691013</td>
          <td>0.078835</td>
          <td>25.414371</td>
          <td>0.055279</td>
          <td>24.954411</td>
          <td>0.060481</td>
          <td>24.220176</td>
          <td>0.059992</td>
          <td>23.572396</td>
          <td>0.076715</td>
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
          <td>27.885311</td>
          <td>1.081597</td>
          <td>26.778752</td>
          <td>0.205324</td>
          <td>25.998329</td>
          <td>0.094554</td>
          <td>25.919287</td>
          <td>0.143938</td>
          <td>26.624687</td>
          <td>0.460288</td>
          <td>25.593768</td>
          <td>0.425266</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.367447</td>
          <td>0.378386</td>
          <td>27.843193</td>
          <td>0.472612</td>
          <td>26.695206</td>
          <td>0.170008</td>
          <td>26.289061</td>
          <td>0.193913</td>
          <td>26.608237</td>
          <td>0.448135</td>
          <td>25.151345</td>
          <td>0.295976</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.728296</td>
          <td>1.677303</td>
          <td>27.183344</td>
          <td>0.284556</td>
          <td>26.819298</td>
          <td>0.190428</td>
          <td>26.183494</td>
          <td>0.178894</td>
          <td>26.053142</td>
          <td>0.292627</td>
          <td>25.619634</td>
          <td>0.430563</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.069128</td>
          <td>0.645733</td>
          <td>26.644983</td>
          <td>0.185143</td>
          <td>26.615966</td>
          <td>0.163164</td>
          <td>26.076500</td>
          <td>0.166417</td>
          <td>25.607375</td>
          <td>0.206353</td>
          <td>29.068894</td>
          <td>2.776238</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.433809</td>
          <td>0.399863</td>
          <td>26.499700</td>
          <td>0.160621</td>
          <td>26.087263</td>
          <td>0.101107</td>
          <td>25.816292</td>
          <td>0.130239</td>
          <td>25.219481</td>
          <td>0.145409</td>
          <td>24.983814</td>
          <td>0.259778</td>
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
          <td>29.534237</td>
          <td>2.227621</td>
          <td>26.671224</td>
          <td>0.160423</td>
          <td>26.023192</td>
          <td>0.080488</td>
          <td>25.261982</td>
          <td>0.066986</td>
          <td>25.026648</td>
          <td>0.103852</td>
          <td>24.712316</td>
          <td>0.175053</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.245124</td>
          <td>1.980832</td>
          <td>28.582163</td>
          <td>0.710725</td>
          <td>27.001597</td>
          <td>0.188030</td>
          <td>27.303522</td>
          <td>0.377253</td>
          <td>26.344107</td>
          <td>0.315540</td>
          <td>27.401698</td>
          <td>1.222054</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.659622</td>
          <td>0.448305</td>
          <td>25.791191</td>
          <td>0.080151</td>
          <td>24.754730</td>
          <td>0.028458</td>
          <td>23.888397</td>
          <td>0.021781</td>
          <td>23.137802</td>
          <td>0.021387</td>
          <td>22.864590</td>
          <td>0.037692</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.164045</td>
          <td>2.808190</td>
          <td>27.410992</td>
          <td>0.324571</td>
          <td>26.606809</td>
          <td>0.267422</td>
          <td>26.484127</td>
          <td>0.429331</td>
          <td>25.099473</td>
          <td>0.300162</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.059193</td>
          <td>0.265613</td>
          <td>25.831319</td>
          <td>0.077335</td>
          <td>25.525996</td>
          <td>0.051879</td>
          <td>24.862509</td>
          <td>0.047056</td>
          <td>24.483315</td>
          <td>0.064416</td>
          <td>23.736991</td>
          <td>0.075075</td>
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
          <td>27.255769</td>
          <td>0.686672</td>
          <td>26.258461</td>
          <td>0.120283</td>
          <td>26.173371</td>
          <td>0.099378</td>
          <td>25.817796</td>
          <td>0.118558</td>
          <td>25.611797</td>
          <td>0.185678</td>
          <td>25.983665</td>
          <td>0.519616</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.696142</td>
          <td>0.886755</td>
          <td>27.465751</td>
          <td>0.314125</td>
          <td>26.628920</td>
          <td>0.138865</td>
          <td>26.465955</td>
          <td>0.193913</td>
          <td>26.336076</td>
          <td>0.317970</td>
          <td>25.266958</td>
          <td>0.282036</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.624638</td>
          <td>0.427626</td>
          <td>27.778358</td>
          <td>0.411506</td>
          <td>26.426203</td>
          <td>0.120298</td>
          <td>26.556442</td>
          <td>0.216077</td>
          <td>26.269063</td>
          <td>0.310400</td>
          <td>25.189022</td>
          <td>0.273093</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.412347</td>
          <td>1.399749</td>
          <td>26.993212</td>
          <td>0.231675</td>
          <td>26.621562</td>
          <td>0.151910</td>
          <td>25.907370</td>
          <td>0.132955</td>
          <td>25.750445</td>
          <td>0.215885</td>
          <td>25.432954</td>
          <td>0.353113</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.454090</td>
          <td>0.372694</td>
          <td>26.511447</td>
          <td>0.144591</td>
          <td>26.162315</td>
          <td>0.094581</td>
          <td>25.518311</td>
          <td>0.087517</td>
          <td>25.182602</td>
          <td>0.123632</td>
          <td>25.133792</td>
          <td>0.258560</td>
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
