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

    <pzflow.flow.Flow at 0x7f27e526f670>



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
          <td>27.069647</td>
          <td>0.576845</td>
          <td>26.843117</td>
          <td>0.185621</td>
          <td>26.153233</td>
          <td>0.090245</td>
          <td>25.352163</td>
          <td>0.072543</td>
          <td>24.963691</td>
          <td>0.098269</td>
          <td>25.143562</td>
          <td>0.251002</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>35.530933</td>
          <td>8.075401</td>
          <td>28.017763</td>
          <td>0.475352</td>
          <td>27.187246</td>
          <td>0.219511</td>
          <td>27.962870</td>
          <td>0.614763</td>
          <td>27.169916</td>
          <td>0.589197</td>
          <td>26.686615</td>
          <td>0.793344</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.240390</td>
          <td>0.650429</td>
          <td>25.882387</td>
          <td>0.080795</td>
          <td>24.781235</td>
          <td>0.026835</td>
          <td>23.872381</td>
          <td>0.019763</td>
          <td>23.164776</td>
          <td>0.020202</td>
          <td>22.834145</td>
          <td>0.033682</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.786213</td>
          <td>0.468940</td>
          <td>28.080167</td>
          <td>0.497876</td>
          <td>27.665652</td>
          <td>0.324208</td>
          <td>26.607082</td>
          <td>0.214701</td>
          <td>25.747151</td>
          <td>0.193006</td>
          <td>25.106300</td>
          <td>0.243423</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.016004</td>
          <td>0.256174</td>
          <td>25.722689</td>
          <td>0.070177</td>
          <td>25.401340</td>
          <td>0.046376</td>
          <td>24.796405</td>
          <td>0.044307</td>
          <td>24.424129</td>
          <td>0.061035</td>
          <td>23.795171</td>
          <td>0.078917</td>
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
          <td>25.945162</td>
          <td>0.241704</td>
          <td>26.534383</td>
          <td>0.142653</td>
          <td>26.099168</td>
          <td>0.086052</td>
          <td>26.133680</td>
          <td>0.143682</td>
          <td>25.638335</td>
          <td>0.176038</td>
          <td>25.306642</td>
          <td>0.286690</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.212946</td>
          <td>0.300518</td>
          <td>27.074498</td>
          <td>0.225323</td>
          <td>26.958137</td>
          <td>0.181080</td>
          <td>26.612581</td>
          <td>0.215688</td>
          <td>25.958584</td>
          <td>0.230315</td>
          <td>25.469920</td>
          <td>0.326793</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.847124</td>
          <td>0.490668</td>
          <td>27.351400</td>
          <td>0.282788</td>
          <td>27.456944</td>
          <td>0.274089</td>
          <td>26.455128</td>
          <td>0.188990</td>
          <td>26.128430</td>
          <td>0.264860</td>
          <td>25.657958</td>
          <td>0.378848</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.236284</td>
          <td>1.211471</td>
          <td>27.193433</td>
          <td>0.248588</td>
          <td>26.445957</td>
          <td>0.116595</td>
          <td>26.189178</td>
          <td>0.150701</td>
          <td>25.725094</td>
          <td>0.189450</td>
          <td>25.933562</td>
          <td>0.467490</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.862253</td>
          <td>0.496186</td>
          <td>26.456407</td>
          <td>0.133380</td>
          <td>25.928897</td>
          <td>0.074046</td>
          <td>25.561434</td>
          <td>0.087259</td>
          <td>25.143897</td>
          <td>0.115029</td>
          <td>25.081018</td>
          <td>0.238398</td>
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
          <td>27.028448</td>
          <td>0.615794</td>
          <td>26.568180</td>
          <td>0.168782</td>
          <td>26.012241</td>
          <td>0.093727</td>
          <td>25.337156</td>
          <td>0.084808</td>
          <td>25.188249</td>
          <td>0.140168</td>
          <td>24.689022</td>
          <td>0.201482</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.553510</td>
          <td>0.377829</td>
          <td>27.833903</td>
          <td>0.426557</td>
          <td>27.869148</td>
          <td>0.658217</td>
          <td>26.862716</td>
          <td>0.539212</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.359041</td>
          <td>0.380565</td>
          <td>26.057307</td>
          <td>0.110857</td>
          <td>24.792088</td>
          <td>0.032582</td>
          <td>23.840240</td>
          <td>0.023201</td>
          <td>23.146439</td>
          <td>0.023824</td>
          <td>22.776457</td>
          <td>0.038778</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.191756</td>
          <td>0.717522</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.520688</td>
          <td>0.355101</td>
          <td>26.515983</td>
          <td>0.249110</td>
          <td>25.558697</td>
          <td>0.204967</td>
          <td>24.957273</td>
          <td>0.268427</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.690203</td>
          <td>0.482327</td>
          <td>25.649148</td>
          <td>0.075979</td>
          <td>25.438167</td>
          <td>0.056458</td>
          <td>24.768536</td>
          <td>0.051287</td>
          <td>24.363204</td>
          <td>0.068096</td>
          <td>23.717935</td>
          <td>0.087214</td>
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
          <td>26.166833</td>
          <td>0.326989</td>
          <td>26.389729</td>
          <td>0.147613</td>
          <td>26.159785</td>
          <td>0.108907</td>
          <td>26.270445</td>
          <td>0.194108</td>
          <td>26.286695</td>
          <td>0.355044</td>
          <td>25.874920</td>
          <td>0.524527</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>30.211812</td>
          <td>2.965760</td>
          <td>26.891928</td>
          <td>0.222420</td>
          <td>27.042666</td>
          <td>0.227685</td>
          <td>26.447566</td>
          <td>0.221428</td>
          <td>27.061497</td>
          <td>0.623339</td>
          <td>24.951295</td>
          <td>0.251528</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.461802</td>
          <td>1.473454</td>
          <td>27.666182</td>
          <td>0.416233</td>
          <td>27.043423</td>
          <td>0.229687</td>
          <td>26.427581</td>
          <td>0.219621</td>
          <td>26.101745</td>
          <td>0.304296</td>
          <td>25.415770</td>
          <td>0.367982</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.741545</td>
          <td>0.511068</td>
          <td>27.158053</td>
          <td>0.283155</td>
          <td>26.897343</td>
          <td>0.206990</td>
          <td>25.933131</td>
          <td>0.147204</td>
          <td>25.236793</td>
          <td>0.150702</td>
          <td>25.443340</td>
          <td>0.382381</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.113027</td>
          <td>1.224159</td>
          <td>26.626432</td>
          <td>0.178896</td>
          <td>26.108883</td>
          <td>0.103038</td>
          <td>25.470671</td>
          <td>0.096366</td>
          <td>24.994839</td>
          <td>0.119744</td>
          <td>25.162491</td>
          <td>0.300294</td>
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
          <td>26.915920</td>
          <td>0.516198</td>
          <td>26.753834</td>
          <td>0.172118</td>
          <td>25.946005</td>
          <td>0.075184</td>
          <td>25.377298</td>
          <td>0.074184</td>
          <td>24.999241</td>
          <td>0.101391</td>
          <td>24.856734</td>
          <td>0.197773</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.337430</td>
          <td>1.880906</td>
          <td>27.238905</td>
          <td>0.229346</td>
          <td>26.859157</td>
          <td>0.264637</td>
          <td>26.705380</td>
          <td>0.418521</td>
          <td>28.281334</td>
          <td>1.885971</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.151809</td>
          <td>0.301853</td>
          <td>26.131279</td>
          <td>0.108000</td>
          <td>24.778127</td>
          <td>0.029047</td>
          <td>23.855394</td>
          <td>0.021176</td>
          <td>23.122377</td>
          <td>0.021108</td>
          <td>22.828298</td>
          <td>0.036502</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.354676</td>
          <td>0.797644</td>
          <td>27.988596</td>
          <td>0.550077</td>
          <td>27.995211</td>
          <td>0.507963</td>
          <td>26.503023</td>
          <td>0.245620</td>
          <td>25.709068</td>
          <td>0.231539</td>
          <td>24.777367</td>
          <td>0.230738</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.661594</td>
          <td>0.427260</td>
          <td>25.846723</td>
          <td>0.078392</td>
          <td>25.445340</td>
          <td>0.048293</td>
          <td>24.767722</td>
          <td>0.043259</td>
          <td>24.334247</td>
          <td>0.056438</td>
          <td>23.650657</td>
          <td>0.069556</td>
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
          <td>26.759560</td>
          <td>0.482010</td>
          <td>26.369291</td>
          <td>0.132400</td>
          <td>26.036104</td>
          <td>0.088093</td>
          <td>26.016974</td>
          <td>0.140869</td>
          <td>26.087256</td>
          <td>0.275483</td>
          <td>26.003932</td>
          <td>0.527367</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.101957</td>
          <td>0.595756</td>
          <td>26.993768</td>
          <td>0.213557</td>
          <td>26.475477</td>
          <td>0.121594</td>
          <td>26.580379</td>
          <td>0.213443</td>
          <td>26.481987</td>
          <td>0.356881</td>
          <td>25.540158</td>
          <td>0.350812</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.874027</td>
          <td>0.198484</td>
          <td>26.587678</td>
          <td>0.138349</td>
          <td>26.307966</td>
          <td>0.175296</td>
          <td>26.483023</td>
          <td>0.367624</td>
          <td>26.413459</td>
          <td>0.687007</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.319772</td>
          <td>0.302399</td>
          <td>26.398350</td>
          <td>0.125308</td>
          <td>25.785753</td>
          <td>0.119652</td>
          <td>25.984365</td>
          <td>0.261899</td>
          <td>25.453240</td>
          <td>0.358780</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.527515</td>
          <td>0.394506</td>
          <td>27.006172</td>
          <td>0.219823</td>
          <td>26.063546</td>
          <td>0.086715</td>
          <td>25.755836</td>
          <td>0.107785</td>
          <td>24.902319</td>
          <td>0.096807</td>
          <td>24.973462</td>
          <td>0.226541</td>
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
