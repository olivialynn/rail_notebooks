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

    <pzflow.flow.Flow at 0x7fc85c53eef0>



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
          <td>27.430513</td>
          <td>0.740182</td>
          <td>26.651666</td>
          <td>0.157747</td>
          <td>26.058355</td>
          <td>0.083012</td>
          <td>25.309365</td>
          <td>0.069847</td>
          <td>25.165593</td>
          <td>0.117222</td>
          <td>25.097263</td>
          <td>0.241616</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.723793</td>
          <td>0.380036</td>
          <td>27.340409</td>
          <td>0.249176</td>
          <td>27.075386</td>
          <td>0.314865</td>
          <td>26.784339</td>
          <td>0.444034</td>
          <td>26.159273</td>
          <td>0.551863</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.500552</td>
          <td>0.377189</td>
          <td>25.950153</td>
          <td>0.085761</td>
          <td>24.767359</td>
          <td>0.026512</td>
          <td>23.891945</td>
          <td>0.020093</td>
          <td>23.119942</td>
          <td>0.019450</td>
          <td>22.801941</td>
          <td>0.032739</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.882815</td>
          <td>1.685230</td>
          <td>27.616259</td>
          <td>0.349398</td>
          <td>27.404152</td>
          <td>0.262542</td>
          <td>26.797967</td>
          <td>0.251469</td>
          <td>25.814319</td>
          <td>0.204215</td>
          <td>24.973998</td>
          <td>0.218143</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.212074</td>
          <td>0.637769</td>
          <td>25.691316</td>
          <td>0.068258</td>
          <td>25.453354</td>
          <td>0.048568</td>
          <td>24.833213</td>
          <td>0.045779</td>
          <td>24.432749</td>
          <td>0.061503</td>
          <td>23.649088</td>
          <td>0.069356</td>
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
          <td>26.657236</td>
          <td>0.425484</td>
          <td>26.377136</td>
          <td>0.124539</td>
          <td>26.129893</td>
          <td>0.088411</td>
          <td>26.124992</td>
          <td>0.142611</td>
          <td>26.233641</td>
          <td>0.288492</td>
          <td>25.525292</td>
          <td>0.341448</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.721422</td>
          <td>0.446682</td>
          <td>26.658395</td>
          <td>0.158657</td>
          <td>27.003867</td>
          <td>0.188219</td>
          <td>26.296857</td>
          <td>0.165242</td>
          <td>26.105404</td>
          <td>0.259921</td>
          <td>26.132047</td>
          <td>0.541100</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.270275</td>
          <td>1.234434</td>
          <td>27.611294</td>
          <td>0.348035</td>
          <td>26.910169</td>
          <td>0.173860</td>
          <td>26.359255</td>
          <td>0.174254</td>
          <td>26.006075</td>
          <td>0.239544</td>
          <td>25.896871</td>
          <td>0.454799</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.276266</td>
          <td>0.666731</td>
          <td>27.595165</td>
          <td>0.343640</td>
          <td>26.499585</td>
          <td>0.122160</td>
          <td>25.677451</td>
          <td>0.096626</td>
          <td>26.091771</td>
          <td>0.257036</td>
          <td>25.311073</td>
          <td>0.287719</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.077805</td>
          <td>0.269423</td>
          <td>26.385885</td>
          <td>0.125487</td>
          <td>26.001372</td>
          <td>0.078942</td>
          <td>25.739494</td>
          <td>0.102026</td>
          <td>25.380504</td>
          <td>0.141199</td>
          <td>24.934540</td>
          <td>0.211076</td>
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
          <td>26.393112</td>
          <td>0.384910</td>
          <td>26.537991</td>
          <td>0.164499</td>
          <td>26.026233</td>
          <td>0.094885</td>
          <td>25.234836</td>
          <td>0.077491</td>
          <td>25.173626</td>
          <td>0.138413</td>
          <td>24.980771</td>
          <td>0.256654</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.230359</td>
          <td>0.338966</td>
          <td>27.755581</td>
          <td>0.441156</td>
          <td>27.787491</td>
          <td>0.411703</td>
          <td>27.015085</td>
          <td>0.349472</td>
          <td>27.013697</td>
          <td>0.600809</td>
          <td>26.339991</td>
          <td>0.715017</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.624405</td>
          <td>0.925225</td>
          <td>25.993369</td>
          <td>0.104846</td>
          <td>24.808387</td>
          <td>0.033053</td>
          <td>23.873430</td>
          <td>0.023874</td>
          <td>23.130874</td>
          <td>0.023507</td>
          <td>22.921943</td>
          <td>0.044109</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.605607</td>
          <td>0.414831</td>
          <td>28.103484</td>
          <td>0.551289</td>
          <td>26.487275</td>
          <td>0.243293</td>
          <td>26.576303</td>
          <td>0.461690</td>
          <td>24.904688</td>
          <td>0.257138</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.313066</td>
          <td>0.361752</td>
          <td>25.729285</td>
          <td>0.081536</td>
          <td>25.535200</td>
          <td>0.061531</td>
          <td>24.779300</td>
          <td>0.051779</td>
          <td>24.400697</td>
          <td>0.070392</td>
          <td>23.749376</td>
          <td>0.089660</td>
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
          <td>26.505962</td>
          <td>0.425649</td>
          <td>26.445414</td>
          <td>0.154827</td>
          <td>26.093559</td>
          <td>0.102783</td>
          <td>26.005716</td>
          <td>0.155021</td>
          <td>25.631961</td>
          <td>0.208566</td>
          <td>25.072546</td>
          <td>0.282169</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.869404</td>
          <td>0.218291</td>
          <td>27.058629</td>
          <td>0.230719</td>
          <td>26.210495</td>
          <td>0.181467</td>
          <td>25.930397</td>
          <td>0.262773</td>
          <td>25.148658</td>
          <td>0.295336</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.886347</td>
          <td>0.561065</td>
          <td>27.040370</td>
          <td>0.253270</td>
          <td>26.868731</td>
          <td>0.198519</td>
          <td>26.794798</td>
          <td>0.296729</td>
          <td>25.829972</td>
          <td>0.243941</td>
          <td>25.411307</td>
          <td>0.366702</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.870339</td>
          <td>0.492391</td>
          <td>26.790486</td>
          <td>0.189209</td>
          <td>25.664581</td>
          <td>0.116696</td>
          <td>25.646749</td>
          <td>0.213260</td>
          <td>24.895934</td>
          <td>0.246710</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.640725</td>
          <td>0.467796</td>
          <td>26.545704</td>
          <td>0.167044</td>
          <td>26.155481</td>
          <td>0.107322</td>
          <td>25.564283</td>
          <td>0.104598</td>
          <td>25.101117</td>
          <td>0.131300</td>
          <td>24.885997</td>
          <td>0.239713</td>
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
          <td>26.854966</td>
          <td>0.493560</td>
          <td>26.541345</td>
          <td>0.143526</td>
          <td>25.999060</td>
          <td>0.078792</td>
          <td>25.217736</td>
          <td>0.064410</td>
          <td>24.946227</td>
          <td>0.096788</td>
          <td>24.892422</td>
          <td>0.203789</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.901240</td>
          <td>0.510914</td>
          <td>28.239208</td>
          <td>0.559454</td>
          <td>27.641369</td>
          <td>0.318269</td>
          <td>28.652739</td>
          <td>0.968038</td>
          <td>27.834045</td>
          <td>0.918192</td>
          <td>26.107162</td>
          <td>0.531837</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>28.048137</td>
          <td>1.131258</td>
          <td>26.051154</td>
          <td>0.100700</td>
          <td>24.795271</td>
          <td>0.029487</td>
          <td>23.876047</td>
          <td>0.021552</td>
          <td>23.139765</td>
          <td>0.021423</td>
          <td>22.770938</td>
          <td>0.034700</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.358684</td>
          <td>0.799729</td>
          <td>28.289450</td>
          <td>0.679728</td>
          <td>27.638375</td>
          <td>0.387987</td>
          <td>26.758779</td>
          <td>0.302424</td>
          <td>26.231237</td>
          <td>0.353078</td>
          <td>25.889519</td>
          <td>0.549583</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.542927</td>
          <td>0.390114</td>
          <td>25.639573</td>
          <td>0.065286</td>
          <td>25.520592</td>
          <td>0.051631</td>
          <td>24.780972</td>
          <td>0.043771</td>
          <td>24.455257</td>
          <td>0.062833</td>
          <td>23.656629</td>
          <td>0.069925</td>
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
          <td>26.819205</td>
          <td>0.503743</td>
          <td>26.660953</td>
          <td>0.170012</td>
          <td>26.087611</td>
          <td>0.092174</td>
          <td>25.775376</td>
          <td>0.114260</td>
          <td>25.621226</td>
          <td>0.187163</td>
          <td>25.421003</td>
          <td>0.338358</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.264373</td>
          <td>0.667265</td>
          <td>27.074030</td>
          <td>0.228303</td>
          <td>26.912358</td>
          <td>0.176978</td>
          <td>26.143307</td>
          <td>0.147349</td>
          <td>26.143798</td>
          <td>0.272322</td>
          <td>25.342061</td>
          <td>0.299665</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.443316</td>
          <td>1.382927</td>
          <td>27.644811</td>
          <td>0.371156</td>
          <td>26.758617</td>
          <td>0.160221</td>
          <td>26.281818</td>
          <td>0.171445</td>
          <td>25.973666</td>
          <td>0.244167</td>
          <td>25.630963</td>
          <td>0.387979</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.114275</td>
          <td>0.635853</td>
          <td>27.984376</td>
          <td>0.504872</td>
          <td>26.649505</td>
          <td>0.155591</td>
          <td>25.653444</td>
          <td>0.106621</td>
          <td>25.583964</td>
          <td>0.187734</td>
          <td>25.681527</td>
          <td>0.427970</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.534879</td>
          <td>0.396751</td>
          <td>26.513134</td>
          <td>0.144800</td>
          <td>25.910690</td>
          <td>0.075777</td>
          <td>25.722238</td>
          <td>0.104666</td>
          <td>25.256589</td>
          <td>0.131816</td>
          <td>24.935250</td>
          <td>0.219456</td>
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
