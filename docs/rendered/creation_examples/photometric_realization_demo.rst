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

    <pzflow.flow.Flow at 0x7f694328a6b0>



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
          <td>26.973595</td>
          <td>0.538318</td>
          <td>26.892144</td>
          <td>0.193456</td>
          <td>26.015025</td>
          <td>0.079899</td>
          <td>25.344626</td>
          <td>0.072061</td>
          <td>25.040629</td>
          <td>0.105116</td>
          <td>25.316331</td>
          <td>0.288945</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.936266</td>
          <td>1.019222</td>
          <td>28.374447</td>
          <td>0.615520</td>
          <td>27.371555</td>
          <td>0.255629</td>
          <td>27.047413</td>
          <td>0.307896</td>
          <td>27.081517</td>
          <td>0.553059</td>
          <td>26.187266</td>
          <td>0.563100</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.938277</td>
          <td>0.524663</td>
          <td>25.942913</td>
          <td>0.085217</td>
          <td>24.804253</td>
          <td>0.027379</td>
          <td>23.862491</td>
          <td>0.019598</td>
          <td>23.132239</td>
          <td>0.019653</td>
          <td>22.864224</td>
          <td>0.034588</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.144445</td>
          <td>1.896434</td>
          <td>31.136756</td>
          <td>2.574378</td>
          <td>27.417309</td>
          <td>0.265379</td>
          <td>26.917420</td>
          <td>0.277240</td>
          <td>25.782093</td>
          <td>0.198765</td>
          <td>25.720395</td>
          <td>0.397604</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.415332</td>
          <td>0.352906</td>
          <td>25.775357</td>
          <td>0.073518</td>
          <td>25.461445</td>
          <td>0.048918</td>
          <td>24.846392</td>
          <td>0.046318</td>
          <td>24.355437</td>
          <td>0.057426</td>
          <td>23.745074</td>
          <td>0.075501</td>
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
          <td>26.228689</td>
          <td>0.304337</td>
          <td>26.306567</td>
          <td>0.117139</td>
          <td>26.035299</td>
          <td>0.081341</td>
          <td>26.100065</td>
          <td>0.139581</td>
          <td>26.134187</td>
          <td>0.266107</td>
          <td>25.406346</td>
          <td>0.310636</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.633265</td>
          <td>0.845055</td>
          <td>27.429053</td>
          <td>0.301068</td>
          <td>27.057542</td>
          <td>0.196928</td>
          <td>26.583058</td>
          <td>0.210435</td>
          <td>26.829294</td>
          <td>0.459325</td>
          <td>25.895951</td>
          <td>0.454485</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.713886</td>
          <td>0.444150</td>
          <td>27.210773</td>
          <td>0.252154</td>
          <td>26.980380</td>
          <td>0.184521</td>
          <td>26.456924</td>
          <td>0.189277</td>
          <td>26.044581</td>
          <td>0.247268</td>
          <td>24.866592</td>
          <td>0.199392</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.929733</td>
          <td>0.199667</td>
          <td>26.573280</td>
          <td>0.130220</td>
          <td>25.792710</td>
          <td>0.106887</td>
          <td>25.785622</td>
          <td>0.199355</td>
          <td>25.785714</td>
          <td>0.418043</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.258809</td>
          <td>0.311761</td>
          <td>26.732181</td>
          <td>0.168960</td>
          <td>26.062545</td>
          <td>0.083319</td>
          <td>25.688926</td>
          <td>0.097604</td>
          <td>24.909112</td>
          <td>0.093673</td>
          <td>25.131777</td>
          <td>0.248582</td>
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
          <td>28.079836</td>
          <td>1.196033</td>
          <td>26.943345</td>
          <td>0.231290</td>
          <td>25.894625</td>
          <td>0.084517</td>
          <td>25.440074</td>
          <td>0.092843</td>
          <td>24.898460</td>
          <td>0.109013</td>
          <td>24.921465</td>
          <td>0.244450</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.024158</td>
          <td>1.159307</td>
          <td>27.447242</td>
          <td>0.347692</td>
          <td>27.808299</td>
          <td>0.418309</td>
          <td>26.799311</td>
          <td>0.294278</td>
          <td>26.516943</td>
          <td>0.416692</td>
          <td>25.936362</td>
          <td>0.538906</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.055651</td>
          <td>0.636193</td>
          <td>25.884530</td>
          <td>0.095326</td>
          <td>24.737870</td>
          <td>0.031066</td>
          <td>23.829538</td>
          <td>0.022988</td>
          <td>23.127487</td>
          <td>0.023438</td>
          <td>22.853965</td>
          <td>0.041531</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.401433</td>
          <td>2.279513</td>
          <td>28.087922</td>
          <td>0.592182</td>
          <td>29.173119</td>
          <td>1.106745</td>
          <td>27.019874</td>
          <td>0.373064</td>
          <td>25.878027</td>
          <td>0.266928</td>
          <td>25.032708</td>
          <td>0.285384</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.626453</td>
          <td>0.459920</td>
          <td>25.823412</td>
          <td>0.088571</td>
          <td>25.424354</td>
          <td>0.055771</td>
          <td>24.802668</td>
          <td>0.052864</td>
          <td>24.427554</td>
          <td>0.072084</td>
          <td>23.627980</td>
          <td>0.080571</td>
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
          <td>26.824567</td>
          <td>0.539374</td>
          <td>26.305223</td>
          <td>0.137265</td>
          <td>26.066594</td>
          <td>0.100386</td>
          <td>26.031731</td>
          <td>0.158511</td>
          <td>25.836051</td>
          <td>0.247056</td>
          <td>25.018150</td>
          <td>0.269971</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.527347</td>
          <td>1.516865</td>
          <td>27.236967</td>
          <td>0.295029</td>
          <td>26.854230</td>
          <td>0.194503</td>
          <td>26.594406</td>
          <td>0.250017</td>
          <td>26.805890</td>
          <td>0.519040</td>
          <td>26.813559</td>
          <td>0.971858</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.378467</td>
          <td>0.332686</td>
          <td>27.013636</td>
          <td>0.224078</td>
          <td>26.421469</td>
          <td>0.218506</td>
          <td>26.776550</td>
          <td>0.511655</td>
          <td>25.375284</td>
          <td>0.356506</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.616008</td>
          <td>1.602492</td>
          <td>27.188317</td>
          <td>0.290167</td>
          <td>26.509361</td>
          <td>0.148936</td>
          <td>25.809552</td>
          <td>0.132332</td>
          <td>25.512237</td>
          <td>0.190498</td>
          <td>25.902282</td>
          <td>0.539833</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.555341</td>
          <td>0.438708</td>
          <td>26.462874</td>
          <td>0.155646</td>
          <td>26.062406</td>
          <td>0.098930</td>
          <td>25.547250</td>
          <td>0.103052</td>
          <td>25.305530</td>
          <td>0.156547</td>
          <td>24.732626</td>
          <td>0.211040</td>
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
          <td>27.005797</td>
          <td>0.551048</td>
          <td>27.231096</td>
          <td>0.256418</td>
          <td>26.117982</td>
          <td>0.087501</td>
          <td>25.437816</td>
          <td>0.078259</td>
          <td>25.034228</td>
          <td>0.104543</td>
          <td>25.402318</td>
          <td>0.309675</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.122638</td>
          <td>0.599294</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.252354</td>
          <td>0.995385</td>
          <td>26.843361</td>
          <td>0.261242</td>
          <td>26.118370</td>
          <td>0.262923</td>
          <td>28.007381</td>
          <td>1.665545</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.039782</td>
          <td>0.275769</td>
          <td>25.963336</td>
          <td>0.093245</td>
          <td>24.790276</td>
          <td>0.029358</td>
          <td>23.858916</td>
          <td>0.021239</td>
          <td>23.129055</td>
          <td>0.021228</td>
          <td>22.829677</td>
          <td>0.036547</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.572023</td>
          <td>0.916061</td>
          <td>30.050513</td>
          <td>1.831883</td>
          <td>29.085800</td>
          <td>1.049240</td>
          <td>27.536178</td>
          <td>0.548283</td>
          <td>25.851309</td>
          <td>0.260301</td>
          <td>25.347604</td>
          <td>0.365429</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.332495</td>
          <td>0.693401</td>
          <td>25.660326</td>
          <td>0.066495</td>
          <td>25.410572</td>
          <td>0.046825</td>
          <td>24.788010</td>
          <td>0.044045</td>
          <td>24.407755</td>
          <td>0.060241</td>
          <td>23.661529</td>
          <td>0.070228</td>
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
          <td>26.820675</td>
          <td>0.504288</td>
          <td>26.300748</td>
          <td>0.124776</td>
          <td>26.110510</td>
          <td>0.094047</td>
          <td>26.340118</td>
          <td>0.185622</td>
          <td>26.303776</td>
          <td>0.327846</td>
          <td>25.342318</td>
          <td>0.317858</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.526483</td>
          <td>0.388761</td>
          <td>26.925488</td>
          <td>0.201698</td>
          <td>26.721258</td>
          <td>0.150346</td>
          <td>26.258870</td>
          <td>0.162680</td>
          <td>26.110602</td>
          <td>0.265052</td>
          <td>25.954110</td>
          <td>0.481649</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.457344</td>
          <td>0.320171</td>
          <td>26.997643</td>
          <td>0.196227</td>
          <td>26.310112</td>
          <td>0.175616</td>
          <td>25.838048</td>
          <td>0.218214</td>
          <td>25.144356</td>
          <td>0.263328</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.051351</td>
          <td>0.608436</td>
          <td>27.226851</td>
          <td>0.280559</td>
          <td>26.564583</td>
          <td>0.144656</td>
          <td>25.764426</td>
          <td>0.117453</td>
          <td>25.340965</td>
          <td>0.152665</td>
          <td>25.533348</td>
          <td>0.381910</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.281314</td>
          <td>0.683414</td>
          <td>26.557434</td>
          <td>0.150413</td>
          <td>26.182609</td>
          <td>0.096281</td>
          <td>25.750495</td>
          <td>0.107284</td>
          <td>25.058475</td>
          <td>0.110976</td>
          <td>24.911229</td>
          <td>0.215105</td>
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
