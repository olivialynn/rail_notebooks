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

    <pzflow.flow.Flow at 0x7fcb6a96de40>



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
          <td>inf</td>
          <td>inf</td>
          <td>27.190105</td>
          <td>0.247909</td>
          <td>26.082766</td>
          <td>0.084817</td>
          <td>25.332567</td>
          <td>0.071296</td>
          <td>25.001996</td>
          <td>0.101622</td>
          <td>24.825870</td>
          <td>0.192676</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.072755</td>
          <td>0.495157</td>
          <td>28.496258</td>
          <td>0.606074</td>
          <td>27.299371</td>
          <td>0.375707</td>
          <td>26.764384</td>
          <td>0.437380</td>
          <td>25.461182</td>
          <td>0.324531</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.463672</td>
          <td>0.366513</td>
          <td>26.018656</td>
          <td>0.091082</td>
          <td>24.793535</td>
          <td>0.027124</td>
          <td>23.870074</td>
          <td>0.019724</td>
          <td>23.113045</td>
          <td>0.019338</td>
          <td>22.848373</td>
          <td>0.034107</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.870455</td>
          <td>0.425411</td>
          <td>26.843001</td>
          <td>0.164197</td>
          <td>26.455483</td>
          <td>0.189047</td>
          <td>26.102079</td>
          <td>0.259215</td>
          <td>25.387951</td>
          <td>0.306092</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.130110</td>
          <td>0.281105</td>
          <td>25.887502</td>
          <td>0.081160</td>
          <td>25.505080</td>
          <td>0.050851</td>
          <td>24.774020</td>
          <td>0.043436</td>
          <td>24.406610</td>
          <td>0.060094</td>
          <td>23.689331</td>
          <td>0.071870</td>
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
          <td>27.145968</td>
          <td>0.608923</td>
          <td>26.338144</td>
          <td>0.120397</td>
          <td>26.126405</td>
          <td>0.088140</td>
          <td>26.189152</td>
          <td>0.150697</td>
          <td>25.697770</td>
          <td>0.185128</td>
          <td>25.034180</td>
          <td>0.229334</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.329939</td>
          <td>0.691670</td>
          <td>27.096270</td>
          <td>0.229430</td>
          <td>26.681769</td>
          <td>0.143005</td>
          <td>26.489519</td>
          <td>0.194550</td>
          <td>26.243473</td>
          <td>0.290792</td>
          <td>26.192496</td>
          <td>0.565219</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.650430</td>
          <td>0.854367</td>
          <td>27.040351</td>
          <td>0.219016</td>
          <td>26.847797</td>
          <td>0.164870</td>
          <td>26.387499</td>
          <td>0.178482</td>
          <td>25.637546</td>
          <td>0.175921</td>
          <td>25.877918</td>
          <td>0.448354</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.571512</td>
          <td>0.812113</td>
          <td>28.192000</td>
          <td>0.540335</td>
          <td>26.526883</td>
          <td>0.125089</td>
          <td>26.127409</td>
          <td>0.142908</td>
          <td>25.591383</td>
          <td>0.169151</td>
          <td>25.117219</td>
          <td>0.245622</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.846530</td>
          <td>0.490452</td>
          <td>26.529316</td>
          <td>0.142032</td>
          <td>26.085807</td>
          <td>0.085045</td>
          <td>25.477584</td>
          <td>0.081043</td>
          <td>25.168734</td>
          <td>0.117543</td>
          <td>25.301125</td>
          <td>0.285414</td>
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
          <td>26.512717</td>
          <td>0.421943</td>
          <td>26.717525</td>
          <td>0.191528</td>
          <td>25.846255</td>
          <td>0.080990</td>
          <td>25.388250</td>
          <td>0.088709</td>
          <td>24.971456</td>
          <td>0.116174</td>
          <td>24.991291</td>
          <td>0.258875</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.537288</td>
          <td>0.768430</td>
          <td>27.338579</td>
          <td>0.289067</td>
          <td>27.326583</td>
          <td>0.444418</td>
          <td>26.955403</td>
          <td>0.576425</td>
          <td>27.396279</td>
          <td>1.349329</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.547656</td>
          <td>0.881900</td>
          <td>25.982685</td>
          <td>0.103872</td>
          <td>24.751119</td>
          <td>0.031429</td>
          <td>23.900903</td>
          <td>0.024447</td>
          <td>23.177527</td>
          <td>0.024472</td>
          <td>22.841698</td>
          <td>0.041083</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.580550</td>
          <td>0.406944</td>
          <td>27.763153</td>
          <td>0.428306</td>
          <td>26.470801</td>
          <td>0.240010</td>
          <td>25.704190</td>
          <td>0.231383</td>
          <td>24.891848</td>
          <td>0.254446</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.152728</td>
          <td>0.318757</td>
          <td>25.717935</td>
          <td>0.080726</td>
          <td>25.392421</td>
          <td>0.054213</td>
          <td>24.801041</td>
          <td>0.052788</td>
          <td>24.449864</td>
          <td>0.073520</td>
          <td>23.611503</td>
          <td>0.079408</td>
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
          <td>26.433767</td>
          <td>0.402794</td>
          <td>26.335252</td>
          <td>0.140861</td>
          <td>26.064354</td>
          <td>0.100189</td>
          <td>26.564761</td>
          <td>0.248002</td>
          <td>25.920851</td>
          <td>0.264837</td>
          <td>25.249259</td>
          <td>0.325180</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.381482</td>
          <td>0.784547</td>
          <td>27.209974</td>
          <td>0.288675</td>
          <td>26.864827</td>
          <td>0.196245</td>
          <td>26.805741</td>
          <td>0.296921</td>
          <td>26.146099</td>
          <td>0.312838</td>
          <td>25.145101</td>
          <td>0.294490</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.152672</td>
          <td>0.676385</td>
          <td>26.785466</td>
          <td>0.205025</td>
          <td>26.789282</td>
          <td>0.185663</td>
          <td>26.403672</td>
          <td>0.215288</td>
          <td>26.564309</td>
          <td>0.436716</td>
          <td>25.516431</td>
          <td>0.397865</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.643076</td>
          <td>0.475186</td>
          <td>28.190280</td>
          <td>0.620137</td>
          <td>26.694313</td>
          <td>0.174417</td>
          <td>25.726777</td>
          <td>0.123176</td>
          <td>25.639982</td>
          <td>0.212059</td>
          <td>25.749315</td>
          <td>0.482471</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>25.840725</td>
          <td>0.249368</td>
          <td>26.246203</td>
          <td>0.129181</td>
          <td>26.112404</td>
          <td>0.103356</td>
          <td>25.663810</td>
          <td>0.114090</td>
          <td>25.372439</td>
          <td>0.165753</td>
          <td>24.731757</td>
          <td>0.210887</td>
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
          <td>26.982050</td>
          <td>0.541668</td>
          <td>26.471923</td>
          <td>0.135194</td>
          <td>26.100629</td>
          <td>0.086174</td>
          <td>25.368418</td>
          <td>0.073604</td>
          <td>24.978665</td>
          <td>0.099580</td>
          <td>25.070884</td>
          <td>0.236440</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.369952</td>
          <td>0.613981</td>
          <td>27.018391</td>
          <td>0.190713</td>
          <td>27.333314</td>
          <td>0.386078</td>
          <td>26.246150</td>
          <td>0.291674</td>
          <td>27.780573</td>
          <td>1.491828</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.529936</td>
          <td>0.825336</td>
          <td>26.053900</td>
          <td>0.100942</td>
          <td>24.797655</td>
          <td>0.029549</td>
          <td>23.859051</td>
          <td>0.021242</td>
          <td>23.153725</td>
          <td>0.021680</td>
          <td>22.901010</td>
          <td>0.038926</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.516567</td>
          <td>1.548113</td>
          <td>28.097598</td>
          <td>0.594695</td>
          <td>28.560866</td>
          <td>0.754874</td>
          <td>27.005519</td>
          <td>0.367708</td>
          <td>27.068718</td>
          <td>0.656596</td>
          <td>24.785376</td>
          <td>0.232274</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.854969</td>
          <td>0.493932</td>
          <td>25.786793</td>
          <td>0.074356</td>
          <td>25.428385</td>
          <td>0.047572</td>
          <td>24.842320</td>
          <td>0.046220</td>
          <td>24.306625</td>
          <td>0.055071</td>
          <td>23.651446</td>
          <td>0.069605</td>
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
          <td>26.776094</td>
          <td>0.487958</td>
          <td>26.190986</td>
          <td>0.113431</td>
          <td>26.117636</td>
          <td>0.094637</td>
          <td>26.168307</td>
          <td>0.160403</td>
          <td>25.794272</td>
          <td>0.216420</td>
          <td>24.875320</td>
          <td>0.217080</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.251020</td>
          <td>0.661159</td>
          <td>27.061700</td>
          <td>0.225979</td>
          <td>26.830298</td>
          <td>0.165046</td>
          <td>26.730859</td>
          <td>0.241838</td>
          <td>26.421292</td>
          <td>0.340228</td>
          <td>25.398199</td>
          <td>0.313461</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.556769</td>
          <td>1.465792</td>
          <td>27.717772</td>
          <td>0.392771</td>
          <td>26.890682</td>
          <td>0.179280</td>
          <td>26.516605</td>
          <td>0.209005</td>
          <td>25.491803</td>
          <td>0.162929</td>
          <td>25.923355</td>
          <td>0.484336</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.104640</td>
          <td>1.185934</td>
          <td>27.186312</td>
          <td>0.271473</td>
          <td>26.371614</td>
          <td>0.122435</td>
          <td>25.809936</td>
          <td>0.122192</td>
          <td>25.935063</td>
          <td>0.251530</td>
          <td>25.656483</td>
          <td>0.419879</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.288462</td>
          <td>0.327200</td>
          <td>26.480271</td>
          <td>0.140765</td>
          <td>26.230891</td>
          <td>0.100444</td>
          <td>25.579831</td>
          <td>0.092382</td>
          <td>25.312631</td>
          <td>0.138353</td>
          <td>24.899463</td>
          <td>0.213003</td>
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
