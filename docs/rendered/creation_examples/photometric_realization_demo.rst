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

    <pzflow.flow.Flow at 0x7f310b877dc0>



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
          <td>26.644600</td>
          <td>0.421409</td>
          <td>26.672606</td>
          <td>0.160595</td>
          <td>25.992141</td>
          <td>0.078301</td>
          <td>25.305130</td>
          <td>0.069586</td>
          <td>24.933634</td>
          <td>0.095712</td>
          <td>24.967989</td>
          <td>0.217053</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.299923</td>
          <td>1.113454</td>
          <td>27.315938</td>
          <td>0.244207</td>
          <td>26.992680</td>
          <td>0.294646</td>
          <td>26.054685</td>
          <td>0.249331</td>
          <td>27.882388</td>
          <td>1.567903</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.222545</td>
          <td>0.642430</td>
          <td>26.060105</td>
          <td>0.094456</td>
          <td>24.826575</td>
          <td>0.027919</td>
          <td>23.852571</td>
          <td>0.019435</td>
          <td>23.155887</td>
          <td>0.020051</td>
          <td>22.851388</td>
          <td>0.034198</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.371274</td>
          <td>0.711326</td>
          <td>28.974477</td>
          <td>0.916255</td>
          <td>27.354736</td>
          <td>0.252126</td>
          <td>26.522923</td>
          <td>0.200093</td>
          <td>26.078090</td>
          <td>0.254170</td>
          <td>24.995270</td>
          <td>0.222040</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.973718</td>
          <td>0.247447</td>
          <td>25.847001</td>
          <td>0.078315</td>
          <td>25.425864</td>
          <td>0.047397</td>
          <td>24.834879</td>
          <td>0.045847</td>
          <td>24.405685</td>
          <td>0.060045</td>
          <td>23.696081</td>
          <td>0.072301</td>
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
          <td>26.598814</td>
          <td>0.406912</td>
          <td>26.290346</td>
          <td>0.115498</td>
          <td>26.324534</td>
          <td>0.104875</td>
          <td>26.167896</td>
          <td>0.147972</td>
          <td>26.009238</td>
          <td>0.240170</td>
          <td>25.982213</td>
          <td>0.484753</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.759358</td>
          <td>0.459609</td>
          <td>27.149295</td>
          <td>0.239714</td>
          <td>27.187158</td>
          <td>0.219495</td>
          <td>26.341257</td>
          <td>0.171609</td>
          <td>25.774296</td>
          <td>0.197467</td>
          <td>25.040022</td>
          <td>0.230447</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.056405</td>
          <td>0.571412</td>
          <td>27.475422</td>
          <td>0.312467</td>
          <td>26.882809</td>
          <td>0.169862</td>
          <td>26.364080</td>
          <td>0.174970</td>
          <td>25.937385</td>
          <td>0.226300</td>
          <td>25.193701</td>
          <td>0.261531</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.723823</td>
          <td>0.447492</td>
          <td>27.538671</td>
          <td>0.328616</td>
          <td>26.378591</td>
          <td>0.109946</td>
          <td>25.923951</td>
          <td>0.119841</td>
          <td>26.208909</td>
          <td>0.282776</td>
          <td>25.906488</td>
          <td>0.458099</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.270439</td>
          <td>0.664063</td>
          <td>26.478680</td>
          <td>0.135969</td>
          <td>26.073606</td>
          <td>0.084136</td>
          <td>25.742229</td>
          <td>0.102270</td>
          <td>25.236891</td>
          <td>0.124713</td>
          <td>25.113971</td>
          <td>0.244966</td>
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
          <td>27.032960</td>
          <td>0.617748</td>
          <td>26.449648</td>
          <td>0.152541</td>
          <td>25.948707</td>
          <td>0.088637</td>
          <td>25.294859</td>
          <td>0.081705</td>
          <td>24.998989</td>
          <td>0.118989</td>
          <td>24.952081</td>
          <td>0.250684</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.499175</td>
          <td>1.493172</td>
          <td>28.160087</td>
          <td>0.593488</td>
          <td>27.658838</td>
          <td>0.372747</td>
          <td>27.305876</td>
          <td>0.437510</td>
          <td>26.400340</td>
          <td>0.380891</td>
          <td>29.255213</td>
          <td>2.915207</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.430961</td>
          <td>0.402289</td>
          <td>25.907400</td>
          <td>0.097254</td>
          <td>24.806481</td>
          <td>0.032997</td>
          <td>23.865383</td>
          <td>0.023708</td>
          <td>23.180402</td>
          <td>0.024533</td>
          <td>22.814076</td>
          <td>0.040091</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.772369</td>
          <td>1.036608</td>
          <td>27.732228</td>
          <td>0.456625</td>
          <td>27.808290</td>
          <td>0.443218</td>
          <td>26.358458</td>
          <td>0.218662</td>
          <td>25.435340</td>
          <td>0.184754</td>
          <td>25.759109</td>
          <td>0.501200</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.441791</td>
          <td>0.814217</td>
          <td>25.880582</td>
          <td>0.093128</td>
          <td>25.373748</td>
          <td>0.053322</td>
          <td>24.776019</td>
          <td>0.051629</td>
          <td>24.298122</td>
          <td>0.064283</td>
          <td>23.730722</td>
          <td>0.088201</td>
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
          <td>27.097075</td>
          <td>0.654209</td>
          <td>26.334610</td>
          <td>0.140783</td>
          <td>26.327146</td>
          <td>0.125974</td>
          <td>25.971221</td>
          <td>0.150505</td>
          <td>26.156515</td>
          <td>0.320302</td>
          <td>25.104075</td>
          <td>0.289458</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.248737</td>
          <td>0.718325</td>
          <td>26.936585</td>
          <td>0.230816</td>
          <td>26.861819</td>
          <td>0.195749</td>
          <td>26.569604</td>
          <td>0.244968</td>
          <td>25.844121</td>
          <td>0.244817</td>
          <td>25.660888</td>
          <td>0.440867</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.484664</td>
          <td>0.416518</td>
          <td>27.324550</td>
          <td>0.318731</td>
          <td>26.753221</td>
          <td>0.180084</td>
          <td>26.319627</td>
          <td>0.200665</td>
          <td>26.281152</td>
          <td>0.350918</td>
          <td>27.422287</td>
          <td>1.378440</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.404695</td>
          <td>0.344843</td>
          <td>26.569435</td>
          <td>0.156804</td>
          <td>25.968590</td>
          <td>0.151753</td>
          <td>26.218360</td>
          <td>0.339592</td>
          <td>25.201552</td>
          <td>0.316100</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.623966</td>
          <td>0.918851</td>
          <td>26.534293</td>
          <td>0.165429</td>
          <td>26.239379</td>
          <td>0.115467</td>
          <td>25.743277</td>
          <td>0.122253</td>
          <td>25.148854</td>
          <td>0.136828</td>
          <td>25.250671</td>
          <td>0.322246</td>
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
          <td>26.566814</td>
          <td>0.397057</td>
          <td>26.764514</td>
          <td>0.173686</td>
          <td>25.926686</td>
          <td>0.073911</td>
          <td>25.273156</td>
          <td>0.067652</td>
          <td>25.074778</td>
          <td>0.108314</td>
          <td>25.448536</td>
          <td>0.321320</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.231738</td>
          <td>1.208917</td>
          <td>28.824266</td>
          <td>0.833722</td>
          <td>27.361875</td>
          <td>0.253832</td>
          <td>26.648075</td>
          <td>0.222371</td>
          <td>26.648245</td>
          <td>0.400580</td>
          <td>25.823863</td>
          <td>0.430741</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.872342</td>
          <td>0.240489</td>
          <td>25.926130</td>
          <td>0.090250</td>
          <td>24.790359</td>
          <td>0.029360</td>
          <td>23.896141</td>
          <td>0.021926</td>
          <td>23.113626</td>
          <td>0.020951</td>
          <td>22.819548</td>
          <td>0.036221</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.166915</td>
          <td>2.073921</td>
          <td>28.896301</td>
          <td>1.004162</td>
          <td>27.601643</td>
          <td>0.377088</td>
          <td>26.618687</td>
          <td>0.270023</td>
          <td>26.168728</td>
          <td>0.336096</td>
          <td>25.616614</td>
          <td>0.449281</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.138837</td>
          <td>0.283354</td>
          <td>25.824482</td>
          <td>0.076870</td>
          <td>25.356966</td>
          <td>0.044650</td>
          <td>24.864300</td>
          <td>0.047131</td>
          <td>24.419858</td>
          <td>0.060891</td>
          <td>23.768925</td>
          <td>0.077224</td>
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
          <td>26.986877</td>
          <td>0.568968</td>
          <td>26.394378</td>
          <td>0.135298</td>
          <td>26.085087</td>
          <td>0.091970</td>
          <td>26.108387</td>
          <td>0.152384</td>
          <td>25.814902</td>
          <td>0.220172</td>
          <td>25.488153</td>
          <td>0.356735</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.942931</td>
          <td>1.743295</td>
          <td>27.204929</td>
          <td>0.254325</td>
          <td>26.856880</td>
          <td>0.168827</td>
          <td>26.200809</td>
          <td>0.154800</td>
          <td>26.378896</td>
          <td>0.328993</td>
          <td>27.210801</td>
          <td>1.108555</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.647228</td>
          <td>0.371856</td>
          <td>26.808186</td>
          <td>0.167143</td>
          <td>26.057176</td>
          <td>0.141450</td>
          <td>26.096223</td>
          <td>0.269958</td>
          <td>26.181603</td>
          <td>0.584451</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.811314</td>
          <td>0.443732</td>
          <td>26.557890</td>
          <td>0.143825</td>
          <td>25.800066</td>
          <td>0.121149</td>
          <td>25.611259</td>
          <td>0.192106</td>
          <td>25.348559</td>
          <td>0.330349</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.685747</td>
          <td>0.890977</td>
          <td>26.768991</td>
          <td>0.180129</td>
          <td>26.188621</td>
          <td>0.096790</td>
          <td>25.640590</td>
          <td>0.097443</td>
          <td>25.098468</td>
          <td>0.114912</td>
          <td>24.949362</td>
          <td>0.222049</td>
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
