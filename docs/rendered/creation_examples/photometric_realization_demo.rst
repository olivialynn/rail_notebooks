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

    <pzflow.flow.Flow at 0x7f6f40e0c4f0>



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
          <td>26.284580</td>
          <td>0.318237</td>
          <td>26.765735</td>
          <td>0.173847</td>
          <td>26.068087</td>
          <td>0.083727</td>
          <td>25.460427</td>
          <td>0.079826</td>
          <td>25.026187</td>
          <td>0.103797</td>
          <td>25.102336</td>
          <td>0.242629</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.836261</td>
          <td>0.370850</td>
          <td>26.919806</td>
          <td>0.277778</td>
          <td>25.880415</td>
          <td>0.215821</td>
          <td>25.681448</td>
          <td>0.385816</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.014282</td>
          <td>0.554389</td>
          <td>26.009795</td>
          <td>0.090377</td>
          <td>24.741173</td>
          <td>0.025915</td>
          <td>23.886519</td>
          <td>0.020001</td>
          <td>23.104340</td>
          <td>0.019196</td>
          <td>22.907646</td>
          <td>0.035940</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.446814</td>
          <td>0.647415</td>
          <td>27.156625</td>
          <td>0.213978</td>
          <td>26.804899</td>
          <td>0.252904</td>
          <td>26.332282</td>
          <td>0.312303</td>
          <td>26.084168</td>
          <td>0.522566</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.341634</td>
          <td>0.332988</td>
          <td>25.730258</td>
          <td>0.070647</td>
          <td>25.424957</td>
          <td>0.047359</td>
          <td>24.790253</td>
          <td>0.044066</td>
          <td>24.308520</td>
          <td>0.055084</td>
          <td>23.677925</td>
          <td>0.071149</td>
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
          <td>26.040944</td>
          <td>0.261450</td>
          <td>26.373426</td>
          <td>0.124139</td>
          <td>26.089652</td>
          <td>0.085334</td>
          <td>26.302913</td>
          <td>0.166097</td>
          <td>25.863365</td>
          <td>0.212771</td>
          <td>25.934000</td>
          <td>0.467643</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.753075</td>
          <td>0.911451</td>
          <td>26.804147</td>
          <td>0.179603</td>
          <td>26.731686</td>
          <td>0.149276</td>
          <td>26.458802</td>
          <td>0.189577</td>
          <td>26.584498</td>
          <td>0.381000</td>
          <td>25.400856</td>
          <td>0.309274</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>33.190617</td>
          <td>5.739955</td>
          <td>27.030823</td>
          <td>0.217285</td>
          <td>26.811388</td>
          <td>0.159824</td>
          <td>26.182812</td>
          <td>0.149880</td>
          <td>26.076827</td>
          <td>0.253906</td>
          <td>24.943754</td>
          <td>0.212707</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.710214</td>
          <td>0.887324</td>
          <td>27.358251</td>
          <td>0.284361</td>
          <td>26.687482</td>
          <td>0.143710</td>
          <td>25.687258</td>
          <td>0.097461</td>
          <td>25.944342</td>
          <td>0.227611</td>
          <td>25.861280</td>
          <td>0.442756</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.695930</td>
          <td>0.438162</td>
          <td>26.328461</td>
          <td>0.119389</td>
          <td>26.123686</td>
          <td>0.087930</td>
          <td>25.597673</td>
          <td>0.090087</td>
          <td>24.965085</td>
          <td>0.098389</td>
          <td>24.697395</td>
          <td>0.172826</td>
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
          <td>26.620402</td>
          <td>0.176435</td>
          <td>25.932853</td>
          <td>0.087410</td>
          <td>25.368084</td>
          <td>0.087149</td>
          <td>24.927076</td>
          <td>0.111769</td>
          <td>24.798696</td>
          <td>0.220822</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.428857</td>
          <td>2.093361</td>
          <td>26.781184</td>
          <td>0.182177</td>
          <td>26.761198</td>
          <td>0.285358</td>
          <td>26.578224</td>
          <td>0.436590</td>
          <td>25.761573</td>
          <td>0.473857</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.239410</td>
          <td>0.346610</td>
          <td>25.935824</td>
          <td>0.099704</td>
          <td>24.747145</td>
          <td>0.031320</td>
          <td>23.866685</td>
          <td>0.023735</td>
          <td>23.136610</td>
          <td>0.023623</td>
          <td>22.776506</td>
          <td>0.038780</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.312018</td>
          <td>1.275959</td>
          <td>28.488094</td>
          <td>0.720998</td>
          <td>26.755986</td>
          <td>0.302763</td>
          <td>26.250741</td>
          <td>0.359666</td>
          <td>25.835044</td>
          <td>0.529884</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.009128</td>
          <td>0.284068</td>
          <td>25.635961</td>
          <td>0.075101</td>
          <td>25.408547</td>
          <td>0.054994</td>
          <td>24.828035</td>
          <td>0.054068</td>
          <td>24.418556</td>
          <td>0.071513</td>
          <td>23.657273</td>
          <td>0.082678</td>
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
          <td>26.304589</td>
          <td>0.364443</td>
          <td>26.217812</td>
          <td>0.127284</td>
          <td>26.171832</td>
          <td>0.110058</td>
          <td>26.281202</td>
          <td>0.195873</td>
          <td>25.643173</td>
          <td>0.210530</td>
          <td>28.427385</td>
          <td>2.189977</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.317108</td>
          <td>0.751926</td>
          <td>27.232720</td>
          <td>0.294021</td>
          <td>26.971087</td>
          <td>0.214519</td>
          <td>26.575288</td>
          <td>0.246116</td>
          <td>26.424336</td>
          <td>0.389398</td>
          <td>25.410538</td>
          <td>0.363609</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.323312</td>
          <td>0.758776</td>
          <td>27.048971</td>
          <td>0.255062</td>
          <td>26.724312</td>
          <td>0.175724</td>
          <td>26.846694</td>
          <td>0.309355</td>
          <td>26.646255</td>
          <td>0.464530</td>
          <td>26.604416</td>
          <td>0.858589</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.104415</td>
          <td>1.230946</td>
          <td>27.146681</td>
          <td>0.280558</td>
          <td>26.398528</td>
          <td>0.135381</td>
          <td>25.880199</td>
          <td>0.140651</td>
          <td>25.919354</td>
          <td>0.267083</td>
          <td>25.585778</td>
          <td>0.426619</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>31.144704</td>
          <td>3.862009</td>
          <td>26.529944</td>
          <td>0.164818</td>
          <td>26.308879</td>
          <td>0.122658</td>
          <td>25.541744</td>
          <td>0.102557</td>
          <td>25.070975</td>
          <td>0.127919</td>
          <td>24.682579</td>
          <td>0.202379</td>
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
          <td>28.794092</td>
          <td>1.616004</td>
          <td>26.715432</td>
          <td>0.166586</td>
          <td>26.114280</td>
          <td>0.087216</td>
          <td>25.391373</td>
          <td>0.075113</td>
          <td>25.199206</td>
          <td>0.120714</td>
          <td>24.906207</td>
          <td>0.206157</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.508103</td>
          <td>0.779580</td>
          <td>27.435169</td>
          <td>0.302777</td>
          <td>27.957627</td>
          <td>0.407698</td>
          <td>26.853525</td>
          <td>0.263422</td>
          <td>26.230536</td>
          <td>0.288019</td>
          <td>26.186054</td>
          <td>0.563063</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.705246</td>
          <td>0.463932</td>
          <td>25.906627</td>
          <td>0.088717</td>
          <td>24.817959</td>
          <td>0.030079</td>
          <td>23.884943</td>
          <td>0.021717</td>
          <td>23.139696</td>
          <td>0.021422</td>
          <td>22.814910</td>
          <td>0.036073</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.563647</td>
          <td>0.815896</td>
          <td>27.379819</td>
          <td>0.316608</td>
          <td>26.547500</td>
          <td>0.254762</td>
          <td>26.081591</td>
          <td>0.313592</td>
          <td>25.559520</td>
          <td>0.430279</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.537629</td>
          <td>0.388520</td>
          <td>25.709499</td>
          <td>0.069450</td>
          <td>25.417264</td>
          <td>0.047104</td>
          <td>24.835854</td>
          <td>0.045956</td>
          <td>24.396049</td>
          <td>0.059619</td>
          <td>23.713341</td>
          <td>0.073522</td>
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
          <td>26.442451</td>
          <td>0.378797</td>
          <td>26.285689</td>
          <td>0.123158</td>
          <td>26.036472</td>
          <td>0.088121</td>
          <td>25.959320</td>
          <td>0.134033</td>
          <td>25.868117</td>
          <td>0.230124</td>
          <td>25.206053</td>
          <td>0.284888</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.890020</td>
          <td>0.999141</td>
          <td>26.876039</td>
          <td>0.193489</td>
          <td>26.941508</td>
          <td>0.181405</td>
          <td>26.599693</td>
          <td>0.216911</td>
          <td>25.948253</td>
          <td>0.231922</td>
          <td>25.556769</td>
          <td>0.355421</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.997301</td>
          <td>1.081299</td>
          <td>26.779331</td>
          <td>0.183259</td>
          <td>27.194971</td>
          <td>0.231378</td>
          <td>26.528791</td>
          <td>0.211145</td>
          <td>26.292734</td>
          <td>0.316329</td>
          <td>26.075780</td>
          <td>0.541661</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.217578</td>
          <td>0.325037</td>
          <td>27.116554</td>
          <td>0.256444</td>
          <td>26.729029</td>
          <td>0.166527</td>
          <td>25.851523</td>
          <td>0.126681</td>
          <td>25.924005</td>
          <td>0.249255</td>
          <td>25.812420</td>
          <td>0.472343</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.219624</td>
          <td>0.309741</td>
          <td>26.491825</td>
          <td>0.142171</td>
          <td>26.077748</td>
          <td>0.087806</td>
          <td>25.601110</td>
          <td>0.094125</td>
          <td>25.280127</td>
          <td>0.134526</td>
          <td>25.216082</td>
          <td>0.276508</td>
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
