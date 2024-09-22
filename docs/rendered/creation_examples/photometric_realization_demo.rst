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

    <pzflow.flow.Flow at 0x7f76c49418d0>



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
          <td>26.655483</td>
          <td>0.158263</td>
          <td>26.077349</td>
          <td>0.084414</td>
          <td>25.446579</td>
          <td>0.078856</td>
          <td>25.101404</td>
          <td>0.110846</td>
          <td>24.773162</td>
          <td>0.184292</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.023001</td>
          <td>0.557880</td>
          <td>28.778121</td>
          <td>0.808748</td>
          <td>27.563378</td>
          <td>0.298733</td>
          <td>27.268496</td>
          <td>0.366771</td>
          <td>26.391200</td>
          <td>0.327322</td>
          <td>33.034235</td>
          <td>6.430385</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.020248</td>
          <td>0.257065</td>
          <td>25.850484</td>
          <td>0.078555</td>
          <td>24.805141</td>
          <td>0.027401</td>
          <td>23.824153</td>
          <td>0.018975</td>
          <td>23.133374</td>
          <td>0.019672</td>
          <td>22.850313</td>
          <td>0.034166</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.843738</td>
          <td>0.963852</td>
          <td>28.935299</td>
          <td>0.894104</td>
          <td>27.907463</td>
          <td>0.391924</td>
          <td>26.804284</td>
          <td>0.252776</td>
          <td>26.255754</td>
          <td>0.293688</td>
          <td>25.272570</td>
          <td>0.278885</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.387238</td>
          <td>0.345197</td>
          <td>25.753597</td>
          <td>0.072119</td>
          <td>25.434679</td>
          <td>0.047770</td>
          <td>24.752502</td>
          <td>0.042615</td>
          <td>24.462913</td>
          <td>0.063170</td>
          <td>23.617867</td>
          <td>0.067465</td>
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
          <td>26.063080</td>
          <td>0.266212</td>
          <td>26.339240</td>
          <td>0.120511</td>
          <td>26.175615</td>
          <td>0.092038</td>
          <td>25.671435</td>
          <td>0.096117</td>
          <td>25.379058</td>
          <td>0.141024</td>
          <td>26.024783</td>
          <td>0.500270</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.878099</td>
          <td>0.502019</td>
          <td>27.202995</td>
          <td>0.250549</td>
          <td>26.986533</td>
          <td>0.185483</td>
          <td>26.434026</td>
          <td>0.185652</td>
          <td>26.490888</td>
          <td>0.354143</td>
          <td>26.101559</td>
          <td>0.529240</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.557917</td>
          <td>1.437771</td>
          <td>27.196320</td>
          <td>0.249179</td>
          <td>27.003956</td>
          <td>0.188233</td>
          <td>26.669330</td>
          <td>0.226121</td>
          <td>26.308940</td>
          <td>0.306520</td>
          <td>26.527097</td>
          <td>0.713591</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.988598</td>
          <td>1.051372</td>
          <td>27.566419</td>
          <td>0.335923</td>
          <td>26.639317</td>
          <td>0.137867</td>
          <td>25.726590</td>
          <td>0.100879</td>
          <td>25.327057</td>
          <td>0.134837</td>
          <td>25.259124</td>
          <td>0.275857</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.288225</td>
          <td>0.319162</td>
          <td>26.547763</td>
          <td>0.144304</td>
          <td>26.102196</td>
          <td>0.086282</td>
          <td>25.648074</td>
          <td>0.094166</td>
          <td>25.065260</td>
          <td>0.107403</td>
          <td>24.703536</td>
          <td>0.173730</td>
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
          <td>28.620239</td>
          <td>1.584728</td>
          <td>26.823858</td>
          <td>0.209403</td>
          <td>26.081831</td>
          <td>0.099625</td>
          <td>25.383914</td>
          <td>0.088371</td>
          <td>25.117069</td>
          <td>0.131816</td>
          <td>24.435918</td>
          <td>0.162623</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.381882</td>
          <td>0.782990</td>
          <td>27.869136</td>
          <td>0.480378</td>
          <td>27.350590</td>
          <td>0.291884</td>
          <td>27.947831</td>
          <td>0.694684</td>
          <td>27.325858</td>
          <td>0.744514</td>
          <td>27.174685</td>
          <td>1.196472</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.793248</td>
          <td>0.527684</td>
          <td>26.130676</td>
          <td>0.118162</td>
          <td>24.790794</td>
          <td>0.032545</td>
          <td>23.838917</td>
          <td>0.023174</td>
          <td>23.142684</td>
          <td>0.023747</td>
          <td>22.787772</td>
          <td>0.039168</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.373690</td>
          <td>0.809239</td>
          <td>28.282314</td>
          <td>0.678134</td>
          <td>27.350196</td>
          <td>0.310205</td>
          <td>26.750412</td>
          <td>0.301411</td>
          <td>26.056761</td>
          <td>0.308424</td>
          <td>24.945211</td>
          <td>0.265799</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.646648</td>
          <td>0.466926</td>
          <td>25.888921</td>
          <td>0.093811</td>
          <td>25.371357</td>
          <td>0.053209</td>
          <td>24.877887</td>
          <td>0.056513</td>
          <td>24.261214</td>
          <td>0.062215</td>
          <td>23.608904</td>
          <td>0.079226</td>
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
          <td>26.739320</td>
          <td>0.506831</td>
          <td>26.249883</td>
          <td>0.130864</td>
          <td>26.151196</td>
          <td>0.108093</td>
          <td>26.046083</td>
          <td>0.160467</td>
          <td>26.643143</td>
          <td>0.466698</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.371227</td>
          <td>0.779286</td>
          <td>26.868569</td>
          <td>0.218140</td>
          <td>26.597013</td>
          <td>0.156344</td>
          <td>26.521323</td>
          <td>0.235399</td>
          <td>26.122592</td>
          <td>0.307006</td>
          <td>25.153581</td>
          <td>0.296509</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.008981</td>
          <td>0.246824</td>
          <td>27.457887</td>
          <td>0.321774</td>
          <td>26.486906</td>
          <td>0.230715</td>
          <td>25.990279</td>
          <td>0.278115</td>
          <td>24.982028</td>
          <td>0.260081</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.637619</td>
          <td>0.473257</td>
          <td>27.125736</td>
          <td>0.275831</td>
          <td>26.511256</td>
          <td>0.149178</td>
          <td>25.669049</td>
          <td>0.117151</td>
          <td>25.529170</td>
          <td>0.193236</td>
          <td>25.446988</td>
          <td>0.383465</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.260542</td>
          <td>0.726524</td>
          <td>26.449465</td>
          <td>0.153871</td>
          <td>26.134058</td>
          <td>0.105332</td>
          <td>25.617041</td>
          <td>0.109531</td>
          <td>25.108925</td>
          <td>0.132190</td>
          <td>24.758852</td>
          <td>0.215712</td>
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
          <td>29.645546</td>
          <td>2.325239</td>
          <td>27.010078</td>
          <td>0.213581</td>
          <td>25.946303</td>
          <td>0.075204</td>
          <td>25.253126</td>
          <td>0.066462</td>
          <td>24.953702</td>
          <td>0.097424</td>
          <td>24.732783</td>
          <td>0.178120</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>37.637390</td>
          <td>8.969754</td>
          <td>27.979931</td>
          <td>0.414726</td>
          <td>27.542233</td>
          <td>0.452880</td>
          <td>26.641202</td>
          <td>0.398413</td>
          <td>26.062937</td>
          <td>0.514929</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.060527</td>
          <td>0.280444</td>
          <td>25.937237</td>
          <td>0.091134</td>
          <td>24.782893</td>
          <td>0.029169</td>
          <td>23.866882</td>
          <td>0.021384</td>
          <td>23.177138</td>
          <td>0.022119</td>
          <td>22.805564</td>
          <td>0.035777</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.530340</td>
          <td>2.390589</td>
          <td>27.965424</td>
          <td>0.540931</td>
          <td>27.370471</td>
          <td>0.314253</td>
          <td>26.294635</td>
          <td>0.206588</td>
          <td>25.892866</td>
          <td>0.269284</td>
          <td>24.921234</td>
          <td>0.259757</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.159434</td>
          <td>0.288108</td>
          <td>25.667468</td>
          <td>0.066917</td>
          <td>25.447839</td>
          <td>0.048401</td>
          <td>24.735904</td>
          <td>0.042055</td>
          <td>24.326591</td>
          <td>0.056055</td>
          <td>23.758619</td>
          <td>0.076524</td>
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
          <td>27.181676</td>
          <td>0.652600</td>
          <td>26.448300</td>
          <td>0.141732</td>
          <td>26.223904</td>
          <td>0.103873</td>
          <td>25.970195</td>
          <td>0.135298</td>
          <td>25.851095</td>
          <td>0.226898</td>
          <td>25.591027</td>
          <td>0.386520</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.692661</td>
          <td>0.441420</td>
          <td>27.549741</td>
          <td>0.335821</td>
          <td>26.505092</td>
          <td>0.124760</td>
          <td>26.364560</td>
          <td>0.177988</td>
          <td>26.614195</td>
          <td>0.395545</td>
          <td>25.479529</td>
          <td>0.334419</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.587107</td>
          <td>0.841284</td>
          <td>27.678493</td>
          <td>0.381007</td>
          <td>26.858962</td>
          <td>0.174520</td>
          <td>27.018177</td>
          <td>0.315135</td>
          <td>25.691340</td>
          <td>0.192972</td>
          <td>25.368436</td>
          <td>0.315594</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.509823</td>
          <td>0.828948</td>
          <td>28.543195</td>
          <td>0.747283</td>
          <td>26.760227</td>
          <td>0.171009</td>
          <td>25.864734</td>
          <td>0.128139</td>
          <td>25.677333</td>
          <td>0.203080</td>
          <td>24.810110</td>
          <td>0.212955</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.644898</td>
          <td>0.162097</td>
          <td>26.007438</td>
          <td>0.082532</td>
          <td>25.690831</td>
          <td>0.101829</td>
          <td>25.221800</td>
          <td>0.127906</td>
          <td>24.891761</td>
          <td>0.211637</td>
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
