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

    <pzflow.flow.Flow at 0x7f596d72fd00>



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
          <td>26.573247</td>
          <td>0.147498</td>
          <td>26.118228</td>
          <td>0.087508</td>
          <td>25.326661</td>
          <td>0.070924</td>
          <td>25.144524</td>
          <td>0.115092</td>
          <td>24.704361</td>
          <td>0.173852</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.139807</td>
          <td>2.577144</td>
          <td>27.434490</td>
          <td>0.269124</td>
          <td>28.514751</td>
          <td>0.888141</td>
          <td>27.289328</td>
          <td>0.640799</td>
          <td>30.887238</td>
          <td>4.301342</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.798788</td>
          <td>0.473362</td>
          <td>26.112219</td>
          <td>0.098869</td>
          <td>24.795020</td>
          <td>0.027160</td>
          <td>23.843420</td>
          <td>0.019285</td>
          <td>23.136243</td>
          <td>0.019720</td>
          <td>22.833182</td>
          <td>0.033653</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.907300</td>
          <td>0.437484</td>
          <td>27.071852</td>
          <td>0.199311</td>
          <td>26.636458</td>
          <td>0.220024</td>
          <td>25.884466</td>
          <td>0.216551</td>
          <td>24.872844</td>
          <td>0.200442</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.878510</td>
          <td>0.502171</td>
          <td>25.680851</td>
          <td>0.067629</td>
          <td>25.444288</td>
          <td>0.048179</td>
          <td>24.757360</td>
          <td>0.042799</td>
          <td>24.467440</td>
          <td>0.063424</td>
          <td>23.666028</td>
          <td>0.070404</td>
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
          <td>27.015965</td>
          <td>0.555062</td>
          <td>26.485238</td>
          <td>0.136740</td>
          <td>25.943659</td>
          <td>0.075018</td>
          <td>25.973474</td>
          <td>0.125107</td>
          <td>25.864391</td>
          <td>0.212953</td>
          <td>25.309416</td>
          <td>0.287334</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.679921</td>
          <td>0.432880</td>
          <td>27.401759</td>
          <td>0.294529</td>
          <td>26.857875</td>
          <td>0.166293</td>
          <td>26.476794</td>
          <td>0.192476</td>
          <td>26.328109</td>
          <td>0.311262</td>
          <td>25.202375</td>
          <td>0.263391</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.875666</td>
          <td>0.501120</td>
          <td>27.398224</td>
          <td>0.293691</td>
          <td>27.164015</td>
          <td>0.215301</td>
          <td>26.243908</td>
          <td>0.157935</td>
          <td>26.087847</td>
          <td>0.256211</td>
          <td>25.491327</td>
          <td>0.332394</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.109298</td>
          <td>0.231919</td>
          <td>26.676132</td>
          <td>0.142312</td>
          <td>25.890523</td>
          <td>0.116406</td>
          <td>25.780877</td>
          <td>0.198562</td>
          <td>25.005473</td>
          <td>0.223932</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>29.008844</td>
          <td>1.785744</td>
          <td>26.240336</td>
          <td>0.110577</td>
          <td>26.055892</td>
          <td>0.082832</td>
          <td>25.570339</td>
          <td>0.087946</td>
          <td>24.925595</td>
          <td>0.095039</td>
          <td>24.859136</td>
          <td>0.198147</td>
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
          <td>30.525536</td>
          <td>3.258158</td>
          <td>26.963707</td>
          <td>0.235219</td>
          <td>26.041328</td>
          <td>0.096150</td>
          <td>25.412419</td>
          <td>0.090614</td>
          <td>25.144298</td>
          <td>0.134954</td>
          <td>25.345164</td>
          <td>0.344090</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.464302</td>
          <td>0.826089</td>
          <td>29.716820</td>
          <td>1.518641</td>
          <td>27.291677</td>
          <td>0.278294</td>
          <td>27.308526</td>
          <td>0.438389</td>
          <td>26.062336</td>
          <td>0.291424</td>
          <td>25.160372</td>
          <td>0.297024</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.408824</td>
          <td>0.395495</td>
          <td>25.919475</td>
          <td>0.098288</td>
          <td>24.793458</td>
          <td>0.032621</td>
          <td>23.883534</td>
          <td>0.024083</td>
          <td>23.126741</td>
          <td>0.023423</td>
          <td>22.851512</td>
          <td>0.041441</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.187804</td>
          <td>2.995018</td>
          <td>27.515030</td>
          <td>0.386904</td>
          <td>26.807776</td>
          <td>0.198671</td>
          <td>26.660989</td>
          <td>0.280421</td>
          <td>26.496210</td>
          <td>0.434627</td>
          <td>24.689679</td>
          <td>0.215260</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.772213</td>
          <td>0.512420</td>
          <td>25.631176</td>
          <td>0.074785</td>
          <td>25.402843</td>
          <td>0.054716</td>
          <td>24.656744</td>
          <td>0.046444</td>
          <td>24.398343</td>
          <td>0.070246</td>
          <td>23.792872</td>
          <td>0.093152</td>
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
          <td>26.744741</td>
          <td>0.508853</td>
          <td>26.548556</td>
          <td>0.169066</td>
          <td>26.178082</td>
          <td>0.110659</td>
          <td>26.126144</td>
          <td>0.171798</td>
          <td>25.932147</td>
          <td>0.267290</td>
          <td>25.885641</td>
          <td>0.528646</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.330061</td>
          <td>0.758413</td>
          <td>27.213870</td>
          <td>0.289585</td>
          <td>27.000850</td>
          <td>0.219907</td>
          <td>26.527915</td>
          <td>0.236685</td>
          <td>26.385887</td>
          <td>0.377960</td>
          <td>25.573377</td>
          <td>0.412455</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.507076</td>
          <td>0.368095</td>
          <td>27.094590</td>
          <td>0.239618</td>
          <td>26.209681</td>
          <td>0.182905</td>
          <td>26.207569</td>
          <td>0.331105</td>
          <td>25.437259</td>
          <td>0.374198</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.915727</td>
          <td>0.579694</td>
          <td>27.645969</td>
          <td>0.415904</td>
          <td>26.495954</td>
          <td>0.147231</td>
          <td>25.634130</td>
          <td>0.113643</td>
          <td>25.832310</td>
          <td>0.248708</td>
          <td>25.555031</td>
          <td>0.416729</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.147300</td>
          <td>0.319518</td>
          <td>26.499355</td>
          <td>0.160574</td>
          <td>26.159343</td>
          <td>0.107685</td>
          <td>25.650448</td>
          <td>0.112769</td>
          <td>25.125395</td>
          <td>0.134085</td>
          <td>25.067539</td>
          <td>0.278122</td>
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
          <td>28.087283</td>
          <td>1.113675</td>
          <td>26.909490</td>
          <td>0.196321</td>
          <td>26.062528</td>
          <td>0.083329</td>
          <td>25.401776</td>
          <td>0.075807</td>
          <td>24.858169</td>
          <td>0.089584</td>
          <td>25.021237</td>
          <td>0.226913</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.892973</td>
          <td>0.871078</td>
          <td>28.108587</td>
          <td>0.457227</td>
          <td>27.862671</td>
          <td>0.573049</td>
          <td>26.235591</td>
          <td>0.289198</td>
          <td>25.506681</td>
          <td>0.336760</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.616145</td>
          <td>0.433812</td>
          <td>26.072689</td>
          <td>0.102614</td>
          <td>24.785771</td>
          <td>0.029242</td>
          <td>23.891425</td>
          <td>0.021838</td>
          <td>23.116551</td>
          <td>0.021003</td>
          <td>22.799801</td>
          <td>0.035595</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.782850</td>
          <td>0.937150</td>
          <td>27.331064</td>
          <td>0.304492</td>
          <td>26.858438</td>
          <td>0.327484</td>
          <td>26.151850</td>
          <td>0.331632</td>
          <td>25.744384</td>
          <td>0.494258</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.665629</td>
          <td>0.428572</td>
          <td>25.645429</td>
          <td>0.065625</td>
          <td>25.400856</td>
          <td>0.046423</td>
          <td>24.850780</td>
          <td>0.046569</td>
          <td>24.315464</td>
          <td>0.055505</td>
          <td>23.822521</td>
          <td>0.080965</td>
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
          <td>26.573375</td>
          <td>0.418955</td>
          <td>26.447183</td>
          <td>0.141596</td>
          <td>26.277636</td>
          <td>0.108867</td>
          <td>26.020818</td>
          <td>0.141337</td>
          <td>26.195351</td>
          <td>0.300635</td>
          <td>25.294679</td>
          <td>0.305974</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.117273</td>
          <td>1.141403</td>
          <td>26.675241</td>
          <td>0.163212</td>
          <td>26.801755</td>
          <td>0.161074</td>
          <td>26.503481</td>
          <td>0.200130</td>
          <td>26.187560</td>
          <td>0.282174</td>
          <td>25.460638</td>
          <td>0.329447</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.317632</td>
          <td>0.704403</td>
          <td>27.136968</td>
          <td>0.246988</td>
          <td>26.863096</td>
          <td>0.175134</td>
          <td>26.320424</td>
          <td>0.177160</td>
          <td>25.848768</td>
          <td>0.220171</td>
          <td>25.483517</td>
          <td>0.345769</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.075758</td>
          <td>0.618964</td>
          <td>27.731471</td>
          <td>0.417615</td>
          <td>26.783537</td>
          <td>0.174431</td>
          <td>25.874253</td>
          <td>0.129200</td>
          <td>25.380513</td>
          <td>0.157923</td>
          <td>25.372097</td>
          <td>0.336569</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.950979</td>
          <td>1.047274</td>
          <td>26.742486</td>
          <td>0.176128</td>
          <td>26.061174</td>
          <td>0.086534</td>
          <td>25.560491</td>
          <td>0.090825</td>
          <td>24.949612</td>
          <td>0.100903</td>
          <td>24.584765</td>
          <td>0.163287</td>
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
