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

    <pzflow.flow.Flow at 0x7f6e8d42de40>



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
          <td>26.294068</td>
          <td>0.320650</td>
          <td>26.588103</td>
          <td>0.149390</td>
          <td>26.145644</td>
          <td>0.089645</td>
          <td>25.325726</td>
          <td>0.070866</td>
          <td>25.029928</td>
          <td>0.104137</td>
          <td>24.497106</td>
          <td>0.145625</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.184176</td>
          <td>0.625475</td>
          <td>30.093263</td>
          <td>1.683396</td>
          <td>27.777081</td>
          <td>0.354065</td>
          <td>28.534490</td>
          <td>0.899212</td>
          <td>26.671572</td>
          <td>0.407490</td>
          <td>27.172238</td>
          <td>1.071801</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.845605</td>
          <td>0.490116</td>
          <td>25.888734</td>
          <td>0.081248</td>
          <td>24.743413</td>
          <td>0.025965</td>
          <td>23.854606</td>
          <td>0.019468</td>
          <td>23.173884</td>
          <td>0.020359</td>
          <td>22.900811</td>
          <td>0.035723</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.483378</td>
          <td>0.372186</td>
          <td>28.504288</td>
          <td>0.673596</td>
          <td>27.207471</td>
          <td>0.223237</td>
          <td>26.756183</td>
          <td>0.242970</td>
          <td>25.806977</td>
          <td>0.202961</td>
          <td>25.502840</td>
          <td>0.335440</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.907606</td>
          <td>0.513024</td>
          <td>25.902710</td>
          <td>0.082254</td>
          <td>25.449584</td>
          <td>0.048406</td>
          <td>24.797356</td>
          <td>0.044345</td>
          <td>24.377018</td>
          <td>0.058537</td>
          <td>23.587221</td>
          <td>0.065658</td>
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
          <td>26.521759</td>
          <td>0.383446</td>
          <td>26.193440</td>
          <td>0.106145</td>
          <td>26.165612</td>
          <td>0.091233</td>
          <td>26.177730</td>
          <td>0.149227</td>
          <td>26.117534</td>
          <td>0.262512</td>
          <td>24.919367</td>
          <td>0.208414</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.267162</td>
          <td>0.313848</td>
          <td>27.185768</td>
          <td>0.247027</td>
          <td>27.036265</td>
          <td>0.193433</td>
          <td>26.374066</td>
          <td>0.176459</td>
          <td>25.735856</td>
          <td>0.191177</td>
          <td>25.504467</td>
          <td>0.335872</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.155418</td>
          <td>0.240928</td>
          <td>26.772423</td>
          <td>0.154583</td>
          <td>26.312069</td>
          <td>0.167398</td>
          <td>26.026389</td>
          <td>0.243591</td>
          <td>24.977899</td>
          <td>0.218853</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.280024</td>
          <td>0.266852</td>
          <td>26.537252</td>
          <td>0.126219</td>
          <td>26.099461</td>
          <td>0.139508</td>
          <td>25.479198</td>
          <td>0.153696</td>
          <td>25.077625</td>
          <td>0.237731</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.411828</td>
          <td>0.730994</td>
          <td>26.786204</td>
          <td>0.176893</td>
          <td>26.203524</td>
          <td>0.094323</td>
          <td>25.683520</td>
          <td>0.097142</td>
          <td>25.336595</td>
          <td>0.135952</td>
          <td>25.565621</td>
          <td>0.352472</td>
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
          <td>27.413301</td>
          <td>0.799144</td>
          <td>26.976826</td>
          <td>0.237782</td>
          <td>26.015765</td>
          <td>0.094017</td>
          <td>25.309851</td>
          <td>0.082792</td>
          <td>24.903597</td>
          <td>0.109503</td>
          <td>24.737655</td>
          <td>0.209860</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.436573</td>
          <td>0.718529</td>
          <td>27.871055</td>
          <td>0.438760</td>
          <td>29.021091</td>
          <td>1.331175</td>
          <td>26.901200</td>
          <td>0.554432</td>
          <td>25.473385</td>
          <td>0.380483</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.289002</td>
          <td>0.360361</td>
          <td>25.899583</td>
          <td>0.096591</td>
          <td>24.805392</td>
          <td>0.032966</td>
          <td>23.901742</td>
          <td>0.024465</td>
          <td>23.158451</td>
          <td>0.024072</td>
          <td>22.867092</td>
          <td>0.042017</td>
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
          <td>28.980079</td>
          <td>0.987396</td>
          <td>26.891244</td>
          <td>0.337231</td>
          <td>29.264600</td>
          <td>2.158321</td>
          <td>25.542249</td>
          <td>0.426010</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.170117</td>
          <td>0.323197</td>
          <td>25.865302</td>
          <td>0.091888</td>
          <td>25.435385</td>
          <td>0.056319</td>
          <td>24.829379</td>
          <td>0.054132</td>
          <td>24.342240</td>
          <td>0.066844</td>
          <td>23.846612</td>
          <td>0.097649</td>
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
          <td>26.120988</td>
          <td>0.315279</td>
          <td>26.176555</td>
          <td>0.122816</td>
          <td>26.059498</td>
          <td>0.099764</td>
          <td>26.288573</td>
          <td>0.197091</td>
          <td>25.904360</td>
          <td>0.261293</td>
          <td>26.782677</td>
          <td>0.965231</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.730008</td>
          <td>0.977612</td>
          <td>27.230130</td>
          <td>0.293408</td>
          <td>26.673690</td>
          <td>0.166923</td>
          <td>26.384402</td>
          <td>0.210064</td>
          <td>26.187112</td>
          <td>0.323242</td>
          <td>26.539835</td>
          <td>0.818442</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.224349</td>
          <td>0.294135</td>
          <td>26.761784</td>
          <td>0.181395</td>
          <td>26.420103</td>
          <td>0.218258</td>
          <td>26.357778</td>
          <td>0.372610</td>
          <td>25.125771</td>
          <td>0.292299</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.402312</td>
          <td>0.807683</td>
          <td>27.730886</td>
          <td>0.443631</td>
          <td>26.536296</td>
          <td>0.152417</td>
          <td>25.907179</td>
          <td>0.143956</td>
          <td>25.529590</td>
          <td>0.193304</td>
          <td>25.714582</td>
          <td>0.470143</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.843570</td>
          <td>1.049554</td>
          <td>26.633648</td>
          <td>0.179992</td>
          <td>26.171168</td>
          <td>0.108802</td>
          <td>25.821922</td>
          <td>0.130875</td>
          <td>25.086941</td>
          <td>0.129700</td>
          <td>25.278844</td>
          <td>0.329544</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>27.018891</td>
          <td>0.215157</td>
          <td>25.976210</td>
          <td>0.077218</td>
          <td>25.271392</td>
          <td>0.067547</td>
          <td>24.977912</td>
          <td>0.099514</td>
          <td>25.013736</td>
          <td>0.225504</td>
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
          <td>27.451882</td>
          <td>0.273202</td>
          <td>27.125586</td>
          <td>0.328008</td>
          <td>26.654295</td>
          <td>0.402449</td>
          <td>25.212646</td>
          <td>0.265853</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.673251</td>
          <td>0.452928</td>
          <td>26.016552</td>
          <td>0.097696</td>
          <td>24.823254</td>
          <td>0.030220</td>
          <td>23.865279</td>
          <td>0.021355</td>
          <td>23.145050</td>
          <td>0.021520</td>
          <td>22.796813</td>
          <td>0.035501</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.244624</td>
          <td>0.659110</td>
          <td>27.339982</td>
          <td>0.306678</td>
          <td>26.581547</td>
          <td>0.261963</td>
          <td>25.671953</td>
          <td>0.224519</td>
          <td>25.782152</td>
          <td>0.508216</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.776928</td>
          <td>0.466087</td>
          <td>25.746105</td>
          <td>0.071732</td>
          <td>25.393381</td>
          <td>0.046116</td>
          <td>24.790966</td>
          <td>0.044161</td>
          <td>24.279408</td>
          <td>0.053756</td>
          <td>23.780110</td>
          <td>0.077990</td>
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
          <td>26.852404</td>
          <td>0.516171</td>
          <td>26.223953</td>
          <td>0.116731</td>
          <td>26.214775</td>
          <td>0.103047</td>
          <td>26.364473</td>
          <td>0.189479</td>
          <td>25.456692</td>
          <td>0.162759</td>
          <td>25.071659</td>
          <td>0.255344</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.583136</td>
          <td>0.825173</td>
          <td>27.101272</td>
          <td>0.233513</td>
          <td>27.085760</td>
          <td>0.204848</td>
          <td>26.342346</td>
          <td>0.174664</td>
          <td>26.889349</td>
          <td>0.487134</td>
          <td>25.771134</td>
          <td>0.419604</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.111365</td>
          <td>0.285802</td>
          <td>28.195442</td>
          <td>0.561016</td>
          <td>27.029234</td>
          <td>0.201506</td>
          <td>26.150170</td>
          <td>0.153217</td>
          <td>26.561285</td>
          <td>0.390672</td>
          <td>25.453916</td>
          <td>0.337781</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.137172</td>
          <td>0.646053</td>
          <td>27.355832</td>
          <td>0.311265</td>
          <td>26.663272</td>
          <td>0.157435</td>
          <td>25.870483</td>
          <td>0.128779</td>
          <td>25.543826</td>
          <td>0.181470</td>
          <td>24.885984</td>
          <td>0.226842</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.674120</td>
          <td>0.166186</td>
          <td>26.058441</td>
          <td>0.086326</td>
          <td>25.557980</td>
          <td>0.090625</td>
          <td>25.379139</td>
          <td>0.146507</td>
          <td>25.238533</td>
          <td>0.281591</td>
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
