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

    <pzflow.flow.Flow at 0x7f15ff5fa8f0>



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
          <td>26.968384</td>
          <td>0.536286</td>
          <td>26.460447</td>
          <td>0.133846</td>
          <td>25.959367</td>
          <td>0.076067</td>
          <td>25.344193</td>
          <td>0.072033</td>
          <td>25.038416</td>
          <td>0.104913</td>
          <td>24.825040</td>
          <td>0.192542</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.485260</td>
          <td>2.184979</td>
          <td>28.573661</td>
          <td>0.706204</td>
          <td>27.329422</td>
          <td>0.246934</td>
          <td>26.682371</td>
          <td>0.228582</td>
          <td>26.112126</td>
          <td>0.261355</td>
          <td>25.593466</td>
          <td>0.360257</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.145725</td>
          <td>0.284678</td>
          <td>25.923747</td>
          <td>0.083792</td>
          <td>24.800050</td>
          <td>0.027279</td>
          <td>23.859441</td>
          <td>0.019548</td>
          <td>23.162961</td>
          <td>0.020171</td>
          <td>22.829077</td>
          <td>0.033532</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.381679</td>
          <td>0.716337</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.692272</td>
          <td>0.331138</td>
          <td>27.011364</td>
          <td>0.299113</td>
          <td>26.159032</td>
          <td>0.271551</td>
          <td>25.522744</td>
          <td>0.340762</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.333222</td>
          <td>0.330777</td>
          <td>25.724319</td>
          <td>0.070278</td>
          <td>25.479288</td>
          <td>0.049700</td>
          <td>24.814667</td>
          <td>0.045031</td>
          <td>24.389643</td>
          <td>0.059196</td>
          <td>23.615229</td>
          <td>0.067308</td>
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
          <td>26.871426</td>
          <td>0.499557</td>
          <td>26.313545</td>
          <td>0.117852</td>
          <td>26.057860</td>
          <td>0.082976</td>
          <td>26.117290</td>
          <td>0.141669</td>
          <td>25.724543</td>
          <td>0.189362</td>
          <td>25.799772</td>
          <td>0.422553</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.169906</td>
          <td>0.243822</td>
          <td>26.746439</td>
          <td>0.151178</td>
          <td>26.245867</td>
          <td>0.158200</td>
          <td>26.063578</td>
          <td>0.251160</td>
          <td>25.944579</td>
          <td>0.471356</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.879985</td>
          <td>0.502717</td>
          <td>27.583212</td>
          <td>0.340413</td>
          <td>26.647283</td>
          <td>0.138818</td>
          <td>26.526433</td>
          <td>0.200683</td>
          <td>25.930008</td>
          <td>0.224918</td>
          <td>25.872565</td>
          <td>0.446546</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.095185</td>
          <td>0.587434</td>
          <td>27.661483</td>
          <td>0.362019</td>
          <td>26.449706</td>
          <td>0.116976</td>
          <td>25.764523</td>
          <td>0.104285</td>
          <td>25.644993</td>
          <td>0.177036</td>
          <td>25.538842</td>
          <td>0.345119</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.268695</td>
          <td>0.663266</td>
          <td>26.369970</td>
          <td>0.123768</td>
          <td>26.087869</td>
          <td>0.085200</td>
          <td>25.693894</td>
          <td>0.098030</td>
          <td>25.291323</td>
          <td>0.130735</td>
          <td>25.517360</td>
          <td>0.339315</td>
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
          <td>26.781166</td>
          <td>0.515687</td>
          <td>27.123571</td>
          <td>0.268200</td>
          <td>25.860538</td>
          <td>0.082016</td>
          <td>25.391199</td>
          <td>0.088939</td>
          <td>25.174517</td>
          <td>0.138519</td>
          <td>24.970752</td>
          <td>0.254555</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>31.108634</td>
          <td>3.818733</td>
          <td>27.782672</td>
          <td>0.450270</td>
          <td>27.941643</td>
          <td>0.462726</td>
          <td>26.481368</td>
          <td>0.226854</td>
          <td>26.973703</td>
          <td>0.583998</td>
          <td>26.010319</td>
          <td>0.568432</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.147308</td>
          <td>0.677755</td>
          <td>25.880074</td>
          <td>0.094955</td>
          <td>24.769799</td>
          <td>0.031950</td>
          <td>23.878752</td>
          <td>0.023983</td>
          <td>23.128254</td>
          <td>0.023454</td>
          <td>22.786802</td>
          <td>0.039135</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.329357</td>
          <td>0.383490</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.275759</td>
          <td>0.292194</td>
          <td>26.836427</td>
          <td>0.322875</td>
          <td>25.899022</td>
          <td>0.271534</td>
          <td>25.301156</td>
          <td>0.353520</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.455094</td>
          <td>0.403837</td>
          <td>25.772437</td>
          <td>0.084691</td>
          <td>25.421458</td>
          <td>0.055627</td>
          <td>24.760461</td>
          <td>0.050921</td>
          <td>24.266960</td>
          <td>0.062532</td>
          <td>23.639493</td>
          <td>0.081393</td>
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
          <td>27.599921</td>
          <td>0.910595</td>
          <td>26.492634</td>
          <td>0.161201</td>
          <td>26.133729</td>
          <td>0.106457</td>
          <td>25.986965</td>
          <td>0.152551</td>
          <td>26.036629</td>
          <td>0.290937</td>
          <td>25.708714</td>
          <td>0.463844</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.535426</td>
          <td>0.430474</td>
          <td>27.086218</td>
          <td>0.261058</td>
          <td>26.640812</td>
          <td>0.162308</td>
          <td>26.664847</td>
          <td>0.264866</td>
          <td>26.187750</td>
          <td>0.323406</td>
          <td>25.534225</td>
          <td>0.400241</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.450715</td>
          <td>0.405834</td>
          <td>26.729169</td>
          <td>0.195565</td>
          <td>27.010989</td>
          <td>0.223586</td>
          <td>26.339532</td>
          <td>0.204044</td>
          <td>25.733814</td>
          <td>0.225288</td>
          <td>25.496026</td>
          <td>0.391649</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.299373</td>
          <td>0.668935</td>
          <td>26.733805</td>
          <td>0.180357</td>
          <td>25.854630</td>
          <td>0.137585</td>
          <td>25.662112</td>
          <td>0.216012</td>
          <td>25.304913</td>
          <td>0.343127</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.440308</td>
          <td>0.152669</td>
          <td>26.180145</td>
          <td>0.109658</td>
          <td>25.583928</td>
          <td>0.106410</td>
          <td>25.143379</td>
          <td>0.136183</td>
          <td>24.508921</td>
          <td>0.174787</td>
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
          <td>26.981541</td>
          <td>0.541469</td>
          <td>26.574756</td>
          <td>0.147705</td>
          <td>26.092476</td>
          <td>0.085557</td>
          <td>25.328729</td>
          <td>0.071064</td>
          <td>25.025430</td>
          <td>0.103742</td>
          <td>24.668649</td>
          <td>0.168673</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.731314</td>
          <td>0.899568</td>
          <td>29.282116</td>
          <td>1.102663</td>
          <td>28.124312</td>
          <td>0.462655</td>
          <td>27.874576</td>
          <td>0.577945</td>
          <td>26.639021</td>
          <td>0.397743</td>
          <td>25.629700</td>
          <td>0.370929</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.631043</td>
          <td>0.438735</td>
          <td>25.848228</td>
          <td>0.084278</td>
          <td>24.753146</td>
          <td>0.028419</td>
          <td>23.852709</td>
          <td>0.021127</td>
          <td>23.156933</td>
          <td>0.021740</td>
          <td>22.812092</td>
          <td>0.035983</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.566585</td>
          <td>0.912966</td>
          <td>28.321042</td>
          <td>0.694536</td>
          <td>27.062036</td>
          <td>0.244655</td>
          <td>27.918473</td>
          <td>0.716215</td>
          <td>26.572629</td>
          <td>0.459020</td>
          <td>25.666372</td>
          <td>0.466390</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.148527</td>
          <td>0.285582</td>
          <td>25.803485</td>
          <td>0.075459</td>
          <td>25.514757</td>
          <td>0.051364</td>
          <td>24.740854</td>
          <td>0.042240</td>
          <td>24.290309</td>
          <td>0.054279</td>
          <td>23.662463</td>
          <td>0.070287</td>
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
          <td>26.158530</td>
          <td>0.302743</td>
          <td>26.215270</td>
          <td>0.115853</td>
          <td>26.153928</td>
          <td>0.097698</td>
          <td>26.093396</td>
          <td>0.150437</td>
          <td>25.634365</td>
          <td>0.189251</td>
          <td>25.663669</td>
          <td>0.408779</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.928131</td>
          <td>0.525779</td>
          <td>26.818005</td>
          <td>0.184245</td>
          <td>26.673846</td>
          <td>0.144344</td>
          <td>26.365245</td>
          <td>0.178091</td>
          <td>26.419377</td>
          <td>0.339714</td>
          <td>25.668982</td>
          <td>0.387911</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.610340</td>
          <td>0.853866</td>
          <td>28.666915</td>
          <td>0.776310</td>
          <td>26.918080</td>
          <td>0.183488</td>
          <td>26.480436</td>
          <td>0.202766</td>
          <td>26.359092</td>
          <td>0.333474</td>
          <td>25.376257</td>
          <td>0.317571</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.926797</td>
          <td>0.219247</td>
          <td>26.538578</td>
          <td>0.141454</td>
          <td>25.919081</td>
          <td>0.134307</td>
          <td>25.722394</td>
          <td>0.210888</td>
          <td>25.722337</td>
          <td>0.441426</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.212444</td>
          <td>0.307967</td>
          <td>26.645375</td>
          <td>0.162163</td>
          <td>26.184373</td>
          <td>0.096430</td>
          <td>25.851861</td>
          <td>0.117198</td>
          <td>25.069928</td>
          <td>0.112090</td>
          <td>25.500548</td>
          <td>0.347204</td>
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
