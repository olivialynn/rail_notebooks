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

    <pzflow.flow.Flow at 0x7fae7b588f40>



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
          <td>28.860464</td>
          <td>1.667654</td>
          <td>27.031560</td>
          <td>0.217418</td>
          <td>26.093440</td>
          <td>0.085619</td>
          <td>25.362714</td>
          <td>0.073223</td>
          <td>24.993823</td>
          <td>0.100898</td>
          <td>25.199020</td>
          <td>0.262670</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.612673</td>
          <td>0.833973</td>
          <td>27.892511</td>
          <td>0.432605</td>
          <td>27.826661</td>
          <td>0.368082</td>
          <td>27.433989</td>
          <td>0.416818</td>
          <td>26.244932</td>
          <td>0.291135</td>
          <td>26.723370</td>
          <td>0.812550</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.885196</td>
          <td>0.504649</td>
          <td>25.918950</td>
          <td>0.083439</td>
          <td>24.766662</td>
          <td>0.026496</td>
          <td>23.865372</td>
          <td>0.019646</td>
          <td>23.176008</td>
          <td>0.020396</td>
          <td>22.843604</td>
          <td>0.033964</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.742913</td>
          <td>1.576520</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.544484</td>
          <td>0.294222</td>
          <td>26.499085</td>
          <td>0.196123</td>
          <td>26.034517</td>
          <td>0.245228</td>
          <td>25.194449</td>
          <td>0.261691</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.855082</td>
          <td>0.224364</td>
          <td>25.700684</td>
          <td>0.068825</td>
          <td>25.466922</td>
          <td>0.049157</td>
          <td>24.755930</td>
          <td>0.042744</td>
          <td>24.324568</td>
          <td>0.055875</td>
          <td>23.631689</td>
          <td>0.068296</td>
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
          <td>26.793231</td>
          <td>0.471404</td>
          <td>26.293352</td>
          <td>0.115801</td>
          <td>26.125651</td>
          <td>0.088082</td>
          <td>26.148676</td>
          <td>0.145548</td>
          <td>26.667392</td>
          <td>0.406184</td>
          <td>25.195640</td>
          <td>0.261946</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.344673</td>
          <td>0.698632</td>
          <td>27.283402</td>
          <td>0.267588</td>
          <td>26.736442</td>
          <td>0.149886</td>
          <td>26.301187</td>
          <td>0.165853</td>
          <td>26.180258</td>
          <td>0.276279</td>
          <td>25.368685</td>
          <td>0.301395</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.700524</td>
          <td>0.768698</td>
          <td>26.755039</td>
          <td>0.152297</td>
          <td>26.552107</td>
          <td>0.205053</td>
          <td>27.032383</td>
          <td>0.533720</td>
          <td>25.478799</td>
          <td>0.329107</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.959684</td>
          <td>0.532907</td>
          <td>27.685157</td>
          <td>0.368778</td>
          <td>26.618822</td>
          <td>0.135450</td>
          <td>25.798509</td>
          <td>0.107430</td>
          <td>25.654394</td>
          <td>0.178453</td>
          <td>25.151587</td>
          <td>0.252661</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.681284</td>
          <td>0.871273</td>
          <td>26.568889</td>
          <td>0.146947</td>
          <td>26.029414</td>
          <td>0.080920</td>
          <td>25.506435</td>
          <td>0.083132</td>
          <td>25.224052</td>
          <td>0.123331</td>
          <td>24.607282</td>
          <td>0.160050</td>
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
          <td>27.455253</td>
          <td>0.752473</td>
          <td>26.676996</td>
          <td>0.161198</td>
          <td>25.987154</td>
          <td>0.077957</td>
          <td>25.450481</td>
          <td>0.079128</td>
          <td>25.082981</td>
          <td>0.109079</td>
          <td>25.240800</td>
          <td>0.271776</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.707226</td>
          <td>0.885657</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.700835</td>
          <td>0.333394</td>
          <td>26.871044</td>
          <td>0.266970</td>
          <td>26.409791</td>
          <td>0.332187</td>
          <td>25.881037</td>
          <td>0.449409</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.967760</td>
          <td>0.536043</td>
          <td>26.061347</td>
          <td>0.094559</td>
          <td>24.802291</td>
          <td>0.027333</td>
          <td>23.845509</td>
          <td>0.019319</td>
          <td>23.127381</td>
          <td>0.019573</td>
          <td>22.823345</td>
          <td>0.033363</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.766900</td>
          <td>0.919322</td>
          <td>28.465663</td>
          <td>0.655918</td>
          <td>26.967602</td>
          <td>0.182537</td>
          <td>27.061569</td>
          <td>0.311406</td>
          <td>25.864843</td>
          <td>0.213034</td>
          <td>24.571163</td>
          <td>0.155180</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.531605</td>
          <td>0.386380</td>
          <td>25.805702</td>
          <td>0.075513</td>
          <td>25.416535</td>
          <td>0.047006</td>
          <td>24.817369</td>
          <td>0.045139</td>
          <td>24.323326</td>
          <td>0.055813</td>
          <td>23.706873</td>
          <td>0.072994</td>
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
          <td>26.098638</td>
          <td>0.274024</td>
          <td>26.411639</td>
          <td>0.128316</td>
          <td>26.018869</td>
          <td>0.080171</td>
          <td>26.314888</td>
          <td>0.167801</td>
          <td>25.904203</td>
          <td>0.220142</td>
          <td>25.351680</td>
          <td>0.297300</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.047056</td>
          <td>0.567600</td>
          <td>26.891357</td>
          <td>0.193328</td>
          <td>26.804613</td>
          <td>0.158901</td>
          <td>26.334857</td>
          <td>0.170678</td>
          <td>25.799243</td>
          <td>0.201649</td>
          <td>25.105721</td>
          <td>0.243307</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.537356</td>
          <td>0.794268</td>
          <td>26.974545</td>
          <td>0.207310</td>
          <td>26.744101</td>
          <td>0.150875</td>
          <td>26.742086</td>
          <td>0.240161</td>
          <td>26.212771</td>
          <td>0.283662</td>
          <td>26.154125</td>
          <td>0.549815</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.512983</td>
          <td>0.380846</td>
          <td>27.111450</td>
          <td>0.232333</td>
          <td>26.455469</td>
          <td>0.117564</td>
          <td>25.741586</td>
          <td>0.102213</td>
          <td>25.516735</td>
          <td>0.158714</td>
          <td>25.367259</td>
          <td>0.301049</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.790814</td>
          <td>0.470555</td>
          <td>26.477569</td>
          <td>0.135839</td>
          <td>25.987316</td>
          <td>0.077968</td>
          <td>25.584428</td>
          <td>0.089043</td>
          <td>25.377137</td>
          <td>0.140790</td>
          <td>24.483815</td>
          <td>0.143970</td>
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
          <td>27.240391</td>
          <td>0.650429</td>
          <td>26.629493</td>
          <td>0.154784</td>
          <td>26.156625</td>
          <td>0.090515</td>
          <td>25.343413</td>
          <td>0.071984</td>
          <td>24.954373</td>
          <td>0.097469</td>
          <td>24.510575</td>
          <td>0.147321</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.665857</td>
          <td>0.363260</td>
          <td>27.285462</td>
          <td>0.238144</td>
          <td>27.199168</td>
          <td>0.347357</td>
          <td>26.311091</td>
          <td>0.307049</td>
          <td>28.812629</td>
          <td>2.339659</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.940865</td>
          <td>0.525654</td>
          <td>25.879907</td>
          <td>0.080619</td>
          <td>24.768637</td>
          <td>0.026542</td>
          <td>23.901372</td>
          <td>0.020255</td>
          <td>23.144287</td>
          <td>0.019855</td>
          <td>22.826535</td>
          <td>0.033457</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.413403</td>
          <td>1.333640</td>
          <td>29.968071</td>
          <td>1.586005</td>
          <td>27.475459</td>
          <td>0.278243</td>
          <td>26.606186</td>
          <td>0.214540</td>
          <td>26.513383</td>
          <td>0.360446</td>
          <td>25.439499</td>
          <td>0.318974</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.710469</td>
          <td>0.198872</td>
          <td>25.753997</td>
          <td>0.072144</td>
          <td>25.411178</td>
          <td>0.046783</td>
          <td>24.835388</td>
          <td>0.045867</td>
          <td>24.361429</td>
          <td>0.057733</td>
          <td>23.702121</td>
          <td>0.072688</td>
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
          <td>26.603640</td>
          <td>0.408420</td>
          <td>26.298134</td>
          <td>0.116283</td>
          <td>26.070767</td>
          <td>0.083925</td>
          <td>25.892692</td>
          <td>0.116626</td>
          <td>25.618028</td>
          <td>0.173028</td>
          <td>25.682372</td>
          <td>0.386092</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.327878</td>
          <td>0.690700</td>
          <td>27.000327</td>
          <td>0.211827</td>
          <td>26.708225</td>
          <td>0.146297</td>
          <td>26.172512</td>
          <td>0.148560</td>
          <td>26.240759</td>
          <td>0.290156</td>
          <td>25.123683</td>
          <td>0.246933</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.329570</td>
          <td>2.051451</td>
          <td>27.456487</td>
          <td>0.307768</td>
          <td>27.003270</td>
          <td>0.188124</td>
          <td>26.490554</td>
          <td>0.194719</td>
          <td>26.142953</td>
          <td>0.268017</td>
          <td>25.918426</td>
          <td>0.462221</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>30.049259</td>
          <td>2.688488</td>
          <td>27.350630</td>
          <td>0.282612</td>
          <td>26.673844</td>
          <td>0.142032</td>
          <td>25.876814</td>
          <td>0.115025</td>
          <td>25.572470</td>
          <td>0.166448</td>
          <td>25.293068</td>
          <td>0.283558</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.693672</td>
          <td>0.437414</td>
          <td>26.442632</td>
          <td>0.131802</td>
          <td>26.147163</td>
          <td>0.089765</td>
          <td>25.593034</td>
          <td>0.089720</td>
          <td>25.110581</td>
          <td>0.111737</td>
          <td>24.850588</td>
          <td>0.196728</td>
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
