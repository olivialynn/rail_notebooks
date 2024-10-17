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

    <pzflow.flow.Flow at 0x7f27c2d0e170>



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
          <td>26.658880</td>
          <td>0.426016</td>
          <td>26.902928</td>
          <td>0.195220</td>
          <td>26.063485</td>
          <td>0.083388</td>
          <td>25.350020</td>
          <td>0.072406</td>
          <td>25.030297</td>
          <td>0.104171</td>
          <td>25.011609</td>
          <td>0.225077</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.913418</td>
          <td>0.881884</td>
          <td>27.231023</td>
          <td>0.227648</td>
          <td>27.145806</td>
          <td>0.333014</td>
          <td>26.215880</td>
          <td>0.284377</td>
          <td>26.180200</td>
          <td>0.560247</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.375785</td>
          <td>0.713495</td>
          <td>25.922283</td>
          <td>0.083684</td>
          <td>24.749526</td>
          <td>0.026104</td>
          <td>23.878697</td>
          <td>0.019869</td>
          <td>23.144127</td>
          <td>0.019852</td>
          <td>22.807416</td>
          <td>0.032898</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.246020</td>
          <td>1.218024</td>
          <td>28.290927</td>
          <td>0.580180</td>
          <td>26.837489</td>
          <td>0.163427</td>
          <td>26.636713</td>
          <td>0.220071</td>
          <td>26.466802</td>
          <td>0.347497</td>
          <td>25.275933</td>
          <td>0.279647</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.260361</td>
          <td>0.659468</td>
          <td>25.749971</td>
          <td>0.071888</td>
          <td>25.430769</td>
          <td>0.047604</td>
          <td>24.733454</td>
          <td>0.041901</td>
          <td>24.202621</td>
          <td>0.050142</td>
          <td>23.654653</td>
          <td>0.069699</td>
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
          <td>26.608974</td>
          <td>0.410093</td>
          <td>26.305033</td>
          <td>0.116983</td>
          <td>26.250478</td>
          <td>0.098290</td>
          <td>25.718772</td>
          <td>0.100191</td>
          <td>25.552308</td>
          <td>0.163611</td>
          <td>26.144302</td>
          <td>0.545924</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.541816</td>
          <td>1.425978</td>
          <td>27.004339</td>
          <td>0.212537</td>
          <td>26.811953</td>
          <td>0.159901</td>
          <td>26.667970</td>
          <td>0.225866</td>
          <td>26.520292</td>
          <td>0.362401</td>
          <td>25.062531</td>
          <td>0.234782</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.366945</td>
          <td>0.339718</td>
          <td>26.853463</td>
          <td>0.187250</td>
          <td>27.017483</td>
          <td>0.190394</td>
          <td>26.750729</td>
          <td>0.241879</td>
          <td>26.285063</td>
          <td>0.300700</td>
          <td>25.563506</td>
          <td>0.351886</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.046648</td>
          <td>0.220167</td>
          <td>26.760219</td>
          <td>0.152975</td>
          <td>25.716229</td>
          <td>0.099968</td>
          <td>25.956917</td>
          <td>0.229997</td>
          <td>25.612572</td>
          <td>0.365683</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.907736</td>
          <td>0.513073</td>
          <td>26.382529</td>
          <td>0.125122</td>
          <td>26.246542</td>
          <td>0.097951</td>
          <td>25.636316</td>
          <td>0.093199</td>
          <td>25.203839</td>
          <td>0.121186</td>
          <td>24.589567</td>
          <td>0.157644</td>
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
          <td>27.618586</td>
          <td>0.910846</td>
          <td>26.805183</td>
          <td>0.206158</td>
          <td>26.003282</td>
          <td>0.092992</td>
          <td>25.353837</td>
          <td>0.086063</td>
          <td>24.953990</td>
          <td>0.114421</td>
          <td>25.079795</td>
          <td>0.278240</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.763822</td>
          <td>2.550219</td>
          <td>29.154181</td>
          <td>1.124349</td>
          <td>27.332077</td>
          <td>0.287552</td>
          <td>29.506379</td>
          <td>1.695058</td>
          <td>26.308804</td>
          <td>0.354620</td>
          <td>25.381250</td>
          <td>0.354069</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.217203</td>
          <td>0.710740</td>
          <td>26.043839</td>
          <td>0.109564</td>
          <td>24.773872</td>
          <td>0.032064</td>
          <td>23.903426</td>
          <td>0.024500</td>
          <td>23.166513</td>
          <td>0.024240</td>
          <td>22.893913</td>
          <td>0.043027</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.825270</td>
          <td>0.964048</td>
          <td>27.666148</td>
          <td>0.397645</td>
          <td>26.749913</td>
          <td>0.301290</td>
          <td>25.730988</td>
          <td>0.236570</td>
          <td>25.091671</td>
          <td>0.299286</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.225668</td>
          <td>0.337740</td>
          <td>25.758025</td>
          <td>0.083625</td>
          <td>25.354267</td>
          <td>0.052408</td>
          <td>24.803442</td>
          <td>0.052901</td>
          <td>24.257464</td>
          <td>0.062008</td>
          <td>23.763855</td>
          <td>0.090808</td>
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
          <td>26.965092</td>
          <td>0.596510</td>
          <td>26.446466</td>
          <td>0.154966</td>
          <td>26.304827</td>
          <td>0.123559</td>
          <td>26.074800</td>
          <td>0.164449</td>
          <td>25.747279</td>
          <td>0.229591</td>
          <td>25.474187</td>
          <td>0.387958</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.929711</td>
          <td>0.229505</td>
          <td>27.000037</td>
          <td>0.219759</td>
          <td>26.502520</td>
          <td>0.231764</td>
          <td>26.696084</td>
          <td>0.478628</td>
          <td>25.871251</td>
          <td>0.515654</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.089753</td>
          <td>1.210087</td>
          <td>27.136350</td>
          <td>0.273916</td>
          <td>26.712365</td>
          <td>0.173950</td>
          <td>26.747865</td>
          <td>0.285699</td>
          <td>26.262477</td>
          <td>0.345796</td>
          <td>25.044145</td>
          <td>0.273599</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.815362</td>
          <td>0.539338</td>
          <td>26.991692</td>
          <td>0.247211</td>
          <td>26.721861</td>
          <td>0.178541</td>
          <td>25.839645</td>
          <td>0.135817</td>
          <td>25.556249</td>
          <td>0.197689</td>
          <td>25.561026</td>
          <td>0.418643</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.054300</td>
          <td>0.630866</td>
          <td>26.301577</td>
          <td>0.135508</td>
          <td>26.246269</td>
          <td>0.116162</td>
          <td>25.483799</td>
          <td>0.097482</td>
          <td>25.080567</td>
          <td>0.128986</td>
          <td>25.441215</td>
          <td>0.374416</td>
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
          <td>27.733400</td>
          <td>0.900382</td>
          <td>26.838192</td>
          <td>0.184871</td>
          <td>25.991684</td>
          <td>0.078280</td>
          <td>25.230667</td>
          <td>0.065153</td>
          <td>25.102536</td>
          <td>0.110970</td>
          <td>24.942428</td>
          <td>0.212499</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.582260</td>
          <td>0.340408</td>
          <td>28.101272</td>
          <td>0.454720</td>
          <td>27.386379</td>
          <td>0.402222</td>
          <td>26.498788</td>
          <td>0.356646</td>
          <td>28.342389</td>
          <td>1.936522</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.798294</td>
          <td>0.497155</td>
          <td>25.957272</td>
          <td>0.092750</td>
          <td>24.760973</td>
          <td>0.028614</td>
          <td>23.886670</td>
          <td>0.021749</td>
          <td>23.150441</td>
          <td>0.021619</td>
          <td>22.838262</td>
          <td>0.036825</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.742852</td>
          <td>1.016625</td>
          <td>27.521753</td>
          <td>0.387808</td>
          <td>27.119769</td>
          <td>0.256538</td>
          <td>26.689702</td>
          <td>0.286046</td>
          <td>27.072856</td>
          <td>0.658475</td>
          <td>25.472084</td>
          <td>0.402456</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.990634</td>
          <td>0.251136</td>
          <td>25.938786</td>
          <td>0.085013</td>
          <td>25.474293</td>
          <td>0.049551</td>
          <td>24.847718</td>
          <td>0.046442</td>
          <td>24.289528</td>
          <td>0.054241</td>
          <td>23.668782</td>
          <td>0.070681</td>
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
          <td>26.529905</td>
          <td>0.405249</td>
          <td>26.376172</td>
          <td>0.133189</td>
          <td>26.476468</td>
          <td>0.129412</td>
          <td>26.133218</td>
          <td>0.155660</td>
          <td>25.775379</td>
          <td>0.213035</td>
          <td>25.310014</td>
          <td>0.309757</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.743788</td>
          <td>0.913590</td>
          <td>27.741805</td>
          <td>0.390271</td>
          <td>26.816038</td>
          <td>0.163050</td>
          <td>26.462061</td>
          <td>0.193278</td>
          <td>25.813580</td>
          <td>0.207315</td>
          <td>25.858835</td>
          <td>0.448481</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.539249</td>
          <td>1.452844</td>
          <td>26.999160</td>
          <td>0.220374</td>
          <td>26.838588</td>
          <td>0.171524</td>
          <td>26.244830</td>
          <td>0.166130</td>
          <td>25.992049</td>
          <td>0.247890</td>
          <td>25.402420</td>
          <td>0.324260</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.954661</td>
          <td>0.568045</td>
          <td>27.626821</td>
          <td>0.385310</td>
          <td>27.022878</td>
          <td>0.213391</td>
          <td>26.151125</td>
          <td>0.163920</td>
          <td>25.449142</td>
          <td>0.167450</td>
          <td>25.934883</td>
          <td>0.517116</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.004201</td>
          <td>0.562802</td>
          <td>26.706633</td>
          <td>0.170847</td>
          <td>26.233691</td>
          <td>0.100690</td>
          <td>25.672828</td>
          <td>0.100236</td>
          <td>25.307452</td>
          <td>0.137737</td>
          <td>25.087261</td>
          <td>0.248875</td>
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
