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

    <pzflow.flow.Flow at 0x7f9b9994ab90>



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
          <td>26.965221</td>
          <td>0.535056</td>
          <td>26.378368</td>
          <td>0.124672</td>
          <td>26.014686</td>
          <td>0.079875</td>
          <td>25.489247</td>
          <td>0.081881</td>
          <td>24.969925</td>
          <td>0.098807</td>
          <td>24.935680</td>
          <td>0.211277</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.835370</td>
          <td>0.958939</td>
          <td>27.558655</td>
          <td>0.333865</td>
          <td>27.900524</td>
          <td>0.389827</td>
          <td>26.968332</td>
          <td>0.288913</td>
          <td>26.441588</td>
          <td>0.340653</td>
          <td>26.301487</td>
          <td>0.610763</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.435427</td>
          <td>0.358510</td>
          <td>25.948482</td>
          <td>0.085635</td>
          <td>24.784989</td>
          <td>0.026923</td>
          <td>23.866756</td>
          <td>0.019669</td>
          <td>23.159135</td>
          <td>0.020106</td>
          <td>22.806032</td>
          <td>0.032858</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.581492</td>
          <td>3.184741</td>
          <td>28.786046</td>
          <td>0.812917</td>
          <td>27.364802</td>
          <td>0.254217</td>
          <td>26.368735</td>
          <td>0.175663</td>
          <td>25.950028</td>
          <td>0.228687</td>
          <td>25.338903</td>
          <td>0.294256</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.031070</td>
          <td>0.561126</td>
          <td>25.802422</td>
          <td>0.075295</td>
          <td>25.409797</td>
          <td>0.046726</td>
          <td>24.816884</td>
          <td>0.045120</td>
          <td>24.445414</td>
          <td>0.062198</td>
          <td>23.684871</td>
          <td>0.071587</td>
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
          <td>26.480572</td>
          <td>0.371373</td>
          <td>26.407152</td>
          <td>0.127819</td>
          <td>26.257634</td>
          <td>0.098908</td>
          <td>26.272021</td>
          <td>0.161776</td>
          <td>25.930162</td>
          <td>0.224947</td>
          <td>25.657824</td>
          <td>0.378808</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.222990</td>
          <td>0.642628</td>
          <td>27.259525</td>
          <td>0.262424</td>
          <td>26.847929</td>
          <td>0.164889</td>
          <td>26.509386</td>
          <td>0.197829</td>
          <td>26.229616</td>
          <td>0.287555</td>
          <td>24.926341</td>
          <td>0.209634</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.354997</td>
          <td>0.283613</td>
          <td>26.912462</td>
          <td>0.174199</td>
          <td>26.294063</td>
          <td>0.164848</td>
          <td>25.567528</td>
          <td>0.165749</td>
          <td>25.398785</td>
          <td>0.308762</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.102824</td>
          <td>0.590629</td>
          <td>27.093629</td>
          <td>0.228928</td>
          <td>27.159054</td>
          <td>0.214412</td>
          <td>25.995869</td>
          <td>0.127560</td>
          <td>25.822759</td>
          <td>0.205665</td>
          <td>25.389626</td>
          <td>0.306504</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.469935</td>
          <td>0.368308</td>
          <td>26.467186</td>
          <td>0.134627</td>
          <td>26.191234</td>
          <td>0.093310</td>
          <td>25.626645</td>
          <td>0.092410</td>
          <td>25.341793</td>
          <td>0.136564</td>
          <td>25.029361</td>
          <td>0.228419</td>
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
          <td>26.523544</td>
          <td>0.425436</td>
          <td>26.904362</td>
          <td>0.223932</td>
          <td>26.129743</td>
          <td>0.103891</td>
          <td>25.547651</td>
          <td>0.102026</td>
          <td>24.878288</td>
          <td>0.107110</td>
          <td>25.133659</td>
          <td>0.290640</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.050015</td>
          <td>0.548514</td>
          <td>27.463792</td>
          <td>0.319626</td>
          <td>27.341987</td>
          <td>0.449614</td>
          <td>26.734783</td>
          <td>0.490936</td>
          <td>27.614976</td>
          <td>1.509290</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.585268</td>
          <td>0.452366</td>
          <td>25.926754</td>
          <td>0.098916</td>
          <td>24.811444</td>
          <td>0.033142</td>
          <td>23.895867</td>
          <td>0.024341</td>
          <td>23.136511</td>
          <td>0.023621</td>
          <td>22.755791</td>
          <td>0.038076</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.881456</td>
          <td>0.510161</td>
          <td>27.366198</td>
          <td>0.314200</td>
          <td>26.614211</td>
          <td>0.269960</td>
          <td>26.291068</td>
          <td>0.371181</td>
          <td>25.171166</td>
          <td>0.318957</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.601257</td>
          <td>0.203058</td>
          <td>25.910682</td>
          <td>0.095617</td>
          <td>25.537815</td>
          <td>0.061674</td>
          <td>24.855517</td>
          <td>0.055402</td>
          <td>24.446674</td>
          <td>0.073313</td>
          <td>23.730902</td>
          <td>0.088215</td>
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
          <td>26.612355</td>
          <td>0.461252</td>
          <td>26.272688</td>
          <td>0.133467</td>
          <td>26.054949</td>
          <td>0.099367</td>
          <td>25.789072</td>
          <td>0.128636</td>
          <td>26.405214</td>
          <td>0.389398</td>
          <td>25.426702</td>
          <td>0.373917</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.345348</td>
          <td>0.766119</td>
          <td>26.858995</td>
          <td>0.216407</td>
          <td>26.999264</td>
          <td>0.219617</td>
          <td>26.651724</td>
          <td>0.262042</td>
          <td>25.967164</td>
          <td>0.270773</td>
          <td>24.951785</td>
          <td>0.251629</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.862340</td>
          <td>1.062709</td>
          <td>27.451176</td>
          <td>0.352330</td>
          <td>26.881249</td>
          <td>0.200618</td>
          <td>26.230252</td>
          <td>0.186114</td>
          <td>25.925384</td>
          <td>0.263801</td>
          <td>26.130424</td>
          <td>0.625389</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.117512</td>
          <td>0.667669</td>
          <td>27.142630</td>
          <td>0.279639</td>
          <td>26.373194</td>
          <td>0.132450</td>
          <td>25.796364</td>
          <td>0.130831</td>
          <td>25.742573</td>
          <td>0.230953</td>
          <td>26.098240</td>
          <td>0.620835</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.142823</td>
          <td>0.318380</td>
          <td>26.572628</td>
          <td>0.170914</td>
          <td>25.936966</td>
          <td>0.088614</td>
          <td>25.625737</td>
          <td>0.110366</td>
          <td>25.043859</td>
          <td>0.124949</td>
          <td>25.613221</td>
          <td>0.427424</td>
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
          <td>27.015264</td>
          <td>0.554823</td>
          <td>26.856435</td>
          <td>0.187740</td>
          <td>26.052960</td>
          <td>0.082629</td>
          <td>25.374043</td>
          <td>0.073971</td>
          <td>24.930060</td>
          <td>0.095425</td>
          <td>25.184017</td>
          <td>0.259500</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.375299</td>
          <td>0.288525</td>
          <td>26.908277</td>
          <td>0.173740</td>
          <td>27.355157</td>
          <td>0.392657</td>
          <td>26.060952</td>
          <td>0.250840</td>
          <td>26.128367</td>
          <td>0.540096</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.942692</td>
          <td>0.254790</td>
          <td>25.895976</td>
          <td>0.087891</td>
          <td>24.805186</td>
          <td>0.029744</td>
          <td>23.911854</td>
          <td>0.022223</td>
          <td>23.147688</td>
          <td>0.021569</td>
          <td>22.880711</td>
          <td>0.038234</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.383353</td>
          <td>0.812649</td>
          <td>28.574982</td>
          <td>0.821897</td>
          <td>27.140261</td>
          <td>0.260877</td>
          <td>26.537778</td>
          <td>0.252738</td>
          <td>26.571741</td>
          <td>0.458714</td>
          <td>26.051400</td>
          <td>0.616772</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.480295</td>
          <td>0.371617</td>
          <td>25.673001</td>
          <td>0.067245</td>
          <td>25.352803</td>
          <td>0.044485</td>
          <td>24.822885</td>
          <td>0.045430</td>
          <td>24.299615</td>
          <td>0.054729</td>
          <td>23.795249</td>
          <td>0.079039</td>
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
          <td>26.513247</td>
          <td>0.400095</td>
          <td>26.465238</td>
          <td>0.143812</td>
          <td>26.008361</td>
          <td>0.085968</td>
          <td>26.292367</td>
          <td>0.178268</td>
          <td>25.787594</td>
          <td>0.215218</td>
          <td>25.382186</td>
          <td>0.328108</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.346026</td>
          <td>1.295513</td>
          <td>26.679513</td>
          <td>0.163808</td>
          <td>26.764546</td>
          <td>0.156029</td>
          <td>26.717452</td>
          <td>0.239177</td>
          <td>26.151487</td>
          <td>0.274031</td>
          <td>25.872569</td>
          <td>0.453145</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.724260</td>
          <td>0.461024</td>
          <td>26.968921</td>
          <td>0.214892</td>
          <td>26.834549</td>
          <td>0.170936</td>
          <td>26.241140</td>
          <td>0.165608</td>
          <td>25.704188</td>
          <td>0.195071</td>
          <td>25.188532</td>
          <td>0.272985</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.986750</td>
          <td>0.230438</td>
          <td>26.546043</td>
          <td>0.142366</td>
          <td>25.886363</td>
          <td>0.130561</td>
          <td>25.481473</td>
          <td>0.172120</td>
          <td>26.179469</td>
          <td>0.616342</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.526177</td>
          <td>0.146432</td>
          <td>26.134296</td>
          <td>0.092282</td>
          <td>25.595825</td>
          <td>0.093689</td>
          <td>25.283637</td>
          <td>0.134934</td>
          <td>25.172907</td>
          <td>0.266958</td>
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
