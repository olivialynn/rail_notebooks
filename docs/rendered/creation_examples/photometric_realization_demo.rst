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

    <pzflow.flow.Flow at 0x7f2055ef21d0>



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
          <td>26.549883</td>
          <td>0.391877</td>
          <td>26.628862</td>
          <td>0.154700</td>
          <td>25.878587</td>
          <td>0.070823</td>
          <td>25.382159</td>
          <td>0.074493</td>
          <td>25.023041</td>
          <td>0.103512</td>
          <td>25.449511</td>
          <td>0.321529</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.594030</td>
          <td>0.824024</td>
          <td>28.313042</td>
          <td>0.589385</td>
          <td>27.907272</td>
          <td>0.391866</td>
          <td>27.169767</td>
          <td>0.339391</td>
          <td>26.180160</td>
          <td>0.276257</td>
          <td>25.400023</td>
          <td>0.309068</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>28.010351</td>
          <td>1.064911</td>
          <td>25.926940</td>
          <td>0.084028</td>
          <td>24.803436</td>
          <td>0.027360</td>
          <td>23.868391</td>
          <td>0.019696</td>
          <td>23.118810</td>
          <td>0.019432</td>
          <td>22.867367</td>
          <td>0.034684</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.954833</td>
          <td>0.453468</td>
          <td>27.862735</td>
          <td>0.378572</td>
          <td>26.519327</td>
          <td>0.199489</td>
          <td>26.078982</td>
          <td>0.254356</td>
          <td>25.100246</td>
          <td>0.242211</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.234852</td>
          <td>0.305843</td>
          <td>25.704644</td>
          <td>0.069066</td>
          <td>25.447734</td>
          <td>0.048327</td>
          <td>24.813200</td>
          <td>0.044973</td>
          <td>24.426064</td>
          <td>0.061140</td>
          <td>23.713247</td>
          <td>0.073407</td>
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
          <td>26.034862</td>
          <td>0.260154</td>
          <td>26.658703</td>
          <td>0.158699</td>
          <td>26.291830</td>
          <td>0.101916</td>
          <td>26.064094</td>
          <td>0.135315</td>
          <td>25.596664</td>
          <td>0.169913</td>
          <td>25.429667</td>
          <td>0.316482</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.774645</td>
          <td>0.464902</td>
          <td>26.904123</td>
          <td>0.195416</td>
          <td>26.710735</td>
          <td>0.146613</td>
          <td>26.449538</td>
          <td>0.188101</td>
          <td>26.075026</td>
          <td>0.253531</td>
          <td>25.962826</td>
          <td>0.477814</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.409989</td>
          <td>0.296487</td>
          <td>26.859189</td>
          <td>0.166479</td>
          <td>26.171370</td>
          <td>0.148414</td>
          <td>26.816504</td>
          <td>0.454932</td>
          <td>25.275937</td>
          <td>0.279648</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.093877</td>
          <td>0.228975</td>
          <td>26.526580</td>
          <td>0.125056</td>
          <td>25.826444</td>
          <td>0.110083</td>
          <td>25.610354</td>
          <td>0.171904</td>
          <td>25.657182</td>
          <td>0.378619</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.436243</td>
          <td>0.743016</td>
          <td>26.598192</td>
          <td>0.150689</td>
          <td>26.263955</td>
          <td>0.099458</td>
          <td>25.631703</td>
          <td>0.092822</td>
          <td>25.318760</td>
          <td>0.133874</td>
          <td>24.979147</td>
          <td>0.219080</td>
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
          <td>26.598024</td>
          <td>0.450106</td>
          <td>26.491421</td>
          <td>0.158091</td>
          <td>26.055652</td>
          <td>0.097365</td>
          <td>25.261316</td>
          <td>0.079323</td>
          <td>25.069372</td>
          <td>0.126484</td>
          <td>24.870434</td>
          <td>0.234366</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.638190</td>
          <td>1.459969</td>
          <td>27.284989</td>
          <td>0.276787</td>
          <td>26.874378</td>
          <td>0.312558</td>
          <td>28.035485</td>
          <td>1.153657</td>
          <td>25.808457</td>
          <td>0.490662</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.073589</td>
          <td>0.644176</td>
          <td>25.944484</td>
          <td>0.100462</td>
          <td>24.845704</td>
          <td>0.034157</td>
          <td>23.892092</td>
          <td>0.024261</td>
          <td>23.133718</td>
          <td>0.023564</td>
          <td>22.846468</td>
          <td>0.041257</td>
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
          <td>27.428567</td>
          <td>0.330198</td>
          <td>26.424559</td>
          <td>0.231006</td>
          <td>25.928847</td>
          <td>0.278197</td>
          <td>25.990157</td>
          <td>0.592410</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.835719</td>
          <td>0.246634</td>
          <td>25.918464</td>
          <td>0.096271</td>
          <td>25.495774</td>
          <td>0.059418</td>
          <td>24.882653</td>
          <td>0.056752</td>
          <td>24.446437</td>
          <td>0.073298</td>
          <td>23.582417</td>
          <td>0.077396</td>
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
          <td>26.628939</td>
          <td>0.467012</td>
          <td>26.329222</td>
          <td>0.140132</td>
          <td>26.045845</td>
          <td>0.098577</td>
          <td>26.113241</td>
          <td>0.169923</td>
          <td>25.589549</td>
          <td>0.201283</td>
          <td>26.004352</td>
          <td>0.575935</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.106237</td>
          <td>0.651744</td>
          <td>26.973444</td>
          <td>0.237959</td>
          <td>26.543570</td>
          <td>0.149344</td>
          <td>26.411053</td>
          <td>0.214793</td>
          <td>26.285637</td>
          <td>0.349456</td>
          <td>24.881271</td>
          <td>0.237431</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.602565</td>
          <td>0.396394</td>
          <td>26.897661</td>
          <td>0.203400</td>
          <td>26.514219</td>
          <td>0.235991</td>
          <td>25.892145</td>
          <td>0.256725</td>
          <td>25.671774</td>
          <td>0.447900</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.547691</td>
          <td>0.385611</td>
          <td>26.590933</td>
          <td>0.159713</td>
          <td>25.993186</td>
          <td>0.154985</td>
          <td>26.212723</td>
          <td>0.338082</td>
          <td>25.542183</td>
          <td>0.412653</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.534410</td>
          <td>0.431806</td>
          <td>26.471391</td>
          <td>0.156784</td>
          <td>26.166085</td>
          <td>0.108320</td>
          <td>25.567655</td>
          <td>0.104907</td>
          <td>25.065953</td>
          <td>0.127364</td>
          <td>25.167846</td>
          <td>0.301589</td>
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
          <td>27.358217</td>
          <td>0.705124</td>
          <td>26.852590</td>
          <td>0.187132</td>
          <td>26.020888</td>
          <td>0.080324</td>
          <td>25.271573</td>
          <td>0.067557</td>
          <td>25.001992</td>
          <td>0.101635</td>
          <td>25.586848</td>
          <td>0.358437</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.442656</td>
          <td>0.746567</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.819769</td>
          <td>0.366414</td>
          <td>27.597650</td>
          <td>0.472091</td>
          <td>27.672544</td>
          <td>0.828993</td>
          <td>27.254465</td>
          <td>1.124826</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.132688</td>
          <td>0.297254</td>
          <td>25.812154</td>
          <td>0.081644</td>
          <td>24.786283</td>
          <td>0.029256</td>
          <td>23.895157</td>
          <td>0.021908</td>
          <td>23.131510</td>
          <td>0.021273</td>
          <td>22.881139</td>
          <td>0.038248</td>
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
          <td>27.273407</td>
          <td>0.290685</td>
          <td>26.867638</td>
          <td>0.329885</td>
          <td>26.331666</td>
          <td>0.381880</td>
          <td>26.639273</td>
          <td>0.911003</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.501100</td>
          <td>0.377678</td>
          <td>25.792473</td>
          <td>0.074730</td>
          <td>25.430124</td>
          <td>0.047645</td>
          <td>24.841271</td>
          <td>0.046177</td>
          <td>24.456349</td>
          <td>0.062894</td>
          <td>23.550485</td>
          <td>0.063650</td>
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
          <td>26.087154</td>
          <td>0.285842</td>
          <td>26.629147</td>
          <td>0.165472</td>
          <td>26.186635</td>
          <td>0.100539</td>
          <td>25.787786</td>
          <td>0.115502</td>
          <td>25.987125</td>
          <td>0.253855</td>
          <td>25.151561</td>
          <td>0.272564</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.064367</td>
          <td>0.226480</td>
          <td>26.780261</td>
          <td>0.158141</td>
          <td>26.164160</td>
          <td>0.150011</td>
          <td>25.732430</td>
          <td>0.193658</td>
          <td>25.425310</td>
          <td>0.320318</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>30.711418</td>
          <td>3.345446</td>
          <td>27.070325</td>
          <td>0.233777</td>
          <td>26.963400</td>
          <td>0.190648</td>
          <td>26.549380</td>
          <td>0.214808</td>
          <td>26.131372</td>
          <td>0.277786</td>
          <td>25.328868</td>
          <td>0.305757</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.032691</td>
          <td>1.138673</td>
          <td>26.738376</td>
          <td>0.187213</td>
          <td>26.537342</td>
          <td>0.141303</td>
          <td>26.004345</td>
          <td>0.144551</td>
          <td>25.409049</td>
          <td>0.161821</td>
          <td>25.583381</td>
          <td>0.396979</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.551425</td>
          <td>0.817875</td>
          <td>26.272842</td>
          <td>0.117641</td>
          <td>25.983473</td>
          <td>0.080806</td>
          <td>25.572286</td>
          <td>0.091772</td>
          <td>25.351935</td>
          <td>0.143119</td>
          <td>25.231492</td>
          <td>0.279988</td>
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
