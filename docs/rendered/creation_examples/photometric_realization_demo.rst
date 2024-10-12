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

    <pzflow.flow.Flow at 0x7f0bf1396410>



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
          <td>27.735820</td>
          <td>0.901688</td>
          <td>26.879117</td>
          <td>0.191345</td>
          <td>26.007399</td>
          <td>0.079363</td>
          <td>25.337422</td>
          <td>0.071603</td>
          <td>24.957362</td>
          <td>0.097725</td>
          <td>24.894842</td>
          <td>0.204176</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.914049</td>
          <td>0.515452</td>
          <td>27.831295</td>
          <td>0.412879</td>
          <td>27.667650</td>
          <td>0.324724</td>
          <td>26.749622</td>
          <td>0.241659</td>
          <td>26.066331</td>
          <td>0.251728</td>
          <td>26.356806</td>
          <td>0.634902</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.632367</td>
          <td>0.417494</td>
          <td>25.897905</td>
          <td>0.081907</td>
          <td>24.747099</td>
          <td>0.026049</td>
          <td>23.874463</td>
          <td>0.019798</td>
          <td>23.138156</td>
          <td>0.019752</td>
          <td>22.839276</td>
          <td>0.033835</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.422813</td>
          <td>3.035124</td>
          <td>32.844524</td>
          <td>4.198637</td>
          <td>28.152127</td>
          <td>0.472003</td>
          <td>27.099899</td>
          <td>0.321084</td>
          <td>25.672822</td>
          <td>0.181261</td>
          <td>25.072284</td>
          <td>0.236684</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.928327</td>
          <td>0.238375</td>
          <td>25.775805</td>
          <td>0.073547</td>
          <td>25.425825</td>
          <td>0.047396</td>
          <td>24.807239</td>
          <td>0.044735</td>
          <td>24.305877</td>
          <td>0.054955</td>
          <td>23.687667</td>
          <td>0.071765</td>
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
          <td>26.498493</td>
          <td>0.376587</td>
          <td>26.466101</td>
          <td>0.134501</td>
          <td>26.084624</td>
          <td>0.084956</td>
          <td>26.202236</td>
          <td>0.152398</td>
          <td>25.565738</td>
          <td>0.165496</td>
          <td>25.368771</td>
          <td>0.301415</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.643057</td>
          <td>0.850359</td>
          <td>27.003744</td>
          <td>0.212432</td>
          <td>26.707038</td>
          <td>0.146148</td>
          <td>26.529816</td>
          <td>0.201254</td>
          <td>26.449339</td>
          <td>0.342745</td>
          <td>26.191745</td>
          <td>0.564915</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.238569</td>
          <td>0.257965</td>
          <td>26.985485</td>
          <td>0.185319</td>
          <td>26.403246</td>
          <td>0.180879</td>
          <td>26.076900</td>
          <td>0.253921</td>
          <td>25.371871</td>
          <td>0.302167</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.230191</td>
          <td>0.645848</td>
          <td>27.098313</td>
          <td>0.229819</td>
          <td>26.607882</td>
          <td>0.134176</td>
          <td>26.185059</td>
          <td>0.150169</td>
          <td>25.413322</td>
          <td>0.145245</td>
          <td>26.676970</td>
          <td>0.788356</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.265446</td>
          <td>0.313418</td>
          <td>26.728824</td>
          <td>0.168478</td>
          <td>26.040835</td>
          <td>0.081740</td>
          <td>25.760582</td>
          <td>0.103926</td>
          <td>25.201470</td>
          <td>0.120936</td>
          <td>25.008112</td>
          <td>0.224424</td>
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
          <td>26.986927</td>
          <td>0.598034</td>
          <td>26.615697</td>
          <td>0.175733</td>
          <td>26.134418</td>
          <td>0.104317</td>
          <td>25.403246</td>
          <td>0.089887</td>
          <td>24.963951</td>
          <td>0.115417</td>
          <td>25.254587</td>
          <td>0.320245</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.592229</td>
          <td>0.896066</td>
          <td>28.799677</td>
          <td>0.909375</td>
          <td>27.529432</td>
          <td>0.336731</td>
          <td>26.830023</td>
          <td>0.301641</td>
          <td>26.034487</td>
          <td>0.284938</td>
          <td>25.957703</td>
          <td>0.547302</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.599321</td>
          <td>0.457167</td>
          <td>25.728597</td>
          <td>0.083130</td>
          <td>24.819009</td>
          <td>0.033363</td>
          <td>23.858751</td>
          <td>0.023573</td>
          <td>23.162660</td>
          <td>0.024160</td>
          <td>22.835858</td>
          <td>0.040871</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.525243</td>
          <td>2.388880</td>
          <td>28.605849</td>
          <td>0.840382</td>
          <td>27.863221</td>
          <td>0.461929</td>
          <td>26.642216</td>
          <td>0.276181</td>
          <td>26.740551</td>
          <td>0.521406</td>
          <td>24.767906</td>
          <td>0.229728</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.892016</td>
          <td>0.258275</td>
          <td>25.661810</td>
          <td>0.076832</td>
          <td>25.391663</td>
          <td>0.054176</td>
          <td>24.894475</td>
          <td>0.057350</td>
          <td>24.303926</td>
          <td>0.064614</td>
          <td>23.764408</td>
          <td>0.090852</td>
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
          <td>27.534127</td>
          <td>0.873754</td>
          <td>26.264007</td>
          <td>0.132470</td>
          <td>26.035510</td>
          <td>0.097689</td>
          <td>25.811169</td>
          <td>0.131120</td>
          <td>25.493236</td>
          <td>0.185601</td>
          <td>25.600977</td>
          <td>0.427606</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.926682</td>
          <td>0.574431</td>
          <td>27.089211</td>
          <td>0.261698</td>
          <td>27.080091</td>
          <td>0.234855</td>
          <td>26.319636</td>
          <td>0.198964</td>
          <td>26.355448</td>
          <td>0.369106</td>
          <td>25.384933</td>
          <td>0.356389</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.378489</td>
          <td>0.786852</td>
          <td>27.130428</td>
          <td>0.272600</td>
          <td>26.908931</td>
          <td>0.205330</td>
          <td>26.875427</td>
          <td>0.316544</td>
          <td>29.034934</td>
          <td>1.911770</td>
          <td>25.519947</td>
          <td>0.398944</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.919379</td>
          <td>0.232897</td>
          <td>26.769654</td>
          <td>0.185911</td>
          <td>25.941186</td>
          <td>0.148226</td>
          <td>25.309448</td>
          <td>0.160373</td>
          <td>28.776774</td>
          <td>2.509807</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>30.111916</td>
          <td>2.877380</td>
          <td>26.503446</td>
          <td>0.161135</td>
          <td>25.939242</td>
          <td>0.088791</td>
          <td>25.470919</td>
          <td>0.096387</td>
          <td>25.021990</td>
          <td>0.122601</td>
          <td>25.649707</td>
          <td>0.439426</td>
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
          <td>27.384383</td>
          <td>0.717692</td>
          <td>26.503282</td>
          <td>0.138899</td>
          <td>26.121299</td>
          <td>0.087757</td>
          <td>25.226299</td>
          <td>0.064901</td>
          <td>24.898125</td>
          <td>0.092786</td>
          <td>24.699342</td>
          <td>0.173135</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>30.345677</td>
          <td>2.963560</td>
          <td>27.703655</td>
          <td>0.374404</td>
          <td>26.969327</td>
          <td>0.182970</td>
          <td>27.902702</td>
          <td>0.589640</td>
          <td>27.818092</td>
          <td>0.909117</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.904860</td>
          <td>0.537482</td>
          <td>26.161190</td>
          <td>0.110853</td>
          <td>24.813480</td>
          <td>0.029961</td>
          <td>23.868154</td>
          <td>0.021408</td>
          <td>23.139062</td>
          <td>0.021410</td>
          <td>22.838203</td>
          <td>0.036823</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.727503</td>
          <td>0.516484</td>
          <td>27.522313</td>
          <td>0.387976</td>
          <td>27.402886</td>
          <td>0.322484</td>
          <td>27.055849</td>
          <td>0.382397</td>
          <td>25.477967</td>
          <td>0.190868</td>
          <td>24.985199</td>
          <td>0.273670</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.295672</td>
          <td>0.321346</td>
          <td>25.775998</td>
          <td>0.073651</td>
          <td>25.334772</td>
          <td>0.043779</td>
          <td>24.746066</td>
          <td>0.042436</td>
          <td>24.461131</td>
          <td>0.063161</td>
          <td>23.822419</td>
          <td>0.080957</td>
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
          <td>25.999957</td>
          <td>0.266322</td>
          <td>26.352526</td>
          <td>0.130495</td>
          <td>26.150920</td>
          <td>0.097441</td>
          <td>26.213931</td>
          <td>0.166770</td>
          <td>25.856752</td>
          <td>0.227965</td>
          <td>28.025995</td>
          <td>1.752927</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.830564</td>
          <td>0.186210</td>
          <td>26.879641</td>
          <td>0.172128</td>
          <td>26.625741</td>
          <td>0.221669</td>
          <td>26.970995</td>
          <td>0.517352</td>
          <td>25.797412</td>
          <td>0.428093</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.865598</td>
          <td>0.439769</td>
          <td>26.610342</td>
          <td>0.141079</td>
          <td>26.491064</td>
          <td>0.204582</td>
          <td>25.965797</td>
          <td>0.242589</td>
          <td>26.218188</td>
          <td>0.599824</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>35.068075</td>
          <td>7.706199</td>
          <td>27.762785</td>
          <td>0.427704</td>
          <td>26.582805</td>
          <td>0.146940</td>
          <td>25.830889</td>
          <td>0.124435</td>
          <td>26.122009</td>
          <td>0.292871</td>
          <td>24.916349</td>
          <td>0.232625</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.725508</td>
          <td>0.173609</td>
          <td>26.159584</td>
          <td>0.094355</td>
          <td>25.767786</td>
          <td>0.108916</td>
          <td>25.271835</td>
          <td>0.133565</td>
          <td>25.016648</td>
          <td>0.234795</td>
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
