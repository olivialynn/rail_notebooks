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

    <pzflow.flow.Flow at 0x7f8109c1bc10>



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
          <td>27.089798</td>
          <td>0.585188</td>
          <td>26.812925</td>
          <td>0.180942</td>
          <td>25.978318</td>
          <td>0.077351</td>
          <td>25.202248</td>
          <td>0.063523</td>
          <td>25.000651</td>
          <td>0.101503</td>
          <td>25.440386</td>
          <td>0.319200</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.907453</td>
          <td>0.512967</td>
          <td>31.342761</td>
          <td>2.762713</td>
          <td>28.180071</td>
          <td>0.481933</td>
          <td>26.798046</td>
          <td>0.251485</td>
          <td>26.852912</td>
          <td>0.467527</td>
          <td>25.828532</td>
          <td>0.431904</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.479399</td>
          <td>0.371034</td>
          <td>25.847953</td>
          <td>0.078380</td>
          <td>24.768491</td>
          <td>0.026538</td>
          <td>23.862473</td>
          <td>0.019598</td>
          <td>23.138621</td>
          <td>0.019760</td>
          <td>22.811605</td>
          <td>0.033019</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.560533</td>
          <td>1.287597</td>
          <td>27.143145</td>
          <td>0.211582</td>
          <td>26.565166</td>
          <td>0.207308</td>
          <td>25.941630</td>
          <td>0.227099</td>
          <td>25.216125</td>
          <td>0.266365</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.998112</td>
          <td>0.252449</td>
          <td>25.794382</td>
          <td>0.074763</td>
          <td>25.369078</td>
          <td>0.045067</td>
          <td>24.846433</td>
          <td>0.046319</td>
          <td>24.316107</td>
          <td>0.055457</td>
          <td>23.614851</td>
          <td>0.067285</td>
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
          <td>26.436322</td>
          <td>0.358761</td>
          <td>26.357657</td>
          <td>0.122453</td>
          <td>26.101354</td>
          <td>0.086218</td>
          <td>25.778251</td>
          <td>0.105545</td>
          <td>25.762054</td>
          <td>0.195443</td>
          <td>25.391002</td>
          <td>0.306842</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.923197</td>
          <td>0.518915</td>
          <td>27.118265</td>
          <td>0.233647</td>
          <td>26.843773</td>
          <td>0.164305</td>
          <td>26.515911</td>
          <td>0.198917</td>
          <td>25.979516</td>
          <td>0.234343</td>
          <td>25.720329</td>
          <td>0.397584</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.952980</td>
          <td>0.452837</td>
          <td>27.160068</td>
          <td>0.214593</td>
          <td>26.326426</td>
          <td>0.169457</td>
          <td>26.561486</td>
          <td>0.374244</td>
          <td>25.195326</td>
          <td>0.261878</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.043682</td>
          <td>1.085854</td>
          <td>27.543889</td>
          <td>0.329979</td>
          <td>26.651481</td>
          <td>0.139321</td>
          <td>25.708156</td>
          <td>0.099263</td>
          <td>25.349937</td>
          <td>0.137527</td>
          <td>24.887559</td>
          <td>0.202933</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.189440</td>
          <td>0.627781</td>
          <td>26.287105</td>
          <td>0.115173</td>
          <td>26.069616</td>
          <td>0.083840</td>
          <td>25.730970</td>
          <td>0.101267</td>
          <td>25.217549</td>
          <td>0.122637</td>
          <td>24.728667</td>
          <td>0.177476</td>
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
          <td>28.239106</td>
          <td>1.304880</td>
          <td>26.953212</td>
          <td>0.233186</td>
          <td>26.013249</td>
          <td>0.093810</td>
          <td>25.345948</td>
          <td>0.085467</td>
          <td>25.012437</td>
          <td>0.120388</td>
          <td>24.944944</td>
          <td>0.249218</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.160489</td>
          <td>0.276423</td>
          <td>26.801632</td>
          <td>0.185355</td>
          <td>27.114952</td>
          <td>0.377865</td>
          <td>26.540006</td>
          <td>0.424091</td>
          <td>25.456949</td>
          <td>0.375654</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.049321</td>
          <td>0.297981</td>
          <td>25.990748</td>
          <td>0.104606</td>
          <td>24.760999</td>
          <td>0.031703</td>
          <td>23.885388</td>
          <td>0.024121</td>
          <td>23.140851</td>
          <td>0.023710</td>
          <td>22.826544</td>
          <td>0.040535</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.474843</td>
          <td>0.863522</td>
          <td>27.431868</td>
          <td>0.362660</td>
          <td>26.822928</td>
          <td>0.201215</td>
          <td>26.607974</td>
          <td>0.268592</td>
          <td>26.070338</td>
          <td>0.311793</td>
          <td>25.694826</td>
          <td>0.477885</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.233396</td>
          <td>0.339807</td>
          <td>25.625108</td>
          <td>0.074385</td>
          <td>25.516372</td>
          <td>0.060513</td>
          <td>24.812193</td>
          <td>0.053313</td>
          <td>24.265807</td>
          <td>0.062468</td>
          <td>23.785401</td>
          <td>0.092543</td>
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
          <td>26.597270</td>
          <td>0.456062</td>
          <td>26.408578</td>
          <td>0.150018</td>
          <td>26.122818</td>
          <td>0.105447</td>
          <td>25.848403</td>
          <td>0.135407</td>
          <td>26.040099</td>
          <td>0.291753</td>
          <td>25.236693</td>
          <td>0.321945</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.651279</td>
          <td>1.611415</td>
          <td>26.899085</td>
          <td>0.223747</td>
          <td>27.202453</td>
          <td>0.259730</td>
          <td>26.552495</td>
          <td>0.241538</td>
          <td>25.805615</td>
          <td>0.237163</td>
          <td>25.843398</td>
          <td>0.505212</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.186587</td>
          <td>2.052689</td>
          <td>26.941134</td>
          <td>0.233390</td>
          <td>26.647382</td>
          <td>0.164591</td>
          <td>26.623235</td>
          <td>0.258138</td>
          <td>26.235799</td>
          <td>0.338590</td>
          <td>24.852925</td>
          <td>0.233870</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.799640</td>
          <td>2.607316</td>
          <td>27.419802</td>
          <td>0.348970</td>
          <td>26.580066</td>
          <td>0.158237</td>
          <td>25.830142</td>
          <td>0.134708</td>
          <td>25.498355</td>
          <td>0.188280</td>
          <td>25.088235</td>
          <td>0.288600</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.883940</td>
          <td>0.559133</td>
          <td>26.414883</td>
          <td>0.149379</td>
          <td>26.070839</td>
          <td>0.099664</td>
          <td>25.608429</td>
          <td>0.108711</td>
          <td>25.220171</td>
          <td>0.145496</td>
          <td>25.220802</td>
          <td>0.314660</td>
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
          <td>27.935957</td>
          <td>1.019097</td>
          <td>26.699256</td>
          <td>0.164306</td>
          <td>26.099802</td>
          <td>0.086111</td>
          <td>25.270835</td>
          <td>0.067513</td>
          <td>24.996250</td>
          <td>0.101126</td>
          <td>24.696701</td>
          <td>0.172747</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.289861</td>
          <td>0.580125</td>
          <td>26.998973</td>
          <td>0.187613</td>
          <td>27.166456</td>
          <td>0.338806</td>
          <td>26.428958</td>
          <td>0.337556</td>
          <td>26.009660</td>
          <td>0.495123</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.868900</td>
          <td>0.239808</td>
          <td>25.832058</td>
          <td>0.083087</td>
          <td>24.790530</td>
          <td>0.029365</td>
          <td>23.813014</td>
          <td>0.020425</td>
          <td>23.172781</td>
          <td>0.022037</td>
          <td>22.798967</td>
          <td>0.035569</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>32.221884</td>
          <td>4.966039</td>
          <td>28.757367</td>
          <td>0.922494</td>
          <td>27.741471</td>
          <td>0.419985</td>
          <td>26.428358</td>
          <td>0.230931</td>
          <td>25.837510</td>
          <td>0.257378</td>
          <td>26.018667</td>
          <td>0.602712</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.388160</td>
          <td>0.345752</td>
          <td>25.768518</td>
          <td>0.073166</td>
          <td>25.432245</td>
          <td>0.047735</td>
          <td>24.847609</td>
          <td>0.046438</td>
          <td>24.336126</td>
          <td>0.056532</td>
          <td>23.751815</td>
          <td>0.076065</td>
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
          <td>27.133348</td>
          <td>0.631051</td>
          <td>26.510774</td>
          <td>0.149547</td>
          <td>26.123011</td>
          <td>0.095084</td>
          <td>25.949531</td>
          <td>0.132904</td>
          <td>25.856967</td>
          <td>0.228006</td>
          <td>25.478995</td>
          <td>0.354180</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.963329</td>
          <td>0.539410</td>
          <td>27.168362</td>
          <td>0.246801</td>
          <td>26.712028</td>
          <td>0.149159</td>
          <td>26.819266</td>
          <td>0.260056</td>
          <td>26.276307</td>
          <td>0.303116</td>
          <td>25.505016</td>
          <td>0.341229</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.568493</td>
          <td>0.349625</td>
          <td>26.634995</td>
          <td>0.144105</td>
          <td>26.335703</td>
          <td>0.179470</td>
          <td>25.816985</td>
          <td>0.214415</td>
          <td>25.383193</td>
          <td>0.319332</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.014069</td>
          <td>0.592614</td>
          <td>27.274947</td>
          <td>0.291685</td>
          <td>26.916992</td>
          <td>0.195265</td>
          <td>25.704105</td>
          <td>0.111442</td>
          <td>25.589690</td>
          <td>0.188643</td>
          <td>25.272710</td>
          <td>0.310974</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.647784</td>
          <td>0.869899</td>
          <td>26.649031</td>
          <td>0.162670</td>
          <td>26.182748</td>
          <td>0.096293</td>
          <td>25.626373</td>
          <td>0.096236</td>
          <td>25.251477</td>
          <td>0.131235</td>
          <td>24.653750</td>
          <td>0.173167</td>
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
