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

    <pzflow.flow.Flow at 0x7f22601a7250>



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
          <td>26.748116</td>
          <td>0.455747</td>
          <td>26.637828</td>
          <td>0.155892</td>
          <td>25.938970</td>
          <td>0.074708</td>
          <td>25.405566</td>
          <td>0.076050</td>
          <td>25.071513</td>
          <td>0.107992</td>
          <td>24.677781</td>
          <td>0.169967</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.714243</td>
          <td>0.377227</td>
          <td>27.359967</td>
          <td>0.253211</td>
          <td>27.874786</td>
          <td>0.577566</td>
          <td>27.256281</td>
          <td>0.626195</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.191569</td>
          <td>0.628716</td>
          <td>25.844011</td>
          <td>0.078109</td>
          <td>24.787706</td>
          <td>0.026987</td>
          <td>23.857552</td>
          <td>0.019517</td>
          <td>23.095628</td>
          <td>0.019056</td>
          <td>22.804617</td>
          <td>0.032817</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.438325</td>
          <td>0.643612</td>
          <td>27.547386</td>
          <td>0.294911</td>
          <td>26.652110</td>
          <td>0.222908</td>
          <td>26.254488</td>
          <td>0.293388</td>
          <td>25.146737</td>
          <td>0.251657</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.414912</td>
          <td>0.732504</td>
          <td>25.755001</td>
          <td>0.072208</td>
          <td>25.423387</td>
          <td>0.047293</td>
          <td>24.790026</td>
          <td>0.044057</td>
          <td>24.404203</td>
          <td>0.059966</td>
          <td>23.560336</td>
          <td>0.064112</td>
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
          <td>26.566830</td>
          <td>0.397031</td>
          <td>26.380432</td>
          <td>0.124895</td>
          <td>26.310146</td>
          <td>0.103563</td>
          <td>25.900646</td>
          <td>0.117436</td>
          <td>26.532206</td>
          <td>0.365793</td>
          <td>25.791749</td>
          <td>0.419975</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.716955</td>
          <td>0.445180</td>
          <td>26.902003</td>
          <td>0.195068</td>
          <td>27.245605</td>
          <td>0.230418</td>
          <td>26.299147</td>
          <td>0.165565</td>
          <td>26.582133</td>
          <td>0.380301</td>
          <td>25.649806</td>
          <td>0.376455</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.630220</td>
          <td>0.416811</td>
          <td>27.520514</td>
          <td>0.323908</td>
          <td>27.196081</td>
          <td>0.221132</td>
          <td>26.460386</td>
          <td>0.189831</td>
          <td>26.457704</td>
          <td>0.345014</td>
          <td>25.347344</td>
          <td>0.296264</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.703404</td>
          <td>0.440647</td>
          <td>27.206896</td>
          <td>0.251353</td>
          <td>26.672036</td>
          <td>0.141811</td>
          <td>25.874111</td>
          <td>0.114754</td>
          <td>25.594426</td>
          <td>0.169590</td>
          <td>25.637849</td>
          <td>0.372967</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.471211</td>
          <td>0.135096</td>
          <td>26.004340</td>
          <td>0.079149</td>
          <td>25.707261</td>
          <td>0.099185</td>
          <td>25.206768</td>
          <td>0.121494</td>
          <td>25.058690</td>
          <td>0.234038</td>
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
          <td>27.167847</td>
          <td>0.678284</td>
          <td>26.844125</td>
          <td>0.212978</td>
          <td>26.131493</td>
          <td>0.104050</td>
          <td>25.253605</td>
          <td>0.078786</td>
          <td>25.085062</td>
          <td>0.128215</td>
          <td>24.537086</td>
          <td>0.177240</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.074621</td>
          <td>0.558334</td>
          <td>27.624921</td>
          <td>0.363006</td>
          <td>27.203715</td>
          <td>0.404699</td>
          <td>26.833453</td>
          <td>0.527856</td>
          <td>25.797033</td>
          <td>0.486524</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.813865</td>
          <td>0.535657</td>
          <td>26.175617</td>
          <td>0.122861</td>
          <td>24.805640</td>
          <td>0.032973</td>
          <td>23.876877</td>
          <td>0.023945</td>
          <td>23.106423</td>
          <td>0.023018</td>
          <td>22.733664</td>
          <td>0.037339</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.713947</td>
          <td>1.001064</td>
          <td>28.844812</td>
          <td>0.975588</td>
          <td>27.202030</td>
          <td>0.275261</td>
          <td>27.696862</td>
          <td>0.616650</td>
          <td>25.786232</td>
          <td>0.247595</td>
          <td>25.313249</td>
          <td>0.356892</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.296802</td>
          <td>0.357177</td>
          <td>25.800488</td>
          <td>0.086805</td>
          <td>25.355548</td>
          <td>0.052468</td>
          <td>24.783087</td>
          <td>0.051954</td>
          <td>24.357793</td>
          <td>0.067770</td>
          <td>23.596284</td>
          <td>0.078349</td>
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
          <td>26.368098</td>
          <td>0.382897</td>
          <td>26.135106</td>
          <td>0.118478</td>
          <td>26.119184</td>
          <td>0.105112</td>
          <td>26.319003</td>
          <td>0.202194</td>
          <td>25.948419</td>
          <td>0.270857</td>
          <td>25.531962</td>
          <td>0.405630</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.390065</td>
          <td>1.415211</td>
          <td>27.098397</td>
          <td>0.263669</td>
          <td>26.776648</td>
          <td>0.182173</td>
          <td>26.861430</td>
          <td>0.310496</td>
          <td>25.942034</td>
          <td>0.265282</td>
          <td>24.777628</td>
          <td>0.217865</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.347411</td>
          <td>0.770952</td>
          <td>26.795932</td>
          <td>0.206829</td>
          <td>26.904617</td>
          <td>0.204589</td>
          <td>26.220669</td>
          <td>0.184613</td>
          <td>25.922124</td>
          <td>0.263099</td>
          <td>25.130343</td>
          <td>0.293379</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.315607</td>
          <td>0.763081</td>
          <td>27.570402</td>
          <td>0.392444</td>
          <td>26.565056</td>
          <td>0.156218</td>
          <td>25.863102</td>
          <td>0.138594</td>
          <td>25.168426</td>
          <td>0.142104</td>
          <td>24.597494</td>
          <td>0.192407</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.308770</td>
          <td>0.750301</td>
          <td>26.444254</td>
          <td>0.153186</td>
          <td>26.275527</td>
          <td>0.119156</td>
          <td>25.510906</td>
          <td>0.099825</td>
          <td>25.127442</td>
          <td>0.134322</td>
          <td>24.734648</td>
          <td>0.211397</td>
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
          <td>27.092319</td>
          <td>0.586281</td>
          <td>26.675011</td>
          <td>0.160943</td>
          <td>25.995480</td>
          <td>0.078543</td>
          <td>25.300487</td>
          <td>0.069310</td>
          <td>25.208942</td>
          <td>0.121740</td>
          <td>24.834629</td>
          <td>0.194129</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.705374</td>
          <td>0.374906</td>
          <td>27.921333</td>
          <td>0.396471</td>
          <td>26.881080</td>
          <td>0.269412</td>
          <td>26.751396</td>
          <td>0.433446</td>
          <td>26.638069</td>
          <td>0.769020</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>29.274243</td>
          <td>2.060968</td>
          <td>25.879672</td>
          <td>0.086641</td>
          <td>24.835557</td>
          <td>0.030548</td>
          <td>23.873197</td>
          <td>0.021500</td>
          <td>23.143595</td>
          <td>0.021493</td>
          <td>22.818223</td>
          <td>0.036179</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.667040</td>
          <td>0.433480</td>
          <td>27.335984</td>
          <td>0.305696</td>
          <td>26.663707</td>
          <td>0.280086</td>
          <td>25.729035</td>
          <td>0.235397</td>
          <td>25.207536</td>
          <td>0.327239</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.298026</td>
          <td>0.321948</td>
          <td>25.747795</td>
          <td>0.071839</td>
          <td>25.451428</td>
          <td>0.048555</td>
          <td>24.786937</td>
          <td>0.044003</td>
          <td>24.358294</td>
          <td>0.057655</td>
          <td>23.656039</td>
          <td>0.069888</td>
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
          <td>25.967711</td>
          <td>0.259408</td>
          <td>26.396559</td>
          <td>0.135553</td>
          <td>26.187190</td>
          <td>0.100588</td>
          <td>25.926217</td>
          <td>0.130251</td>
          <td>25.912467</td>
          <td>0.238725</td>
          <td>25.275718</td>
          <td>0.301352</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.816328</td>
          <td>0.484261</td>
          <td>26.790245</td>
          <td>0.179969</td>
          <td>26.930853</td>
          <td>0.179775</td>
          <td>26.509672</td>
          <td>0.201174</td>
          <td>25.841545</td>
          <td>0.212221</td>
          <td>25.003164</td>
          <td>0.227147</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.816125</td>
          <td>0.493651</td>
          <td>26.960493</td>
          <td>0.213387</td>
          <td>26.784980</td>
          <td>0.163868</td>
          <td>26.414079</td>
          <td>0.191761</td>
          <td>26.209886</td>
          <td>0.295996</td>
          <td>25.804624</td>
          <td>0.443106</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.850367</td>
          <td>0.526805</td>
          <td>26.943961</td>
          <td>0.222400</td>
          <td>26.600289</td>
          <td>0.149163</td>
          <td>25.801531</td>
          <td>0.121304</td>
          <td>25.571144</td>
          <td>0.185712</td>
          <td>25.397465</td>
          <td>0.343383</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.530326</td>
          <td>0.395362</td>
          <td>26.340006</td>
          <td>0.124702</td>
          <td>25.932631</td>
          <td>0.077260</td>
          <td>25.609505</td>
          <td>0.094822</td>
          <td>25.269571</td>
          <td>0.133304</td>
          <td>25.489509</td>
          <td>0.344196</td>
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
