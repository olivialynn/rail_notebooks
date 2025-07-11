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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f8d0ce71600>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.009907  0.008664  
    1      25.391064  0.153227  0.091936  
    2      24.304707  0.098330  0.072241  
    3      25.291103  0.033368  0.017141  
    4      25.096743  0.092213  0.074591  
    ...          ...       ...       ...  
    99995  24.737946  0.019711  0.018125  
    99996  24.224169  0.049338  0.029780  
    99997  25.613836  0.063189  0.062103  
    99998  25.274899  0.115781  0.062141  
    99999  25.699642  0.040037  0.031582  
    
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
          <td>1.398944</td>
          <td>27.314836</td>
          <td>0.684586</td>
          <td>26.646814</td>
          <td>0.157094</td>
          <td>26.007355</td>
          <td>0.079360</td>
          <td>25.216480</td>
          <td>0.064330</td>
          <td>24.620589</td>
          <td>0.072636</td>
          <td>23.965644</td>
          <td>0.091703</td>
          <td>0.009907</td>
          <td>0.008664</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.473881</td>
          <td>0.312082</td>
          <td>26.402483</td>
          <td>0.112262</td>
          <td>26.364669</td>
          <td>0.175058</td>
          <td>25.900870</td>
          <td>0.219532</td>
          <td>25.423956</td>
          <td>0.315041</td>
          <td>0.153227</td>
          <td>0.091936</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.266283</td>
          <td>1.997977</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.956070</td>
          <td>2.284812</td>
          <td>26.000208</td>
          <td>0.128040</td>
          <td>24.853096</td>
          <td>0.089173</td>
          <td>24.233195</td>
          <td>0.115889</td>
          <td>0.098330</td>
          <td>0.072241</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.615791</td>
          <td>0.412238</td>
          <td>27.881289</td>
          <td>0.428932</td>
          <td>27.857393</td>
          <td>0.377003</td>
          <td>26.294266</td>
          <td>0.164877</td>
          <td>25.523068</td>
          <td>0.159576</td>
          <td>24.811019</td>
          <td>0.190279</td>
          <td>0.033368</td>
          <td>0.017141</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.438506</td>
          <td>0.359375</td>
          <td>26.046992</td>
          <td>0.093376</td>
          <td>26.008323</td>
          <td>0.079428</td>
          <td>25.636122</td>
          <td>0.093183</td>
          <td>25.644814</td>
          <td>0.177009</td>
          <td>25.195331</td>
          <td>0.261880</td>
          <td>0.092213</td>
          <td>0.074591</td>
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
          <td>0.389450</td>
          <td>26.568124</td>
          <td>0.397427</td>
          <td>26.066775</td>
          <td>0.095010</td>
          <td>25.474996</td>
          <td>0.049511</td>
          <td>25.037961</td>
          <td>0.054906</td>
          <td>24.925557</td>
          <td>0.095036</td>
          <td>24.758228</td>
          <td>0.181978</td>
          <td>0.019711</td>
          <td>0.018125</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.848760</td>
          <td>0.186508</td>
          <td>26.001447</td>
          <td>0.078947</td>
          <td>25.218714</td>
          <td>0.064457</td>
          <td>24.943182</td>
          <td>0.096517</td>
          <td>23.997260</td>
          <td>0.094285</td>
          <td>0.049338</td>
          <td>0.029780</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.614498</td>
          <td>0.411830</td>
          <td>27.075962</td>
          <td>0.225597</td>
          <td>26.439094</td>
          <td>0.115901</td>
          <td>26.212163</td>
          <td>0.153700</td>
          <td>25.993238</td>
          <td>0.237017</td>
          <td>25.507708</td>
          <td>0.336735</td>
          <td>0.063189</td>
          <td>0.062103</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.648099</td>
          <td>0.853098</td>
          <td>26.339968</td>
          <td>0.120588</td>
          <td>25.952494</td>
          <td>0.075606</td>
          <td>25.723719</td>
          <td>0.100626</td>
          <td>25.716698</td>
          <td>0.188112</td>
          <td>25.616962</td>
          <td>0.366939</td>
          <td>0.115781</td>
          <td>0.062141</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.905735</td>
          <td>0.512321</td>
          <td>26.608823</td>
          <td>0.152068</td>
          <td>27.038072</td>
          <td>0.193727</td>
          <td>26.354574</td>
          <td>0.173563</td>
          <td>26.459927</td>
          <td>0.345619</td>
          <td>25.169087</td>
          <td>0.256314</td>
          <td>0.040037</td>
          <td>0.031582</td>
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
          <td>1.398944</td>
          <td>26.608190</td>
          <td>0.453639</td>
          <td>26.406623</td>
          <td>0.147050</td>
          <td>26.078736</td>
          <td>0.099381</td>
          <td>25.109129</td>
          <td>0.069361</td>
          <td>24.777249</td>
          <td>0.098074</td>
          <td>23.815703</td>
          <td>0.095031</td>
          <td>0.009907</td>
          <td>0.008664</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.041368</td>
          <td>0.566042</td>
          <td>26.609147</td>
          <td>0.165325</td>
          <td>26.427391</td>
          <td>0.227928</td>
          <td>26.296751</td>
          <td>0.367519</td>
          <td>25.517108</td>
          <td>0.411928</td>
          <td>0.153227</td>
          <td>0.091936</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.535267</td>
          <td>1.402980</td>
          <td>28.276082</td>
          <td>0.602464</td>
          <td>26.114066</td>
          <td>0.170663</td>
          <td>24.997742</td>
          <td>0.121791</td>
          <td>24.353425</td>
          <td>0.155330</td>
          <td>0.098330</td>
          <td>0.072241</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.550347</td>
          <td>0.377548</td>
          <td>28.715181</td>
          <td>0.798137</td>
          <td>26.286958</td>
          <td>0.193222</td>
          <td>25.456707</td>
          <td>0.176748</td>
          <td>24.804026</td>
          <td>0.222313</td>
          <td>0.033368</td>
          <td>0.017141</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.908243</td>
          <td>0.266030</td>
          <td>26.480810</td>
          <td>0.159912</td>
          <td>25.971085</td>
          <td>0.092536</td>
          <td>25.639718</td>
          <td>0.113256</td>
          <td>25.434071</td>
          <td>0.176932</td>
          <td>24.956371</td>
          <td>0.257260</td>
          <td>0.092213</td>
          <td>0.074591</td>
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
          <td>0.389450</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.379347</td>
          <td>0.143762</td>
          <td>25.341360</td>
          <td>0.051857</td>
          <td>25.240506</td>
          <td>0.077976</td>
          <td>24.911635</td>
          <td>0.110404</td>
          <td>24.586329</td>
          <td>0.185003</td>
          <td>0.019711</td>
          <td>0.018125</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.679532</td>
          <td>2.478449</td>
          <td>26.676833</td>
          <td>0.185965</td>
          <td>26.074563</td>
          <td>0.099546</td>
          <td>25.119646</td>
          <td>0.070400</td>
          <td>24.804708</td>
          <td>0.100996</td>
          <td>24.291521</td>
          <td>0.144505</td>
          <td>0.049338</td>
          <td>0.029780</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.479485</td>
          <td>0.158307</td>
          <td>26.396794</td>
          <td>0.132773</td>
          <td>26.278284</td>
          <td>0.193882</td>
          <td>25.539892</td>
          <td>0.191609</td>
          <td>25.639208</td>
          <td>0.437154</td>
          <td>0.063189</td>
          <td>0.062103</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.180271</td>
          <td>0.332177</td>
          <td>26.197960</td>
          <td>0.125936</td>
          <td>26.074098</td>
          <td>0.101784</td>
          <td>25.937409</td>
          <td>0.147278</td>
          <td>25.651768</td>
          <td>0.213520</td>
          <td>24.810516</td>
          <td>0.229208</td>
          <td>0.115781</td>
          <td>0.062141</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.676922</td>
          <td>0.478836</td>
          <td>26.914296</td>
          <td>0.226630</td>
          <td>26.516330</td>
          <td>0.145920</td>
          <td>26.231063</td>
          <td>0.184690</td>
          <td>25.982557</td>
          <td>0.274238</td>
          <td>25.669800</td>
          <td>0.443929</td>
          <td>0.040037</td>
          <td>0.031582</td>
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
          <td>1.398944</td>
          <td>28.122803</td>
          <td>1.137119</td>
          <td>26.525766</td>
          <td>0.141741</td>
          <td>25.997610</td>
          <td>0.078774</td>
          <td>25.325649</td>
          <td>0.070949</td>
          <td>24.629161</td>
          <td>0.073276</td>
          <td>24.035318</td>
          <td>0.097606</td>
          <td>0.009907</td>
          <td>0.008664</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.150067</td>
          <td>0.675422</td>
          <td>32.914278</td>
          <td>4.439223</td>
          <td>26.999674</td>
          <td>0.221360</td>
          <td>26.155536</td>
          <td>0.174560</td>
          <td>26.271109</td>
          <td>0.347961</td>
          <td>25.216223</td>
          <td>0.314090</td>
          <td>0.153227</td>
          <td>0.091936</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.244418</td>
          <td>0.598802</td>
          <td>28.710439</td>
          <td>0.753486</td>
          <td>26.048065</td>
          <td>0.146532</td>
          <td>25.084228</td>
          <td>0.119519</td>
          <td>24.327788</td>
          <td>0.138070</td>
          <td>0.098330</td>
          <td>0.072241</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.431311</td>
          <td>0.644643</td>
          <td>27.212688</td>
          <td>0.226217</td>
          <td>26.144590</td>
          <td>0.146451</td>
          <td>25.675500</td>
          <td>0.183330</td>
          <td>25.486802</td>
          <td>0.334146</td>
          <td>0.033368</td>
          <td>0.017141</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.192975</td>
          <td>0.312896</td>
          <td>26.293109</td>
          <td>0.124842</td>
          <td>25.898351</td>
          <td>0.078663</td>
          <td>25.613281</td>
          <td>0.100015</td>
          <td>25.242298</td>
          <td>0.136482</td>
          <td>24.999725</td>
          <td>0.242585</td>
          <td>0.092213</td>
          <td>0.074591</td>
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
          <td>0.389450</td>
          <td>27.267640</td>
          <td>0.664531</td>
          <td>26.286416</td>
          <td>0.115585</td>
          <td>25.391410</td>
          <td>0.046196</td>
          <td>25.056560</td>
          <td>0.056109</td>
          <td>24.764496</td>
          <td>0.082882</td>
          <td>24.691939</td>
          <td>0.172867</td>
          <td>0.019711</td>
          <td>0.018125</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.104924</td>
          <td>2.755602</td>
          <td>26.897818</td>
          <td>0.197968</td>
          <td>25.999990</td>
          <td>0.080606</td>
          <td>25.287399</td>
          <td>0.070112</td>
          <td>24.886232</td>
          <td>0.093847</td>
          <td>24.255973</td>
          <td>0.120907</td>
          <td>0.049338</td>
          <td>0.029780</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.718188</td>
          <td>0.459367</td>
          <td>26.541256</td>
          <td>0.149831</td>
          <td>26.397677</td>
          <td>0.117545</td>
          <td>25.846814</td>
          <td>0.118104</td>
          <td>26.474238</td>
          <td>0.365650</td>
          <td>26.816922</td>
          <td>0.895948</td>
          <td>0.063189</td>
          <td>0.062103</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.519846</td>
          <td>0.408361</td>
          <td>26.097254</td>
          <td>0.106738</td>
          <td>26.071619</td>
          <td>0.093040</td>
          <td>25.851804</td>
          <td>0.125073</td>
          <td>25.455520</td>
          <td>0.166325</td>
          <td>24.582437</td>
          <td>0.173600</td>
          <td>0.115781</td>
          <td>0.062141</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.116426</td>
          <td>0.602169</td>
          <td>26.541435</td>
          <td>0.145654</td>
          <td>27.063554</td>
          <td>0.201235</td>
          <td>26.257999</td>
          <td>0.162703</td>
          <td>26.006486</td>
          <td>0.243552</td>
          <td>25.631882</td>
          <td>0.377206</td>
          <td>0.040037</td>
          <td>0.031582</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
