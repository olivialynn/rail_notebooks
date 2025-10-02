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

    <pzflow.flow.Flow at 0x7fb1b9614400>



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
    0      23.994413  0.007842  0.006512  
    1      25.391064  0.015428  0.007791  
    2      24.304707  0.021623  0.020681  
    3      25.291103  0.183820  0.169775  
    4      25.096743  0.255062  0.231313  
    ...          ...       ...       ...  
    99995  24.737946  0.011704  0.008798  
    99996  24.224169  0.024277  0.013165  
    99997  25.613836  0.226320  0.132375  
    99998  25.274899  0.026427  0.014111  
    99999  25.699642  0.116545  0.077948  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>27.150166</td>
          <td>0.610726</td>
          <td>26.850393</td>
          <td>0.186765</td>
          <td>26.098604</td>
          <td>0.086009</td>
          <td>25.052745</td>
          <td>0.055632</td>
          <td>24.582554</td>
          <td>0.070233</td>
          <td>24.057744</td>
          <td>0.099422</td>
          <td>0.007842</td>
          <td>0.006512</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.396586</td>
          <td>0.723557</td>
          <td>28.129702</td>
          <td>0.516348</td>
          <td>26.617092</td>
          <td>0.135248</td>
          <td>26.405954</td>
          <td>0.181295</td>
          <td>25.751101</td>
          <td>0.193649</td>
          <td>25.165909</td>
          <td>0.255647</td>
          <td>0.015428</td>
          <td>0.007791</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.516109</td>
          <td>1.909064</td>
          <td>25.875462</td>
          <td>0.114889</td>
          <td>25.168624</td>
          <td>0.117532</td>
          <td>24.116745</td>
          <td>0.104692</td>
          <td>0.021623</td>
          <td>0.020681</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.429688</td>
          <td>0.268073</td>
          <td>26.424265</td>
          <td>0.184126</td>
          <td>25.752082</td>
          <td>0.193809</td>
          <td>25.444778</td>
          <td>0.320319</td>
          <td>0.183820</td>
          <td>0.169775</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.065961</td>
          <td>0.266838</td>
          <td>26.081683</td>
          <td>0.096259</td>
          <td>25.836185</td>
          <td>0.068214</td>
          <td>25.539527</td>
          <td>0.085592</td>
          <td>25.502687</td>
          <td>0.156818</td>
          <td>25.136327</td>
          <td>0.249514</td>
          <td>0.255062</td>
          <td>0.231313</td>
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
          <td>26.802655</td>
          <td>0.474728</td>
          <td>26.271772</td>
          <td>0.113646</td>
          <td>25.359112</td>
          <td>0.044670</td>
          <td>25.112336</td>
          <td>0.058653</td>
          <td>25.078540</td>
          <td>0.108656</td>
          <td>24.664725</td>
          <td>0.168088</td>
          <td>0.011704</td>
          <td>0.008798</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.946596</td>
          <td>1.025521</td>
          <td>26.617193</td>
          <td>0.153162</td>
          <td>26.081716</td>
          <td>0.084739</td>
          <td>25.114115</td>
          <td>0.058746</td>
          <td>24.765047</td>
          <td>0.082520</td>
          <td>24.094017</td>
          <td>0.102631</td>
          <td>0.024277</td>
          <td>0.013165</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.163295</td>
          <td>0.616388</td>
          <td>26.696552</td>
          <td>0.163910</td>
          <td>26.586694</td>
          <td>0.131740</td>
          <td>26.431003</td>
          <td>0.185178</td>
          <td>25.975550</td>
          <td>0.233575</td>
          <td>25.434456</td>
          <td>0.317693</td>
          <td>0.226320</td>
          <td>0.132375</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.118000</td>
          <td>0.278362</td>
          <td>26.289426</td>
          <td>0.115406</td>
          <td>26.112095</td>
          <td>0.087037</td>
          <td>25.971853</td>
          <td>0.124931</td>
          <td>25.533553</td>
          <td>0.161012</td>
          <td>25.399317</td>
          <td>0.308893</td>
          <td>0.026427</td>
          <td>0.014111</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.974275</td>
          <td>0.247560</td>
          <td>27.034896</td>
          <td>0.218023</td>
          <td>26.544738</td>
          <td>0.127041</td>
          <td>26.382157</td>
          <td>0.177675</td>
          <td>26.281132</td>
          <td>0.299751</td>
          <td>25.603546</td>
          <td>0.363111</td>
          <td>0.116545</td>
          <td>0.077948</td>
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
          <td>29.087661</td>
          <td>1.960147</td>
          <td>26.429502</td>
          <td>0.149950</td>
          <td>26.006100</td>
          <td>0.093236</td>
          <td>25.143425</td>
          <td>0.071489</td>
          <td>24.705982</td>
          <td>0.092118</td>
          <td>24.038480</td>
          <td>0.115446</td>
          <td>0.007842</td>
          <td>0.006512</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.238720</td>
          <td>2.087996</td>
          <td>27.613258</td>
          <td>0.395800</td>
          <td>26.645220</td>
          <td>0.162335</td>
          <td>26.217836</td>
          <td>0.181926</td>
          <td>26.255619</td>
          <td>0.340154</td>
          <td>24.846801</td>
          <td>0.229930</td>
          <td>0.015428</td>
          <td>0.007791</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.141238</td>
          <td>0.666603</td>
          <td>30.765328</td>
          <td>2.388074</td>
          <td>29.670254</td>
          <td>1.394951</td>
          <td>26.016515</td>
          <td>0.153413</td>
          <td>24.923344</td>
          <td>0.111570</td>
          <td>24.024613</td>
          <td>0.114218</td>
          <td>0.021623</td>
          <td>0.020681</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.622903</td>
          <td>0.962449</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.241327</td>
          <td>0.291883</td>
          <td>26.283737</td>
          <td>0.211377</td>
          <td>25.380841</td>
          <td>0.181442</td>
          <td>24.806235</td>
          <td>0.243857</td>
          <td>0.183820</td>
          <td>0.169775</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.488072</td>
          <td>0.464272</td>
          <td>26.166157</td>
          <td>0.139046</td>
          <td>26.018003</td>
          <td>0.111410</td>
          <td>26.114554</td>
          <td>0.196843</td>
          <td>25.758521</td>
          <td>0.265956</td>
          <td>24.860490</td>
          <td>0.273016</td>
          <td>0.255062</td>
          <td>0.231313</td>
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
          <td>26.504373</td>
          <td>0.159896</td>
          <td>25.393209</td>
          <td>0.054251</td>
          <td>24.995805</td>
          <td>0.062742</td>
          <td>24.799505</td>
          <td>0.100012</td>
          <td>24.750391</td>
          <td>0.212174</td>
          <td>0.011704</td>
          <td>0.008798</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.729433</td>
          <td>0.496841</td>
          <td>26.888795</td>
          <td>0.221292</td>
          <td>26.057637</td>
          <td>0.097658</td>
          <td>25.180548</td>
          <td>0.073960</td>
          <td>25.121124</td>
          <td>0.132443</td>
          <td>24.233351</td>
          <td>0.136845</td>
          <td>0.024277</td>
          <td>0.013165</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.202965</td>
          <td>2.135044</td>
          <td>26.766902</td>
          <td>0.218229</td>
          <td>26.360931</td>
          <td>0.140542</td>
          <td>26.343085</td>
          <td>0.223446</td>
          <td>25.413399</td>
          <td>0.187634</td>
          <td>25.261709</td>
          <td>0.353988</td>
          <td>0.226320</td>
          <td>0.132375</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.421150</td>
          <td>0.393732</td>
          <td>26.511396</td>
          <td>0.161021</td>
          <td>26.277024</td>
          <td>0.118306</td>
          <td>25.706095</td>
          <td>0.117334</td>
          <td>25.781145</td>
          <td>0.231826</td>
          <td>25.624437</td>
          <td>0.427825</td>
          <td>0.026427</td>
          <td>0.014111</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.393668</td>
          <td>0.393522</td>
          <td>26.430224</td>
          <td>0.154350</td>
          <td>26.532051</td>
          <td>0.151980</td>
          <td>26.389627</td>
          <td>0.216873</td>
          <td>26.387231</td>
          <td>0.387821</td>
          <td>25.100034</td>
          <td>0.291579</td>
          <td>0.116545</td>
          <td>0.077948</td>
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
          <td>27.506281</td>
          <td>0.778551</td>
          <td>27.080662</td>
          <td>0.226610</td>
          <td>26.264747</td>
          <td>0.099597</td>
          <td>25.172459</td>
          <td>0.061913</td>
          <td>24.719527</td>
          <td>0.079328</td>
          <td>23.926084</td>
          <td>0.088633</td>
          <td>0.007842</td>
          <td>0.006512</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.632703</td>
          <td>0.354499</td>
          <td>26.707256</td>
          <td>0.146465</td>
          <td>26.369614</td>
          <td>0.176155</td>
          <td>25.801167</td>
          <td>0.202365</td>
          <td>25.319513</td>
          <td>0.290249</td>
          <td>0.015428</td>
          <td>0.007791</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.266106</td>
          <td>1.007160</td>
          <td>26.082745</td>
          <td>0.138372</td>
          <td>25.217305</td>
          <td>0.123345</td>
          <td>24.267357</td>
          <td>0.120130</td>
          <td>0.021623</td>
          <td>0.020681</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.927322</td>
          <td>0.516157</td>
          <td>26.633273</td>
          <td>0.294444</td>
          <td>24.998443</td>
          <td>0.136860</td>
          <td>25.951612</td>
          <td>0.613270</td>
          <td>0.183820</td>
          <td>0.169775</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.987405</td>
          <td>0.350728</td>
          <td>26.050496</td>
          <td>0.143361</td>
          <td>26.004418</td>
          <td>0.126649</td>
          <td>25.264625</td>
          <td>0.109321</td>
          <td>25.701520</td>
          <td>0.289340</td>
          <td>26.882205</td>
          <td>1.241362</td>
          <td>0.255062</td>
          <td>0.231313</td>
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
          <td>28.234612</td>
          <td>1.211116</td>
          <td>26.350090</td>
          <td>0.121804</td>
          <td>25.431544</td>
          <td>0.047707</td>
          <td>25.116569</td>
          <td>0.058965</td>
          <td>24.781887</td>
          <td>0.083876</td>
          <td>24.401942</td>
          <td>0.134356</td>
          <td>0.011704</td>
          <td>0.008798</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.483792</td>
          <td>0.373485</td>
          <td>26.432447</td>
          <td>0.131223</td>
          <td>26.004669</td>
          <td>0.079583</td>
          <td>25.210550</td>
          <td>0.064343</td>
          <td>24.735373</td>
          <td>0.080805</td>
          <td>24.166181</td>
          <td>0.109896</td>
          <td>0.024277</td>
          <td>0.013165</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.579237</td>
          <td>0.491483</td>
          <td>26.839021</td>
          <td>0.241591</td>
          <td>26.929456</td>
          <td>0.237811</td>
          <td>26.436559</td>
          <td>0.252589</td>
          <td>25.503472</td>
          <td>0.211740</td>
          <td>26.392863</td>
          <td>0.830280</td>
          <td>0.226320</td>
          <td>0.132375</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.164532</td>
          <td>0.618994</td>
          <td>26.225311</td>
          <td>0.109709</td>
          <td>26.024214</td>
          <td>0.081041</td>
          <td>25.815931</td>
          <td>0.109768</td>
          <td>25.603674</td>
          <td>0.171932</td>
          <td>25.134756</td>
          <td>0.250661</td>
          <td>0.026427</td>
          <td>0.014111</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.165220</td>
          <td>0.312164</td>
          <td>26.689528</td>
          <td>0.179899</td>
          <td>26.490744</td>
          <td>0.135938</td>
          <td>26.154063</td>
          <td>0.164570</td>
          <td>25.710435</td>
          <td>0.209087</td>
          <td>26.784396</td>
          <td>0.921132</td>
          <td>0.116545</td>
          <td>0.077948</td>
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
