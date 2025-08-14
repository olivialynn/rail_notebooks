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

    <pzflow.flow.Flow at 0x7f293b749720>



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
    0      23.994413  0.012098  0.010499  
    1      25.391064  0.037775  0.029592  
    2      24.304707  0.150297  0.095855  
    3      25.291103  0.243542  0.194983  
    4      25.096743  0.210332  0.112329  
    ...          ...       ...       ...  
    99995  24.737946  0.088448  0.046136  
    99996  24.224169  0.188785  0.162084  
    99997  25.613836  0.119968  0.092302  
    99998  25.274899  0.122404  0.075988  
    99999  25.699642  0.037664  0.034745  
    
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
          <td>28.323118</td>
          <td>1.270592</td>
          <td>26.625998</td>
          <td>0.154322</td>
          <td>25.974643</td>
          <td>0.077101</td>
          <td>25.180097</td>
          <td>0.062287</td>
          <td>24.819517</td>
          <td>0.086577</td>
          <td>24.224699</td>
          <td>0.115035</td>
          <td>0.012098</td>
          <td>0.010499</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.720470</td>
          <td>0.379057</td>
          <td>27.022041</td>
          <td>0.191128</td>
          <td>26.377203</td>
          <td>0.176930</td>
          <td>25.807120</td>
          <td>0.202986</td>
          <td>25.877210</td>
          <td>0.448114</td>
          <td>0.037775</td>
          <td>0.029592</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.018974</td>
          <td>0.475781</td>
          <td>29.697245</td>
          <td>1.282777</td>
          <td>25.944532</td>
          <td>0.122003</td>
          <td>25.171052</td>
          <td>0.117780</td>
          <td>24.348391</td>
          <td>0.128083</td>
          <td>0.150297</td>
          <td>0.095855</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.890267</td>
          <td>0.869073</td>
          <td>26.949701</td>
          <td>0.179790</td>
          <td>26.160696</td>
          <td>0.147059</td>
          <td>25.457151</td>
          <td>0.150817</td>
          <td>25.198683</td>
          <td>0.262598</td>
          <td>0.243542</td>
          <td>0.194983</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.117867</td>
          <td>0.278332</td>
          <td>26.042505</td>
          <td>0.093009</td>
          <td>25.887696</td>
          <td>0.071396</td>
          <td>25.800778</td>
          <td>0.107643</td>
          <td>25.571828</td>
          <td>0.166357</td>
          <td>25.671695</td>
          <td>0.382910</td>
          <td>0.210332</td>
          <td>0.112329</td>
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
          <td>27.151817</td>
          <td>0.611436</td>
          <td>26.404566</td>
          <td>0.127533</td>
          <td>25.389949</td>
          <td>0.045910</td>
          <td>24.976391</td>
          <td>0.051985</td>
          <td>24.790146</td>
          <td>0.084366</td>
          <td>24.505690</td>
          <td>0.146704</td>
          <td>0.088448</td>
          <td>0.046136</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.868917</td>
          <td>0.498633</td>
          <td>26.902508</td>
          <td>0.195151</td>
          <td>26.070771</td>
          <td>0.083926</td>
          <td>25.221139</td>
          <td>0.064596</td>
          <td>24.778517</td>
          <td>0.083506</td>
          <td>24.146429</td>
          <td>0.107444</td>
          <td>0.188785</td>
          <td>0.162084</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.429533</td>
          <td>0.739698</td>
          <td>26.638370</td>
          <td>0.155964</td>
          <td>26.407779</td>
          <td>0.112781</td>
          <td>26.038447</td>
          <td>0.132349</td>
          <td>25.821976</td>
          <td>0.205530</td>
          <td>25.294259</td>
          <td>0.283832</td>
          <td>0.119968</td>
          <td>0.092302</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.556306</td>
          <td>0.393823</td>
          <td>26.370761</td>
          <td>0.123852</td>
          <td>26.009415</td>
          <td>0.079505</td>
          <td>25.867690</td>
          <td>0.114114</td>
          <td>26.017420</td>
          <td>0.241797</td>
          <td>25.313234</td>
          <td>0.288222</td>
          <td>0.122404</td>
          <td>0.075988</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.696995</td>
          <td>0.438516</td>
          <td>27.017473</td>
          <td>0.214880</td>
          <td>26.413591</td>
          <td>0.113354</td>
          <td>26.157433</td>
          <td>0.146648</td>
          <td>25.966863</td>
          <td>0.231901</td>
          <td>25.265499</td>
          <td>0.277289</td>
          <td>0.037664</td>
          <td>0.034745</td>
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
          <td>29.106032</td>
          <td>1.975704</td>
          <td>26.764857</td>
          <td>0.199374</td>
          <td>25.873756</td>
          <td>0.083011</td>
          <td>25.221171</td>
          <td>0.076594</td>
          <td>24.673415</td>
          <td>0.089543</td>
          <td>23.972970</td>
          <td>0.109068</td>
          <td>0.012098</td>
          <td>0.010499</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.855882</td>
          <td>0.545829</td>
          <td>27.248111</td>
          <td>0.297614</td>
          <td>26.675589</td>
          <td>0.167145</td>
          <td>26.142821</td>
          <td>0.171290</td>
          <td>25.534010</td>
          <td>0.188970</td>
          <td>25.247767</td>
          <td>0.319661</td>
          <td>0.037775</td>
          <td>0.029592</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.244063</td>
          <td>2.130337</td>
          <td>29.854005</td>
          <td>1.664210</td>
          <td>29.113954</td>
          <td>1.057733</td>
          <td>25.959919</td>
          <td>0.153604</td>
          <td>24.818640</td>
          <td>0.106945</td>
          <td>24.383014</td>
          <td>0.163526</td>
          <td>0.150297</td>
          <td>0.095855</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.671488</td>
          <td>1.016265</td>
          <td>30.373550</td>
          <td>2.172082</td>
          <td>27.509950</td>
          <td>0.376232</td>
          <td>26.499677</td>
          <td>0.263989</td>
          <td>25.695974</td>
          <td>0.246462</td>
          <td>26.329447</td>
          <td>0.792063</td>
          <td>0.243542</td>
          <td>0.194983</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.417698</td>
          <td>0.415979</td>
          <td>26.219196</td>
          <td>0.135120</td>
          <td>25.961348</td>
          <td>0.097691</td>
          <td>25.598459</td>
          <td>0.116498</td>
          <td>25.298808</td>
          <td>0.167584</td>
          <td>25.141511</td>
          <td>0.317020</td>
          <td>0.210332</td>
          <td>0.112329</td>
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
          <td>26.531469</td>
          <td>0.432703</td>
          <td>26.304636</td>
          <td>0.136655</td>
          <td>25.395645</td>
          <td>0.055269</td>
          <td>25.013416</td>
          <td>0.064819</td>
          <td>24.702497</td>
          <td>0.093356</td>
          <td>24.668675</td>
          <td>0.201312</td>
          <td>0.088448</td>
          <td>0.046136</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.454674</td>
          <td>0.866254</td>
          <td>26.955671</td>
          <td>0.253412</td>
          <td>26.126473</td>
          <td>0.113874</td>
          <td>25.316234</td>
          <td>0.091875</td>
          <td>24.791837</td>
          <td>0.109175</td>
          <td>24.226642</td>
          <td>0.149512</td>
          <td>0.188785</td>
          <td>0.162084</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.900130</td>
          <td>0.575447</td>
          <td>26.645567</td>
          <td>0.186208</td>
          <td>26.503522</td>
          <td>0.149063</td>
          <td>26.614242</td>
          <td>0.262390</td>
          <td>25.595597</td>
          <td>0.205500</td>
          <td>26.395409</td>
          <td>0.763821</td>
          <td>0.119968</td>
          <td>0.092302</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.301424</td>
          <td>0.366772</td>
          <td>26.044802</td>
          <td>0.110806</td>
          <td>26.133448</td>
          <td>0.107804</td>
          <td>25.815070</td>
          <td>0.133297</td>
          <td>25.955189</td>
          <td>0.275616</td>
          <td>24.915808</td>
          <td>0.251365</td>
          <td>0.122404</td>
          <td>0.075988</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.615845</td>
          <td>0.457490</td>
          <td>26.694228</td>
          <td>0.188524</td>
          <td>26.744953</td>
          <td>0.177390</td>
          <td>27.040626</td>
          <td>0.357948</td>
          <td>26.137939</td>
          <td>0.310874</td>
          <td>26.968142</td>
          <td>1.066087</td>
          <td>0.037664</td>
          <td>0.034745</td>
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
          <td>27.403449</td>
          <td>0.727568</td>
          <td>26.645985</td>
          <td>0.157214</td>
          <td>25.970581</td>
          <td>0.076959</td>
          <td>25.151769</td>
          <td>0.060854</td>
          <td>24.635010</td>
          <td>0.073698</td>
          <td>23.900725</td>
          <td>0.086770</td>
          <td>0.012098</td>
          <td>0.010499</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>32.611117</td>
          <td>5.176383</td>
          <td>27.275411</td>
          <td>0.269166</td>
          <td>26.942732</td>
          <td>0.181403</td>
          <td>26.631485</td>
          <td>0.222491</td>
          <td>26.214809</td>
          <td>0.288178</td>
          <td>25.707683</td>
          <td>0.399283</td>
          <td>0.037775</td>
          <td>0.029592</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.771256</td>
          <td>0.900675</td>
          <td>27.507758</td>
          <td>0.334689</td>
          <td>25.952923</td>
          <td>0.146871</td>
          <td>24.941202</td>
          <td>0.114565</td>
          <td>24.404048</td>
          <td>0.160228</td>
          <td>0.150297</td>
          <td>0.095855</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.811235</td>
          <td>0.557689</td>
          <td>27.567607</td>
          <td>0.432277</td>
          <td>26.906876</td>
          <td>0.403046</td>
          <td>25.841702</td>
          <td>0.306593</td>
          <td>24.822430</td>
          <td>0.285749</td>
          <td>0.243542</td>
          <td>0.194983</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.741871</td>
          <td>0.537868</td>
          <td>26.242078</td>
          <td>0.140486</td>
          <td>26.134158</td>
          <td>0.115922</td>
          <td>25.810891</td>
          <td>0.142865</td>
          <td>25.457654</td>
          <td>0.195484</td>
          <td>24.589100</td>
          <td>0.205535</td>
          <td>0.210332</td>
          <td>0.112329</td>
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
          <td>26.646046</td>
          <td>0.438236</td>
          <td>26.541007</td>
          <td>0.151302</td>
          <td>25.565277</td>
          <td>0.057123</td>
          <td>25.035698</td>
          <td>0.058520</td>
          <td>24.741099</td>
          <td>0.085986</td>
          <td>24.983918</td>
          <td>0.233728</td>
          <td>0.088448</td>
          <td>0.046136</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.492922</td>
          <td>0.456662</td>
          <td>27.255165</td>
          <td>0.334836</td>
          <td>25.993868</td>
          <td>0.105960</td>
          <td>25.235407</td>
          <td>0.089529</td>
          <td>24.729445</td>
          <td>0.108000</td>
          <td>24.090494</td>
          <td>0.138962</td>
          <td>0.188785</td>
          <td>0.162084</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.208934</td>
          <td>1.267616</td>
          <td>26.798493</td>
          <td>0.200210</td>
          <td>26.207649</td>
          <td>0.108186</td>
          <td>26.595948</td>
          <td>0.242702</td>
          <td>26.396217</td>
          <td>0.370418</td>
          <td>26.173257</td>
          <td>0.623404</td>
          <td>0.119968</td>
          <td>0.092302</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.330279</td>
          <td>0.356951</td>
          <td>26.317035</td>
          <td>0.131374</td>
          <td>26.398094</td>
          <td>0.126088</td>
          <td>25.944544</td>
          <td>0.138194</td>
          <td>26.013683</td>
          <td>0.269865</td>
          <td>24.893690</td>
          <td>0.229731</td>
          <td>0.122404</td>
          <td>0.075988</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.167777</td>
          <td>0.293053</td>
          <td>26.744261</td>
          <td>0.173237</td>
          <td>26.436335</td>
          <td>0.117648</td>
          <td>26.687696</td>
          <td>0.233608</td>
          <td>25.888536</td>
          <td>0.220922</td>
          <td>25.971230</td>
          <td>0.488261</td>
          <td>0.037664</td>
          <td>0.034745</td>
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
