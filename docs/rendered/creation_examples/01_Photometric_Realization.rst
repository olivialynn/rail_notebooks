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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f6fee3964d0>



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
    0      23.994413  0.029137  0.024218  
    1      25.391064  0.123153  0.094852  
    2      24.304707  0.128776  0.089963  
    3      25.291103  0.165923  0.103546  
    4      25.096743  0.176975  0.124997  
    ...          ...       ...       ...  
    99995  24.737946  0.022252  0.013002  
    99996  24.224169  0.006454  0.004858  
    99997  25.613836  0.171669  0.165947  
    99998  25.274899  0.015863  0.014641  
    99999  25.699642  0.013253  0.010882  
    
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
          <td>27.041650</td>
          <td>0.565405</td>
          <td>26.689883</td>
          <td>0.162980</td>
          <td>25.949239</td>
          <td>0.075389</td>
          <td>25.129488</td>
          <td>0.059553</td>
          <td>24.673867</td>
          <td>0.076139</td>
          <td>24.189712</td>
          <td>0.111581</td>
          <td>0.029137</td>
          <td>0.024218</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.343279</td>
          <td>0.697971</td>
          <td>27.810272</td>
          <td>0.406278</td>
          <td>26.635607</td>
          <td>0.137427</td>
          <td>26.411394</td>
          <td>0.182132</td>
          <td>25.708377</td>
          <td>0.186795</td>
          <td>25.062680</td>
          <td>0.234811</td>
          <td>0.123153</td>
          <td>0.094852</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.101741</td>
          <td>0.990591</td>
          <td>28.281097</td>
          <td>0.519222</td>
          <td>26.064255</td>
          <td>0.135334</td>
          <td>25.048762</td>
          <td>0.105866</td>
          <td>24.191471</td>
          <td>0.111752</td>
          <td>0.128776</td>
          <td>0.089963</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.908529</td>
          <td>1.002425</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.081993</td>
          <td>0.201016</td>
          <td>26.360458</td>
          <td>0.174433</td>
          <td>25.456364</td>
          <td>0.150716</td>
          <td>25.413425</td>
          <td>0.312401</td>
          <td>0.165923</td>
          <td>0.103546</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.808736</td>
          <td>0.215888</td>
          <td>26.146784</td>
          <td>0.101904</td>
          <td>25.888754</td>
          <td>0.071463</td>
          <td>25.770587</td>
          <td>0.104840</td>
          <td>25.324251</td>
          <td>0.134511</td>
          <td>25.027358</td>
          <td>0.228039</td>
          <td>0.176975</td>
          <td>0.124997</td>
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
          <td>28.288848</td>
          <td>1.247080</td>
          <td>26.534177</td>
          <td>0.142628</td>
          <td>25.422613</td>
          <td>0.047261</td>
          <td>25.151300</td>
          <td>0.060717</td>
          <td>24.828445</td>
          <td>0.087260</td>
          <td>24.665094</td>
          <td>0.168141</td>
          <td>0.022252</td>
          <td>0.013002</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.717393</td>
          <td>0.891335</td>
          <td>26.624741</td>
          <td>0.154156</td>
          <td>25.972707</td>
          <td>0.076969</td>
          <td>25.338556</td>
          <td>0.071675</td>
          <td>24.731892</td>
          <td>0.080142</td>
          <td>24.142111</td>
          <td>0.107039</td>
          <td>0.006454</td>
          <td>0.004858</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.646131</td>
          <td>0.421901</td>
          <td>26.837077</td>
          <td>0.184676</td>
          <td>26.465047</td>
          <td>0.118548</td>
          <td>26.257175</td>
          <td>0.159737</td>
          <td>26.503045</td>
          <td>0.357537</td>
          <td>25.563995</td>
          <td>0.352021</td>
          <td>0.171669</td>
          <td>0.165947</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.263012</td>
          <td>0.312809</td>
          <td>26.212355</td>
          <td>0.107911</td>
          <td>26.069072</td>
          <td>0.083800</td>
          <td>25.679853</td>
          <td>0.096830</td>
          <td>25.789963</td>
          <td>0.200084</td>
          <td>25.126890</td>
          <td>0.247585</td>
          <td>0.015863</td>
          <td>0.014641</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.964753</td>
          <td>0.534874</td>
          <td>27.068629</td>
          <td>0.224228</td>
          <td>26.527198</td>
          <td>0.125123</td>
          <td>26.390474</td>
          <td>0.178932</td>
          <td>26.439238</td>
          <td>0.340021</td>
          <td>25.795173</td>
          <td>0.421074</td>
          <td>0.013253</td>
          <td>0.010882</td>
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
          <td>29.570098</td>
          <td>2.378262</td>
          <td>26.855868</td>
          <td>0.215518</td>
          <td>25.966248</td>
          <td>0.090232</td>
          <td>25.085750</td>
          <td>0.068093</td>
          <td>24.636094</td>
          <td>0.086825</td>
          <td>24.145874</td>
          <td>0.127023</td>
          <td>0.029137</td>
          <td>0.024218</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.720121</td>
          <td>0.990029</td>
          <td>27.622917</td>
          <td>0.411214</td>
          <td>26.799803</td>
          <td>0.192165</td>
          <td>26.354923</td>
          <td>0.212171</td>
          <td>25.822745</td>
          <td>0.248604</td>
          <td>25.772188</td>
          <td>0.494114</td>
          <td>0.123153</td>
          <td>0.094852</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.620827</td>
          <td>0.374890</td>
          <td>26.336584</td>
          <td>0.209089</td>
          <td>24.995162</td>
          <td>0.123389</td>
          <td>24.474724</td>
          <td>0.174924</td>
          <td>0.128776</td>
          <td>0.089963</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.087068</td>
          <td>0.249012</td>
          <td>25.784292</td>
          <td>0.133344</td>
          <td>25.253859</td>
          <td>0.157306</td>
          <td>26.080537</td>
          <td>0.627873</td>
          <td>0.165923</td>
          <td>0.103546</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.030929</td>
          <td>0.645325</td>
          <td>26.068602</td>
          <td>0.117239</td>
          <td>26.034748</td>
          <td>0.102864</td>
          <td>25.695238</td>
          <td>0.125091</td>
          <td>25.346950</td>
          <td>0.172457</td>
          <td>24.847621</td>
          <td>0.246840</td>
          <td>0.176975</td>
          <td>0.124997</td>
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
          <td>26.567005</td>
          <td>0.440010</td>
          <td>26.238145</td>
          <td>0.127267</td>
          <td>25.489649</td>
          <td>0.059142</td>
          <td>25.169765</td>
          <td>0.073246</td>
          <td>25.092259</td>
          <td>0.129156</td>
          <td>24.640588</td>
          <td>0.193652</td>
          <td>0.022252</td>
          <td>0.013002</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.501371</td>
          <td>0.159451</td>
          <td>25.927421</td>
          <td>0.087000</td>
          <td>25.038050</td>
          <td>0.065118</td>
          <td>24.716526</td>
          <td>0.092969</td>
          <td>24.214716</td>
          <td>0.134500</td>
          <td>0.006454</td>
          <td>0.004858</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.263626</td>
          <td>0.369844</td>
          <td>26.985773</td>
          <td>0.258246</td>
          <td>27.199427</td>
          <td>0.280131</td>
          <td>26.736922</td>
          <td>0.304192</td>
          <td>26.346533</td>
          <td>0.394834</td>
          <td>28.134072</td>
          <td>2.006365</td>
          <td>0.171669</td>
          <td>0.165947</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.345336</td>
          <td>0.371084</td>
          <td>26.013030</td>
          <td>0.104612</td>
          <td>26.041427</td>
          <td>0.096231</td>
          <td>25.746526</td>
          <td>0.121440</td>
          <td>25.620736</td>
          <td>0.202669</td>
          <td>25.444059</td>
          <td>0.372090</td>
          <td>0.015863</td>
          <td>0.014641</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.813121</td>
          <td>0.528020</td>
          <td>26.743672</td>
          <td>0.195868</td>
          <td>26.913708</td>
          <td>0.203746</td>
          <td>26.046015</td>
          <td>0.157175</td>
          <td>26.068750</td>
          <td>0.293006</td>
          <td>24.848299</td>
          <td>0.230214</td>
          <td>0.013253</td>
          <td>0.010882</td>
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
          <td>29.678470</td>
          <td>2.361003</td>
          <td>27.211344</td>
          <td>0.254253</td>
          <td>26.192909</td>
          <td>0.094355</td>
          <td>25.133144</td>
          <td>0.060362</td>
          <td>24.824608</td>
          <td>0.087811</td>
          <td>24.054296</td>
          <td>0.100120</td>
          <td>0.029137</td>
          <td>0.024218</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.276765</td>
          <td>0.721360</td>
          <td>27.905261</td>
          <td>0.485451</td>
          <td>26.547490</td>
          <td>0.146147</td>
          <td>26.768207</td>
          <td>0.281114</td>
          <td>26.040969</td>
          <td>0.280798</td>
          <td>24.990240</td>
          <td>0.253391</td>
          <td>0.123153</td>
          <td>0.094852</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>33.469331</td>
          <td>6.133836</td>
          <td>28.552244</td>
          <td>0.765795</td>
          <td>28.425223</td>
          <td>0.645590</td>
          <td>26.023672</td>
          <td>0.151073</td>
          <td>24.900676</td>
          <td>0.107106</td>
          <td>24.476879</td>
          <td>0.165114</td>
          <td>0.128776</td>
          <td>0.089963</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.187786</td>
          <td>0.623262</td>
          <td>27.992456</td>
          <td>0.497119</td>
          <td>26.320187</td>
          <td>0.206232</td>
          <td>25.470440</td>
          <td>0.185460</td>
          <td>25.435927</td>
          <td>0.383188</td>
          <td>0.165923</td>
          <td>0.103546</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.352110</td>
          <td>0.803695</td>
          <td>26.373073</td>
          <td>0.153204</td>
          <td>26.123834</td>
          <td>0.111754</td>
          <td>25.913570</td>
          <td>0.151773</td>
          <td>25.634898</td>
          <td>0.220808</td>
          <td>25.428315</td>
          <td>0.394331</td>
          <td>0.176975</td>
          <td>0.124997</td>
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
          <td>27.607074</td>
          <td>0.832869</td>
          <td>26.403778</td>
          <td>0.127937</td>
          <td>25.427334</td>
          <td>0.047676</td>
          <td>25.165527</td>
          <td>0.061782</td>
          <td>24.873153</td>
          <td>0.091169</td>
          <td>24.582087</td>
          <td>0.157352</td>
          <td>0.022252</td>
          <td>0.013002</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.079405</td>
          <td>0.581018</td>
          <td>26.449911</td>
          <td>0.132684</td>
          <td>25.929560</td>
          <td>0.074122</td>
          <td>25.153812</td>
          <td>0.060881</td>
          <td>24.758150</td>
          <td>0.082056</td>
          <td>24.206800</td>
          <td>0.113307</td>
          <td>0.006454</td>
          <td>0.004858</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.996972</td>
          <td>0.649809</td>
          <td>26.633807</td>
          <td>0.198599</td>
          <td>26.306395</td>
          <td>0.136693</td>
          <td>26.083927</td>
          <td>0.183298</td>
          <td>25.923335</td>
          <td>0.291333</td>
          <td>25.789845</td>
          <td>0.537166</td>
          <td>0.171669</td>
          <td>0.165947</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.856455</td>
          <td>0.225085</td>
          <td>26.048225</td>
          <td>0.093733</td>
          <td>25.944170</td>
          <td>0.075291</td>
          <td>25.888185</td>
          <td>0.116552</td>
          <td>25.477184</td>
          <td>0.153903</td>
          <td>25.164752</td>
          <td>0.256188</td>
          <td>0.015863</td>
          <td>0.014641</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.045275</td>
          <td>0.567515</td>
          <td>26.758467</td>
          <td>0.173067</td>
          <td>26.388154</td>
          <td>0.111087</td>
          <td>26.379838</td>
          <td>0.177685</td>
          <td>25.586674</td>
          <td>0.168800</td>
          <td>25.377605</td>
          <td>0.304137</td>
          <td>0.013253</td>
          <td>0.010882</td>
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
