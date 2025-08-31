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

    <pzflow.flow.Flow at 0x7f142b413400>



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
    0      23.994413  0.080380  0.047418  
    1      25.391064  0.040014  0.022696  
    2      24.304707  0.051929  0.029082  
    3      25.291103  0.119001  0.078162  
    4      25.096743  0.127233  0.093150  
    ...          ...       ...       ...  
    99995  24.737946  0.048913  0.048127  
    99996  24.224169  0.007378  0.006294  
    99997  25.613836  0.098203  0.060606  
    99998  25.274899  0.009975  0.008067  
    99999  25.699642  0.095365  0.067784  
    
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
          <td>27.115164</td>
          <td>0.595819</td>
          <td>26.685347</td>
          <td>0.162351</td>
          <td>26.199563</td>
          <td>0.093995</td>
          <td>25.231716</td>
          <td>0.065204</td>
          <td>24.707064</td>
          <td>0.078405</td>
          <td>24.035406</td>
          <td>0.097494</td>
          <td>0.080380</td>
          <td>0.047418</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.806975</td>
          <td>0.476259</td>
          <td>27.096564</td>
          <td>0.229486</td>
          <td>26.675196</td>
          <td>0.142198</td>
          <td>26.142862</td>
          <td>0.144822</td>
          <td>26.114900</td>
          <td>0.261948</td>
          <td>25.028479</td>
          <td>0.228252</td>
          <td>0.040014</td>
          <td>0.022696</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.926565</td>
          <td>1.013329</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.396000</td>
          <td>1.083169</td>
          <td>25.954325</td>
          <td>0.123045</td>
          <td>25.061461</td>
          <td>0.107048</td>
          <td>24.539869</td>
          <td>0.151073</td>
          <td>0.051929</td>
          <td>0.029082</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.549111</td>
          <td>0.694539</td>
          <td>27.376574</td>
          <td>0.256683</td>
          <td>26.195802</td>
          <td>0.151559</td>
          <td>25.729848</td>
          <td>0.190211</td>
          <td>24.869893</td>
          <td>0.199946</td>
          <td>0.119001</td>
          <td>0.078162</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.556204</td>
          <td>0.393792</td>
          <td>26.100047</td>
          <td>0.097821</td>
          <td>25.963538</td>
          <td>0.076348</td>
          <td>25.763468</td>
          <td>0.104189</td>
          <td>25.528370</td>
          <td>0.160301</td>
          <td>25.347084</td>
          <td>0.296202</td>
          <td>0.127233</td>
          <td>0.093150</td>
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
          <td>26.371787</td>
          <td>0.123963</td>
          <td>25.423066</td>
          <td>0.047280</td>
          <td>25.087690</td>
          <td>0.057384</td>
          <td>24.952494</td>
          <td>0.097309</td>
          <td>24.574373</td>
          <td>0.155607</td>
          <td>0.048913</td>
          <td>0.048127</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.333518</td>
          <td>0.693357</td>
          <td>26.715530</td>
          <td>0.166582</td>
          <td>25.943436</td>
          <td>0.075004</td>
          <td>25.333128</td>
          <td>0.071332</td>
          <td>24.772993</td>
          <td>0.083100</td>
          <td>24.140099</td>
          <td>0.106851</td>
          <td>0.007378</td>
          <td>0.006294</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.052984</td>
          <td>0.570015</td>
          <td>27.033055</td>
          <td>0.217689</td>
          <td>26.694766</td>
          <td>0.144613</td>
          <td>26.224423</td>
          <td>0.155323</td>
          <td>25.921621</td>
          <td>0.223355</td>
          <td>25.526474</td>
          <td>0.341767</td>
          <td>0.098203</td>
          <td>0.060606</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.183607</td>
          <td>0.293512</td>
          <td>26.253411</td>
          <td>0.111844</td>
          <td>26.005788</td>
          <td>0.079250</td>
          <td>25.984826</td>
          <td>0.126344</td>
          <td>25.782060</td>
          <td>0.198759</td>
          <td>25.024201</td>
          <td>0.227443</td>
          <td>0.009975</td>
          <td>0.008067</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.871602</td>
          <td>0.499621</td>
          <td>26.713817</td>
          <td>0.166339</td>
          <td>26.460806</td>
          <td>0.118111</td>
          <td>26.259386</td>
          <td>0.160039</td>
          <td>25.656404</td>
          <td>0.178757</td>
          <td>26.160732</td>
          <td>0.552445</td>
          <td>0.095365</td>
          <td>0.067784</td>
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
          <td>27.188360</td>
          <td>0.693783</td>
          <td>26.691201</td>
          <td>0.189700</td>
          <td>26.150096</td>
          <td>0.107298</td>
          <td>25.104110</td>
          <td>0.070091</td>
          <td>24.728726</td>
          <td>0.095338</td>
          <td>24.053720</td>
          <td>0.118714</td>
          <td>0.080380</td>
          <td>0.047418</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.588728</td>
          <td>0.448003</td>
          <td>27.467445</td>
          <td>0.354214</td>
          <td>26.597217</td>
          <td>0.156278</td>
          <td>26.565010</td>
          <td>0.243897</td>
          <td>25.834082</td>
          <td>0.242661</td>
          <td>25.907963</td>
          <td>0.529398</td>
          <td>0.040014</td>
          <td>0.022696</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.383280</td>
          <td>0.695855</td>
          <td>28.086795</td>
          <td>0.517772</td>
          <td>26.170270</td>
          <td>0.175697</td>
          <td>24.867891</td>
          <td>0.106773</td>
          <td>24.343410</td>
          <td>0.151147</td>
          <td>0.051929</td>
          <td>0.029082</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.994009</td>
          <td>0.613403</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.517234</td>
          <td>0.343526</td>
          <td>26.148340</td>
          <td>0.177205</td>
          <td>25.435179</td>
          <td>0.178783</td>
          <td>25.187336</td>
          <td>0.313037</td>
          <td>0.119001</td>
          <td>0.078162</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.815861</td>
          <td>0.249734</td>
          <td>26.177079</td>
          <td>0.125031</td>
          <td>25.749724</td>
          <td>0.077465</td>
          <td>25.729296</td>
          <td>0.124586</td>
          <td>25.796833</td>
          <td>0.243598</td>
          <td>25.056899</td>
          <td>0.283782</td>
          <td>0.127233</td>
          <td>0.093150</td>
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
          <td>27.148635</td>
          <td>0.672567</td>
          <td>26.384491</td>
          <td>0.145261</td>
          <td>25.378574</td>
          <td>0.053965</td>
          <td>25.053438</td>
          <td>0.066556</td>
          <td>24.777685</td>
          <td>0.098865</td>
          <td>24.488867</td>
          <td>0.171475</td>
          <td>0.048913</td>
          <td>0.048127</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.796015</td>
          <td>0.521360</td>
          <td>26.761379</td>
          <td>0.198746</td>
          <td>26.033909</td>
          <td>0.095538</td>
          <td>25.043600</td>
          <td>0.065442</td>
          <td>24.906812</td>
          <td>0.109825</td>
          <td>24.319920</td>
          <td>0.147265</td>
          <td>0.007378</td>
          <td>0.006294</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.507570</td>
          <td>0.163419</td>
          <td>26.309795</td>
          <td>0.124221</td>
          <td>26.457126</td>
          <td>0.227130</td>
          <td>25.803707</td>
          <td>0.240796</td>
          <td>27.945674</td>
          <td>1.787081</td>
          <td>0.098203</td>
          <td>0.060606</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.411723</td>
          <td>0.390553</td>
          <td>26.079032</td>
          <td>0.110760</td>
          <td>26.051124</td>
          <td>0.097003</td>
          <td>25.758724</td>
          <td>0.122668</td>
          <td>25.746471</td>
          <td>0.224989</td>
          <td>25.279848</td>
          <td>0.326824</td>
          <td>0.009975</td>
          <td>0.008067</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.692077</td>
          <td>0.490012</td>
          <td>26.795165</td>
          <td>0.208469</td>
          <td>26.462880</td>
          <td>0.141885</td>
          <td>26.162239</td>
          <td>0.177445</td>
          <td>26.321490</td>
          <td>0.365404</td>
          <td>25.800747</td>
          <td>0.497444</td>
          <td>0.095365</td>
          <td>0.067784</td>
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
          <td>26.577658</td>
          <td>0.414197</td>
          <td>26.828207</td>
          <td>0.191987</td>
          <td>25.924190</td>
          <td>0.077948</td>
          <td>25.122471</td>
          <td>0.062742</td>
          <td>24.854706</td>
          <td>0.094362</td>
          <td>24.058656</td>
          <td>0.105337</td>
          <td>0.080380</td>
          <td>0.047418</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.281812</td>
          <td>0.270298</td>
          <td>26.432724</td>
          <td>0.116884</td>
          <td>26.269884</td>
          <td>0.163823</td>
          <td>26.138721</td>
          <td>0.270609</td>
          <td>25.484742</td>
          <td>0.335064</td>
          <td>0.040014</td>
          <td>0.022696</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.403289</td>
          <td>2.130241</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.018454</td>
          <td>0.435444</td>
          <td>25.788058</td>
          <td>0.109078</td>
          <td>24.998220</td>
          <td>0.103669</td>
          <td>24.233820</td>
          <td>0.118771</td>
          <td>0.051929</td>
          <td>0.029082</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.564025</td>
          <td>0.760096</td>
          <td>27.660060</td>
          <td>0.359246</td>
          <td>26.243198</td>
          <td>0.178066</td>
          <td>25.652306</td>
          <td>0.199716</td>
          <td>24.950308</td>
          <td>0.240282</td>
          <td>0.119001</td>
          <td>0.078162</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.846527</td>
          <td>0.245634</td>
          <td>26.217807</td>
          <td>0.122921</td>
          <td>25.823111</td>
          <td>0.077903</td>
          <td>25.773766</td>
          <td>0.121930</td>
          <td>25.517496</td>
          <td>0.182416</td>
          <td>25.461052</td>
          <td>0.370529</td>
          <td>0.127233</td>
          <td>0.093150</td>
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
          <td>26.572668</td>
          <td>0.406441</td>
          <td>26.449317</td>
          <td>0.136128</td>
          <td>25.593817</td>
          <td>0.056773</td>
          <td>25.067962</td>
          <td>0.058274</td>
          <td>24.787117</td>
          <td>0.086795</td>
          <td>24.995534</td>
          <td>0.228945</td>
          <td>0.048913</td>
          <td>0.048127</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.439443</td>
          <td>0.359780</td>
          <td>26.741533</td>
          <td>0.170400</td>
          <td>25.965457</td>
          <td>0.076527</td>
          <td>25.170807</td>
          <td>0.061818</td>
          <td>24.795923</td>
          <td>0.084851</td>
          <td>24.154574</td>
          <td>0.108282</td>
          <td>0.007378</td>
          <td>0.006294</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.779306</td>
          <td>0.964883</td>
          <td>26.851767</td>
          <td>0.200278</td>
          <td>26.082810</td>
          <td>0.092058</td>
          <td>26.268442</td>
          <td>0.175199</td>
          <td>25.903776</td>
          <td>0.237677</td>
          <td>25.242093</td>
          <td>0.294118</td>
          <td>0.098203</td>
          <td>0.060606</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.748504</td>
          <td>0.205459</td>
          <td>26.243157</td>
          <td>0.110956</td>
          <td>26.181476</td>
          <td>0.092617</td>
          <td>25.821298</td>
          <td>0.109717</td>
          <td>25.733147</td>
          <td>0.190946</td>
          <td>25.200417</td>
          <td>0.263255</td>
          <td>0.009975</td>
          <td>0.008067</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.809844</td>
          <td>0.502038</td>
          <td>26.715586</td>
          <td>0.178946</td>
          <td>26.600596</td>
          <td>0.144841</td>
          <td>26.494085</td>
          <td>0.212443</td>
          <td>26.488383</td>
          <td>0.380910</td>
          <td>25.217625</td>
          <td>0.289101</td>
          <td>0.095365</td>
          <td>0.067784</td>
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
